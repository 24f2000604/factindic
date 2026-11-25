"""Faithfulness-weighted QLoRA SFT (mFACT-style loss weighting w_i ∈ [0.3, 1.0]), resumable."""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch


def build_examples(tok, rows, prompt_fn, max_len):
    """rows: dicts with prompt args + 'summary' + 'w'. Labels mask the prompt; article shrinks to fit."""
    from .llm import chat_text
    ex = []
    for r in rows:
        art = r["article"].split()
        while True:
            prompt = chat_text(tok, prompt_fn(r, " ".join(art)))
            full = chat_text(tok, prompt_fn(r, " ".join(art)), assistant=r["summary"])
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            f_ids = tok(full, add_special_tokens=False)["input_ids"]
            if len(f_ids) <= max_len or len(art) < 60:
                break
            art = art[: int(len(art) * 0.85)]
        if f_ids[:len(p_ids)] != p_ids:      # template adds something after the turn start: append the answer instead
            tail = full[full.rfind(r["summary"].strip()):]
            f_ids = p_ids + tok(tail, add_special_tokens=False)["input_ids"]
        f_ids = f_ids[:max_len]
        labels = [-100] * min(len(p_ids), len(f_ids)) + f_ids[len(p_ids):]
        ex.append((f_ids, labels[: len(f_ids)], float(r.get("w", 1.0))))
    return ex


def train_lora(model_id, examples, out_dir, cfg, seed=42):
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from transformers import get_cosine_schedule_with_warmup
    from .llm import load_causal

    out_dir = Path(out_dir)
    final, ckpt = out_dir / "final", out_dir / "ckpt"
    if (final / "adapter_config.json").exists():
        return final
    model, tok, desc = load_causal(model_id)
    # Like prepare_model_for_kbit_training, but Gemma-4's per-layer embedding (2.3 B values) stays in half
    # precision: in fp32 it alone would need 9 GB. The vision and audio towers are not used.
    for n, p in model.named_parameters():
        p.requires_grad = False
        if not desc.startswith("bf16") and p.dtype in (torch.float16, torch.bfloat16) and p.__class__.__name__ != "Params4bit" and \
                not any(x in n for x in ("embed_tokens_per_layer", "vision", "audio")):
            p.data = p.data.to(torch.float32)
    torch.cuda.empty_cache()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    # text layers only: the towers also have layers called q_proj
    targets = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj)" if "gemma-4" in model_id.lower() else \
        ["q_proj", "k_proj", "v_proj", "o_proj"]
    model = get_peft_model(model, LoraConfig(r=cfg["r"], lora_alpha=cfg["alpha"], lora_dropout=cfg["dropout"],
                                             target_modules=targets, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    order = list(range(len(examples)))
    random.Random(seed).shuffle(order)
    mbs, acc = cfg["micro_bs"], cfg["accum"]
    spe = math.ceil(len(examples) / (mbs * acc))
    total = spe * cfg["epochs"]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.03 * total)), total)
    step = 0
    if (ckpt / "state.pt").exists():
        from safetensors.torch import load_file
        set_peft_model_state_dict(model, load_file(str(ckpt / "adapter_model.safetensors")))
        st = torch.load(ckpt / "state.pt", weights_only=False)
        opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"]); step = st["step"]
        print("resumed SFT at step", step)
    pad = tok.pad_token_id

    def batch(sel):
        # left padding: every summary ends at the last position, so logits are only needed for the
        # final K positions (Gemma-4's 262k vocab makes full-sequence fp32 logits ~2 GB per batch)
        L = max(len(examples[i][0]) for i in sel)
        inp = torch.full((len(sel), L), pad); lab = torch.full((len(sel), L), -100)
        att = torch.zeros((len(sel), L), dtype=torch.long); w = torch.zeros(len(sel))
        for k, i in enumerate(sel):
            a, b, wi = examples[i]
            inp[k, L - len(a):] = torch.tensor(a); lab[k, L - len(b):] = torch.tensor(b)
            att[k, L - len(a):] = 1; w[k] = wi
        K = int((lab != -100).sum(1).max()) + 1
        return inp.cuda(), att.cuda(), lab.cuda(), w.cuda(), min(K, L)

    model.train()
    t0, hist, bad = time.time(), [], 0
    while step < total:
        base = (step % spe) * mbs * acc
        opt.zero_grad(set_to_none=True)
        step_loss, ok = 0.0, True
        for a in range(acc):
            sel = order[base + a * mbs: base + (a + 1) * mbs]
            if not sel:
                continue
            inp, att, lab, w, K = batch(sel)
            try:
                logits = model(input_ids=inp, attention_mask=att, logits_to_keep=K, use_cache=False).logits
            except TypeError:
                logits = model(input_ids=inp, attention_mask=att, use_cache=False).logits[:, -K:]
            logits = logits[:, :-1].float()
            tgt = lab[:, lab.shape[1] - K + 1:]
            ce = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                                                   ignore_index=-100, reduction="none").view(tgt.shape)
            mask = (tgt != -100).float()
            per_ex = (ce * mask).sum(1) / mask.sum(1).clamp(min=1)
            loss = (per_ex * w).sum() / w.sum().clamp(min=1e-6) / acc   # faithfulness-weighted CE
            if not torch.isfinite(loss):
                ok = False
                break
            loss.backward()
            step_loss += loss.item()
        if not ok:
            bad += 1
            print(f"non-finite loss at step {step} (skipped); consecutive={bad}")
            if bad >= 5:
                raise RuntimeError("SFT diverged (5 consecutive non-finite steps)")
            step += 1
            sched.step()
            continue
        bad = 0
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); step += 1
        hist.append(step_loss)
        if step % 10 == 0:
            print(f"step {step}/{total} loss {np.mean(hist[-10:]):.4f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        if step % 50 == 0 or step == total:
            model.save_pretrained(ckpt)
            torch.save(dict(opt=opt.state_dict(), sched=sched.state_dict(), step=step), ckpt / "state.pt")
            with open(out_dir / "loss_history.jsonl", "a") as f:
                f.write(json.dumps(dict(step=step, loss=float(np.mean(hist[-10:])))) + "\n")
    model.save_pretrained(final)
    with open(out_dir / "train_info.json", "w") as f:
        json.dump(dict(model=model_id, precision=desc, steps=total, n_examples=len(examples),
                       final_loss=float(np.mean(hist[-20:])) if hist else None), f)
    del model, opt
    return final
