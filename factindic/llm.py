"""Causal LM loading with T4-safe precision fallbacks, chat formatting, batched generation."""
from __future__ import annotations

import gc
import re
import time

import torch

from .config import IS_LOCAL, safe_attention

DEV = "cuda"


def load_tokenizer(model_id, padding_side="left"):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = padding_side
    return tok


def chat_text(tok, user, assistant=None):
    msgs = [{"role": "user", "content": user}]
    if assistant is not None:
        msgs.append({"role": "assistant", "content": assistant})
    kw = dict(tokenize=False, add_generation_prompt=assistant is None)
    try:
        return tok.apply_chat_template(msgs, enable_thinking=False, **kw)
    except TypeError:
        return tok.apply_chat_template(msgs, **kw)


def _bnb4(compute):
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=compute)


def _trials(model_id):
    if IS_LOCAL and torch.cuda.is_bf16_supported(including_emulation=False):     # RTX 40/50, 8 GB
        if "gemma" in model_id.lower():
            return [("bf16, per-layer embeddings on the CPU", "gemma_split")]
        return [("4bit-nf4/bf16", dict(quantization_config=_bnb4(torch.bfloat16), dtype=torch.bfloat16))]
    if "gemma" in model_id.lower():   # bf16-trained; fp16 compute overflows → fp32 compute (H1-proven)
        return [("4bit-nf4/fp16-storage/fp32-compute", dict(quantization_config=_bnb4(torch.float32), dtype=torch.float16)),
                ("4bit-nf4/fp32", dict(quantization_config=_bnb4(torch.float32), dtype=torch.float32))]
    if "llama" in model_id.lower() or "qwen" in model_id.lower():
        return [("fp16", dict(dtype=torch.float16)),
                ("4bit-nf4/fp32-compute", dict(quantization_config=_bnb4(torch.float32), dtype=torch.float16))]
    return [("fp32", dict(dtype=torch.float32))]


def _gemma_split(model_id, **kw):
    """Gemma-4 E2B in bf16 on an 8 GB card. Its per-layer embedding table (4.7 GB) stays in RAM, where looking
    up a few rows is cheap; the unused vision and audio towers are dropped. About 5 GB of GPU memory."""
    from transformers import AutoModelForCausalLM
    lm = "model.language_model."
    ple_name = lm + "embed_tokens_per_layer"
    dm = {lm + x: 0 for x in ("embed_tokens", "layers", "norm", "rotary_emb", "per_layer_model_projection", "per_layer_projection_norm")}
    dm.update({"lm_head": 0, ple_name: "cpu"}, **{f"model.{x}": "cpu" for x in ("vision_tower", "audio_tower", "embed_vision", "embed_audio")})
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map=dm, **kw)
        from accelerate.hooks import remove_hook_from_module
        remove_hook_from_module(model, recurse=True)      # plain modules again: accelerate would copy the table to the GPU
    except Exception as e:
        print("split loading failed, loading through RAM (needs ~11 GB):", repr(e)[:300], flush=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, **kw)
    inner = model.model
    for x in ("vision_tower", "audio_tower", "embed_vision", "embed_audio"):
        if hasattr(inner, x):
            setattr(inner, x, None)
    text = inner.language_model
    for name, child in text.named_children():
        child.to("cpu" if name == "embed_tokens_per_layer" else DEV)
    model.lm_head.to(DEV)
    if hasattr(model, "hf_device_map"):
        del model.hf_device_map                            # PEFT would otherwise dispatch the model again
    ple = text.embed_tokens_per_layer
    ple.register_forward_pre_hook(lambda m, a: (a[0].cpu(),))
    ple.register_forward_hook(lambda m, a, out: out.to(DEV))
    gc.collect(); torch.cuda.empty_cache()
    meta = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    assert not meta, f"weights not loaded: {meta[:5]}"
    return model


def load_causal(model_id, adapter=None, attn_implementation=None, trials=None):
    """Returns (model, tok, precision_desc). Keeps the first config whose logits are finite."""
    import transformers as tf
    safe_attention()
    tok = load_tokenizer(model_id)
    classes = [getattr(tf, c) for c in ("AutoModelForCausalLM", "AutoModelForImageTextToText",
                                        "AutoModelForMultimodalLM") if hasattr(tf, c)]
    probe = tok([chat_text(tok, "Language: Gujarati\nArticle: ગુજરાતમાં 23 એપ્રિલે મતદાન થશે.\nSummary:")],
                return_tensors="pt", add_special_tokens=False)
    last = None
    for desc, kw in (trials or _trials(model_id)):
        split = kw == "gemma_split"
        kw = {} if split else kw
        if attn_implementation:
            kw = dict(kw, attn_implementation=attn_implementation)
        for cls in (classes[:1] if split else classes):
            try:
                try:
                    if split:
                        model = _gemma_split(model_id, **kw)
                    else:
                        model = cls.from_pretrained(model_id, device_map={"": 0}, **kw)
                except TypeError:   # transformers 4.x uses torch_dtype
                    if split:
                        raise
                    kw2 = {("torch_dtype" if k == "dtype" else k): v for k, v in kw.items()}
                    model = cls.from_pretrained(model_id, device_map={"": 0}, **kw2)
            except Exception as e:
                last = e
                print(f"{cls.__name__} [{desc}] failed: {repr(e)[:300]}", flush=True)
                continue
            if adapter:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, adapter)
            model.eval()
            with torch.no_grad():
                lg = model(**{k: v.to(DEV) for k, v in probe.items()}).logits
            if torch.isfinite(lg).all():
                return model, tok, desc
            del model
            torch.cuda.empty_cache()
            break
    raise RuntimeError(f"could not load {model_id} with finite logits: {repr(last)[:300]}")


def clean_output(text):
    text = re.sub(r"(?s)^.*?</think>", "", text)
    text = re.sub(r"(?im)^\s*(summary|सारांश|સારાંશ)\s*[:：]\s*", "", text.strip())
    text = text.replace("**", "").replace("##", "")
    text = re.split(r"\n\s*\n", text.strip())[0]
    return re.sub(r"\s+", " ", text).strip()


def _eos(model, tok):
    from .decoding import eos_ids
    return eos_ids(model, tok)


@torch.no_grad()
def generate(model, tok, texts, max_new_tokens=160, do_sample=False, num_return_sequences=1,
             temperature=0.7, top_p=0.9, seed=None):
    """texts are already chat-templated. Returns (outputs[list[list[str]]], seconds, n_new_tokens).
    A batch that does not fit in memory is split in two."""
    args = dict(max_new_tokens=max_new_tokens, do_sample=do_sample, num_return_sequences=num_return_sequences,
                temperature=temperature, top_p=top_p, seed=seed)
    try:
        return _generate(model, tok, texts, **args)
    except torch.cuda.OutOfMemoryError:
        if len(texts) == 1:
            raise
    torch.cuda.empty_cache()
    mid = len(texts) // 2
    a, b = generate(model, tok, texts[:mid], **args), generate(model, tok, texts[mid:], **args)
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


def _generate(model, tok, texts, max_new_tokens, do_sample, num_return_sequences, temperature, top_p, seed):
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
    if seed is not None:
        torch.manual_seed(seed)
    kw = dict(max_new_tokens=max_new_tokens, do_sample=do_sample, pad_token_id=tok.pad_token_id,
              num_return_sequences=num_return_sequences, eos_token_id=_eos(model, tok))
    if do_sample:
        kw.update(temperature=temperature, top_p=top_p, top_k=0)
    else:
        kw.update(temperature=None, top_p=None, top_k=None)
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = model.generate(**enc, **kw)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t
    new = out[:, enc["input_ids"].shape[1]:]
    dec = tok.batch_decode(new, skip_special_tokens=True)
    n_new = int((new != tok.pad_token_id).sum())
    res = [[clean_output(dec[i * num_return_sequences + j]) for j in range(num_return_sequences)]
           for i in range(len(texts))]
    return res, dt, n_new


def n_tokens(tok, text):
    return len(tok(text, add_special_tokens=False)["input_ids"])


def lm_parts(model):
    """(decoder layers count, final norm, lm_head, softcap) for early-exit (DoLa) projections."""
    lm_head = model.get_output_embeddings()
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    norm = None
    for path in ("model.norm", "model.language_model.norm", "language_model.model.norm",
                 "model.text_model.norm", "language_model.norm"):
        obj = base
        try:
            for p in path.split("."):
                obj = getattr(obj, p)
            norm = obj
            break
        except AttributeError:
            continue
    cfg = getattr(base.config, "text_config", base.config)
    n_layers = cfg.num_hidden_layers
    softcap = getattr(cfg, "final_logit_softcapping", None)
    if norm is None:
        raise RuntimeError("final norm not found; DoLa unavailable for this architecture")
    return n_layers, norm, lm_head, softcap
