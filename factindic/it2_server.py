"""IndicTrans2 worker. Runs inside its own pinned environment (see it2_env.py), never imported by the
package. Protocol: one JSON object per line on stdin/stdout; everything libraries print goes to stderr."""
import json
import os
import sys

_out = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
os.dup2(2, 1)                       # library prints must not corrupt the protocol stream


def send(obj):
    _out.write(json.dumps(obj, ensure_ascii=False) + "\n")
    _out.flush()


def main(model_id):
    import torch
    from IndicTransToolkit.processor import IndicProcessor
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=torch.float16, attn_implementation="eager").cuda().eval()
    send(dict(ready=True, torch=torch.__version__, model=model_id))

    @torch.no_grad()
    def run(sents, src, retry):
        ip = IndicProcessor(inference=True)      # fresh per batch: it queues placeholders between pre and post
        batch = ip.preprocess_batch(sents, src_lang=src, tgt_lang="eng_Latn")
        enc = tok(batch, truncation=True, padding="longest", max_length=256, return_tensors="pt",
                  return_attention_mask=True).to("cuda")
        out = model.generate(**enc, use_cache=True, min_length=0, max_length=256, num_beams=4,
                             num_return_sequences=1, **(dict(no_repeat_ngram_size=4) if retry else {}))
        dec = tok.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return ip.postprocess_batch(dec, lang="eng_Latn")

    def safe(sents, src, retry):
        try:
            return run(sents, src, retry)
        except torch.cuda.OutOfMemoryError:      # not stored: its traceback would keep the failed batch on the GPU
            if len(sents) == 1:
                raise
        torch.cuda.empty_cache()
        mid = len(sents) // 2
        return safe(sents[:mid], src, retry) + safe(sents[mid:], src, retry)

    for line in open(0, encoding="utf-8", closefd=False):
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            send(dict(out=safe(req["sents"], req["src"], req.get("retry", False))))
        except Exception as e:
            send(dict(error=repr(e)[:1500]))


if __name__ == "__main__":
    try:
        main(sys.argv[1])
    except Exception as e:
        import traceback
        traceback.print_exc()
        send(dict(fatal=repr(e)[:1500]))
        sys.exit(1)
