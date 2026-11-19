"""Post-pilot diagnostic: native MLX-LM generation, without the lab engine."""
import argparse
import hashlib
import importlib
import inspect
import json
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import make_prompt_cache

from inference_lab.core import make_tasks, score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Control output already exists.")
    model, tokenizer = load(args.model)
    gen = importlib.import_module("mlx_lm.generate")
    records = []
    tasks = make_tasks("warmup", 0, "burst_short")
    for bits, start in [(None, 0), (8, 0), (4, 0), (4, 5000)]:
        for task in tasks:
            prompt = tokenizer.apply_chat_template(
                [{"role": "system", "content": "Follow the requested output schema exactly."},
                 {"role": "user", "content": task["prompt"]}],
                tokenize=True, add_generation_prompt=True,
            )
            cache = make_prompt_cache(model)
            tokens = []
            started = time.perf_counter()
            for response in stream_generate(
                model, tokenizer, prompt, max_tokens=192,
                sampler=gen.greedy_sampler, prompt_cache=cache,
                kv_bits=bits, kv_group_size=64, quantized_kv_start=start,
                prefill_step_size=512,
            ):
                if response.token not in tokenizer.eos_token_ids:
                    tokens.append(int(response.token))
            mx.synchronize()
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            row = {
                "task_id": task["id"], "kv_bits": bits, "quantized_kv_start": start,
                "cache_types": sorted({type(c).__name__ for c in cache}),
                "text": text, "tokens": tokens, "correct": score(text, task["expected"]),
                "elapsed_s_diagnostic_only": time.perf_counter() - started,
            }
            records.append(row)
            print(f"Native control bits={bits} start={start} correct={row['correct']}", flush=True)
    result = {
        "scope": "Post-pilot diagnostic; direct MLX-LM stream_generate, no lab Engine, no padding mask, no draft model. Not used for policy selection or the primary result.",
        "model": str(Path(args.model).resolve()),
        "generation_source_sha256": hashlib.sha256(Path(inspect.getfile(gen)).read_bytes()).hexdigest(),
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
