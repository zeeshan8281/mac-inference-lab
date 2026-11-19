from __future__ import annotations

import dataclasses
import gc
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import make_prompt_cache
from .core import episode_metrics, feature_key, score

GEN = importlib.import_module("mlx_lm.generate")
GIB = 1024 ** 3


class Engine:
    def __init__(self, target_path, draft_path):
        self.target_path, self.draft_path = str(Path(target_path).resolve()), str(Path(draft_path).resolve())
        self.model, self.tokenizer = load(self.target_path)
        self.draft, self.draft_tokenizer = load(self.draft_path)
        vocab = self.tokenizer.get_vocab()
        if vocab != self.draft_tokenizer.get_vocab():
            raise ValueError("Target and draft token-to-ID mappings differ.")
        if set(vocab.values()) != set(range(len(vocab))):
            raise ValueError("This implementation requires contiguous actual token IDs.")
        self.vocab_size = len(vocab)
        self.vocab_hash = hashlib.sha256(json.dumps(vocab, sort_keys=True).encode()).hexdigest()
        self.eos = sorted(self.tokenizer.eos_token_ids)
        if set(self.eos) != set(self.draft_tokenizer.eos_token_ids):
            raise ValueError("Target/draft EOS IDs differ.")
        self.versions = {p: importlib.metadata.version(p) for p in ["mlx", "mlx-lm", "transformers", "numpy"]}
        # This cache is reusable allocator memory, not model/context memory.
        mx.set_cache_limit(2 * GIB)
        mx.eval(self.model.parameters(), self.draft.parameters())
        self.resident_active_gib = mx.get_active_memory() / GIB

    def mask_padding(self, tokens, logits):
        """Both models mask only non-token padding rows, including the baseline."""
        return mx.where(mx.arange(logits.shape[-1]) < self.vocab_size, logits, -float("inf"))

    def tokenize(self, prompt):
        return self.tokenizer.apply_chat_template(
            [{"role": "system", "content": "Follow the requested output schema exactly."},
             {"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True,
        )

    def manifest(self):
        files = {}
        for name, path in [("target", self.target_path), ("draft", self.draft_path)]:
            files[name] = {}
            for f in sorted(Path(path).glob("*")):
                if f.is_file() and (f.suffix in [".json", ".safetensors"]):
                    h = hashlib.sha256()
                    with f.open("rb") as stream:
                        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                            h.update(block)
                    files[name][f.name] = {"bytes": f.stat().st_size, "sha256": h.hexdigest()}
        source_file = Path(inspect.getfile(GEN))
        return {
            "versions": self.versions, "device": mx.device_info(),
            "target_path": self.target_path, "draft_path": self.draft_path,
            "model_files": files, "tokenizer_mapping_sha256": self.vocab_hash,
            "actual_vocab_size": self.vocab_size, "eos_ids": self.eos,
            "resident_active_gib": self.resident_active_gib,
            "generation_source_sha256": hashlib.sha256(source_file.read_bytes()).hexdigest(),
            "decoding": "greedy; unused padded vocabulary rows masked in every configuration",
            "model_residency": "Target and draft remain resident for all comparisons, including non-draft baselines.",
            "capabilities": {
                "serial": True, "serial_kv4_kv8": True, "serial_greedy_speculation": True,
                "batch_plain": True, "batch_quantized_kv": False, "batch_speculation": False,
                "unsupported_reason": "Installed BatchGenerator API accepts neither kv_bits nor draft_model; combinations are rejected, never silently emulated.",
            },
        }

    def run(self, tasks, config, *, deadline=60.0, phase="probe", profile="", replicate=0,
            memory_ceiling_gib=32.0, hard_timeout=300.0, token_callback=None):
        if config.mode == "batch" and (config.kv_bits or config.draft_tokens):
            raise ValueError("Unsupported composition: BatchGenerator has no KV-quantization/draft arguments.")
        encoded = [self.tokenize(t["prompt"]) for t in tasks]
        gc.collect()
        mx.clear_cache()
        mx.reset_peak_memory()
        records = [{
            "id": t["id"], "prompt_tokens": len(tokens), "tokens": [], "token_times_s": [],
            "ttft_s": None, "finish_s": None, "text": "", "correct": False,
            "draft_accepted_tokens": 0, "finish_reason": None,
        } for t, tokens in zip(tasks, encoded)]
        started = time.perf_counter()
        max_observed_batch = 0
        cache_types = set()
        error = None

        def observe(i, token, from_draft=False):
            now = time.perf_counter() - started
            row = records[i]
            if token not in self.eos:
                if row["ttft_s"] is None:
                    row["ttft_s"] = now
                row["tokens"].append(int(token))
                row["token_times_s"].append(now)
                row["draft_accepted_tokens"] += int(from_draft)
                if token_callback:
                    token_callback(row["id"], int(token), now)
            if now > hard_timeout:
                raise TimeoutError("Episode exceeded hard timeout at a token boundary.")
            if mx.get_active_memory() / GIB > memory_ceiling_gib:
                raise MemoryError("Active MLX memory exceeded the soft guard at a token boundary.")

        try:
            if config.mode == "serial":
                for i, (task, tokens) in enumerate(zip(tasks, encoded)):
                    caches = make_prompt_cache(self.model)
                    if config.draft_tokens:
                        caches += make_prompt_cache(self.draft)
                    options = dict(
                        sampler=GEN.greedy_sampler, logits_processors=[self.mask_padding],
                        prompt_cache=caches, prefill_step_size=config.prefill_step,
                        kv_bits=config.kv_bits, kv_group_size=64, quantized_kv_start=0,
                    )
                    if config.draft_tokens:
                        options["num_draft_tokens"] = config.draft_tokens
                    stream = stream_generate(
                        self.model, self.tokenizer, tokens, max_tokens=task["max_tokens"],
                        draft_model=self.draft if config.draft_tokens else None, **options,
                    )
                    try:
                        for response in stream:
                            observe(i, response.token, response.from_draft)
                            records[i]["finish_reason"] = response.finish_reason
                    finally:
                        stream.close()
                    records[i]["text"] = self.tokenizer.decode(records[i]["tokens"], skip_special_tokens=True)
                    records[i]["finish_s"] = time.perf_counter() - started
                    cache_types.update(type(c).__name__ for c in caches)
                    del caches
                max_observed_batch = 1
            else:
                batch = GEN.BatchGenerator(
                    self.model, max_tokens=max(t["max_tokens"] for t in tasks),
                    stop_tokens=[[t] for t in self.eos], sampler=GEN.greedy_sampler,
                    logits_processors=[self.mask_padding],
                    completion_batch_size=config.batch_size,
                    prefill_batch_size=config.batch_size,
                    prefill_step_size=config.prefill_step,
                )
                try:
                    uids = batch.insert(encoded, max_tokens=[t["max_tokens"] for t in tasks])
                    indices = dict(zip(uids, range(len(tasks))))
                    unfinished = set(uids)
                    while unfinished:
                        prompt_events, responses = batch.next()
                        mx.synchronize()
                        max_observed_batch = max(max_observed_batch, len(responses))
                        for response in responses:
                            i = indices[response.uid]
                            observe(i, response.token)
                            if response.finish_reason:
                                records[i]["text"] = self.tokenizer.decode(records[i]["tokens"], skip_special_tokens=True)
                                records[i]["finish_s"] = time.perf_counter() - started
                                records[i]["finish_reason"] = response.finish_reason
                                unfinished.discard(response.uid)
                        if time.perf_counter() - started > hard_timeout:
                            raise TimeoutError("Batch exceeded hard timeout.")
                finally:
                    batch.close()
                cache_types.add("BatchKVCache (unquantized)")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        mx.synchronize()
        elapsed = time.perf_counter() - started
        peak = mx.get_peak_memory() / GIB
        for task, row in zip(tasks, records):
            if row["finish_s"] is None:
                row["text"] = self.tokenizer.decode(row["tokens"], skip_special_tokens=True)
            row["finish_s"] = row["finish_s"] if row["finish_s"] is not None else elapsed
            row["correct"] = score(row["text"], task["expected"]) if "expected" in task else None
            gaps = [b - a for a, b in zip(row["token_times_s"], row["token_times_s"][1:])]
            row["max_token_gap_s"] = max(gaps, default=0)
            row["output_tokens"] = len(row["tokens"])
            if error and row["finish_reason"] is None:
                row["correct"] = False
                row["error"] = error
        return {
            "phase": phase, "profile": profile, "replicate": replicate,
            "configuration": config.name, "configuration_parameters": dataclasses.asdict(config),
            "feature": feature_key([len(t) for t in encoded]),
            "prompt_lengths": [len(t) for t in encoded], "deadline_s": deadline,
            "metrics": episode_metrics(records, elapsed, deadline),
            "peak_active_gib": peak, "allocator_cache_gib": mx.get_cache_memory() / GIB,
            "effective": {"max_observed_decode_batch": max_observed_batch,
                          "cache_types": sorted(cache_types), "quantized_kv_start": 0 if config.kv_bits else None},
            "records": records, "error": error,
        }
