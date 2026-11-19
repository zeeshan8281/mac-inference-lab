from __future__ import annotations

import hashlib
import json
import random
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Configuration:
    name: str
    mode: str = "serial"
    batch_size: int = 1
    prefill_step: int = 512
    kv_bits: int | None = None
    draft_tokens: int = 0


CONFIGURATIONS = [
    Configuration("serial"),
    Configuration("kv8", kv_bits=8),
    Configuration("kv4", kv_bits=4),
    Configuration("draft2", draft_tokens=2),
    Configuration("batch4_chunk256", "batch", 4, 256),
    Configuration("batch4_chunk1024", "batch", 4, 1024),
]
CONFIG_BY_NAME = {c.name: c for c in CONFIGURATIONS}
PROFILES = ("solo_short", "solo_long", "burst_short", "burst_mixed")


def dump(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def feature_key(prompt_lengths):
    """Only pre-inference workload information; never answers or task IDs."""
    queue = "solo" if len(prompt_lengths) == 1 else "burst"
    size = "short" if max(prompt_lengths) <= 512 else "long"
    return f"{queue}_{size}"


def make_tasks(split, replicate, profile):
    """Controlled extraction fixtures, not a general intelligence benchmark."""
    seed = int(hashlib.sha256(f"{split}:{replicate}:{profile}".encode()).hexdigest()[:12], 16)
    rng = random.Random(seed)
    requests = []
    for index in range(1 if profile.startswith("solo") else 4):
        long = profile == "solo_long" or (profile == "burst_mixed" and index % 2 == 1)
        records = []
        for _ in range(100 if long else 8):
            records.append({
                "ref": f"R{rng.randrange(100000, 999999)}",
                "owner": rng.choice(["Asha", "Noah", "Lina", "Omar", "Maya", "Iris"]),
                "units": rng.randrange(1, 80),
                "status": rng.choice(["open", "closed", "pending"]),
            })
        # IDs must be unique, or the answer would be ambiguous.
        for i, record in enumerate(records):
            record["ref"] = f"{split[:1].upper()}{replicate:02d}{index}{i:03d}{rng.randrange(10, 99)}"
        selected = rng.sample(records, 3)
        fields = ["ref", "owner", "units", "status"]
        expected = [{k: r[k] for k in fields} for r in selected]
        prompt = (
            "Read this register. Return ONLY a JSON array of the three requested records, "
            "in the requested order. Each object must contain ref, owner, units (an integer), "
            "and status. Copy values exactly. Do not add commentary or markdown.\n"
            "REGISTER:\n" + "\n".join(json.dumps(r, separators=(",", ":")) for r in records)
            + "\nREQUESTED REFS: " + ", ".join(r["ref"] for r in selected)
        )
        requests.append({
            "id": f"{split}-{replicate}-{profile}-{index}",
            "prompt": prompt, "expected": expected, "max_tokens": 192,
        })
    return requests


def score(text, expected):
    """Strict value-level scoring. Fences allowed; duplicate keys rejected."""
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result:
                raise ValueError("duplicate key")
            result[k] = v
        return result
    candidate = text.strip()
    if candidate.startswith("```") and candidate.endswith("```"):
        candidate = candidate.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        parsed = json.loads(candidate, object_pairs_hook=pairs)
    except (ValueError, TypeError):
        return False
    # Python considers True == 1 and 1.0 == 1; the schema requires integer units.
    return (
        parsed == expected and isinstance(parsed, list)
        and all(type(row.get("units")) is int for row in parsed if isinstance(row, dict))
    )


def episode_metrics(records, elapsed, deadline):
    correct = sum(bool(r.get("correct")) for r in records)
    on_time = sum(bool(r.get("correct")) and r["finish_s"] <= deadline for r in records)
    return {
        "requests": len(records), "correct": correct, "on_time_correct": on_time,
        "accuracy": correct / len(records) if records else 0,
        "elapsed_s": elapsed,
        "goodput_per_min": 60 * on_time / elapsed if elapsed > 0 else 0,
        "mean_completion_s": statistics.mean(r["finish_s"] for r in records) if records else 0,
        "mean_ttft_s": statistics.mean(r["ttft_s"] for r in records if r.get("ttft_s") is not None)
        if any(r.get("ttft_s") is not None for r in records) else None,
    }


def train_policy(episodes, memory_ceiling_gib=32.0):
    calibration = [e for e in episodes if e["phase"] == "calibration"]
    if not calibration:
        raise ValueError("No successful calibration episodes.")
    keys = sorted({e["feature"] for e in calibration})
    table = {}
    for key in keys:
        table[key] = {}
        for cfg in CONFIGURATIONS:
            rows = [e for e in calibration if e["feature"] == key and e["configuration"] == cfg.name]
            if not rows:
                continue
            table[key][cfg.name] = {
                "accuracy": sum(e["metrics"]["correct"] for e in rows) / sum(e["metrics"]["requests"] for e in rows),
                "goodput_per_min": statistics.mean(e["metrics"]["goodput_per_min"] for e in rows),
                "mean_completion_s": statistics.mean(e["metrics"]["mean_completion_s"] for e in rows),
                "peak_active_gib": max(e["peak_active_gib"] for e in rows),
                "samples": len(rows),
                "max_prompt_tokens": max(max(e["prompt_lengths"]) for e in rows),
                "max_queue": max(len(e["prompt_lengths"]) for e in rows),
                "errors": sum(bool(e.get("error")) for e in rows),
            }
    # Accuracy cannot be traded away relative to the serial baseline in a state.
    eligible = {}
    for key, rows in table.items():
        if "serial" not in rows:
            raise ValueError(f"Missing serial baseline for {key}")
        if rows["serial"]["errors"]:
            raise ValueError(f"Serial baseline failed for {key}; resolve before training a policy.")
        eligible[key] = [
            name for name, row in rows.items()
            if row["accuracy"] + 1e-9 >= rows["serial"]["accuracy"]
            and row["peak_active_gib"] * 1.15 <= memory_ceiling_gib
            and not row["errors"]
        ]
        if not eligible[key]:
            raise ValueError(f"No quality/memory eligible configuration for {key}")
    fixed_candidates = set.intersection(*(set(v) for v in eligible.values()))
    if not fixed_candidates:
        raise ValueError("No fixed configuration is eligible in every state.")
    def global_rank(name):
        return (
            statistics.mean(table[k][name]["goodput_per_min"] for k in keys),
            -statistics.mean(table[k][name]["mean_completion_s"] for k in keys),
        )
    fixed = max(sorted(fixed_candidates), key=global_rank)
    choices = {}
    for key in keys:
        rows = table[key]
        winner = max(sorted(eligible[key]), key=lambda n: (rows[n]["goodput_per_min"], -rows[n]["mean_completion_s"]))
        # Prefer the fixed policy when a small apparent difference could be noise.
        if rows[winner]["goodput_per_min"] < rows[fixed]["goodput_per_min"] * 1.03:
            winner = fixed
        choices[key] = winner
    return {
        "schema_version": 1, "training_phase": "calibration",
        "features": ["queue_count", "max_prompt_tokens"],
        "choices": choices, "best_fixed": fixed, "table": table,
        "eligible": eligible, "memory_ceiling_gib": memory_ceiling_gib,
        "memory_margin": 1.15,
        "scope": "Selection at drained-batch boundaries; fixed resident models; empirical soft admission guard.",
    }


def select_configuration(policy, prompt_lengths, budget_gib=None):
    key = feature_key(prompt_lengths)
    budget = min(policy["memory_ceiling_gib"], budget_gib if budget_gib is not None else policy["memory_ceiling_gib"])
    if key not in policy["table"]:
        raise ValueError(f"Workload state {key} was not calibrated.")
    rows = policy["table"][key]
    allowed = [
        name for name in policy["eligible"][key]
        if rows[name]["peak_active_gib"] * policy["memory_margin"] <= budget
        and max(prompt_lengths) <= rows[name]["max_prompt_tokens"] * 1.25
        and len(prompt_lengths) <= rows[name]["max_queue"]
    ]
    if not allowed:
        raise ValueError("No calibrated configuration fits this queue, prompt length and soft memory budget.")
    preferred = policy["choices"][key]
    if preferred not in allowed:
        preferred = max(allowed, key=lambda n: (rows[n]["goodput_per_min"], -rows[n]["mean_completion_s"]))
    return CONFIG_BY_NAME[preferred]


def percentile(values, p):
    values = sorted(values)
    if not values:
        return None
    return values[round((len(values) - 1) * p)]
