from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import random
import statistics
import time
from pathlib import Path

from .core import (CONFIGURATIONS, CONFIG_BY_NAME, PROFILES, dump, make_tasks,
                   select_configuration, sha, train_policy)


def parser():
    p = argparse.ArgumentParser(description="Measure MLX configurations, freeze a controller, evaluate held-out workloads.")
    root = Path(__file__).resolve().parents[3]
    p.add_argument("--target", default=str(root / "work/models/qwen-coder-7b"))
    p.add_argument("--draft", default=str(root / "work/models/qwen-coder-1.5b"))
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("pilot")
    run.add_argument("--out", required=True)
    run.add_argument("--calibration-reps", type=int, default=3)
    run.add_argument("--test-reps", type=int, default=8)
    run.add_argument("--memory-ceiling-gib", type=float, default=32.0)
    run.add_argument("--max-hours", type=float, default=4.0)
    serve = sub.add_parser("serve")
    serve.add_argument("--policy", required=True)
    serve.add_argument("--port", type=int, default=8766)
    serve.add_argument("--microbatch-ms", type=float, default=40)
    return p


def summarize(episodes, policy):
    test = [e for e in episodes if e["phase"] == "test"]
    summary = {"best_fixed": policy["best_fixed"], "choices": policy["choices"], "profiles": {}}
    pairs = []
    for profile in PROFILES:
        group = {}
        for mode in ["fixed", "adaptive"]:
            rows = [e for e in test if e["profile"] == profile and e["policy_mode"] == mode]
            if not rows:
                continue
            group[mode] = {
                "episodes": len(rows), "requests": sum(e["metrics"]["requests"] for e in rows),
                "correct": sum(e["metrics"]["correct"] for e in rows),
                "on_time_correct": sum(e["metrics"]["on_time_correct"] for e in rows),
                "mean_goodput_per_min": statistics.mean(e["metrics"]["goodput_per_min"] for e in rows),
                "mean_completion_s": statistics.mean(e["metrics"]["mean_completion_s"] for e in rows),
                "mean_ttft_s": statistics.mean(e["metrics"]["mean_ttft_s"] for e in rows if e["metrics"]["mean_ttft_s"] is not None),
                "peak_active_gib": max(e["peak_active_gib"] for e in rows),
            }
        summary["profiles"][profile] = group
        for rep in {e["replicate"] for e in test if e["profile"] == profile}:
            match = {e["policy_mode"]: e for e in test if e["profile"] == profile and e["replicate"] == rep}
            if set(match) == {"fixed", "adaptive"}:
                pairs.append({
                    "profile": profile, "replicate": rep,
                    "fixed": match["fixed"]["metrics"]["goodput_per_min"],
                    "adaptive": match["adaptive"]["metrics"]["goodput_per_min"],
                })
    if pairs:
        # Paired resampling within workload strata preserves the fixed workload mix.
        rng = random.Random(219)
        draws = []
        groups = [[x for x in pairs if x["profile"] == profile] for profile in PROFILES]
        groups = [g for g in groups if g]
        for _ in range(5000):
            sample = [rng.choice(g) for g in groups for _ in range(len(g))]
            draws.append(statistics.mean(x["adaptive"] - x["fixed"] for x in sample))
        draws.sort()
        fixed = statistics.mean(p["fixed"] for p in pairs)
        adaptive = statistics.mean(p["adaptive"] for p in pairs)
        summary["paired_result"] = {
            "blocks": len(pairs), "fixed_mean_goodput": fixed, "adaptive_mean_goodput": adaptive,
            "difference_goodput": adaptive - fixed,
            "relative_change_percent": 100 * (adaptive / fixed - 1) if fixed else None,
            "difference_95ci": [draws[125], draws[4874]],
            "scope": "Exploratory stratified paired bootstrap over controlled workload blocks; not a claim about arbitrary workloads.",
        }
    summary["test_errors"] = [e["error"] for e in test if e.get("error")]
    return summary


def pilot(args):
    from .mlx_backend import Engine
    if args.calibration_reps < 1 or args.test_reps < 1:
        raise ValueError("Replicate counts must be positive.")
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output directory must be new/empty; completed runs are never overwritten.")
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    fixtures = {
        phase: {
            f"{rep}:{profile}": make_tasks(phase, rep, profile)
            for rep in range(count) for profile in PROFILES
        }
        for phase, count in [("calibration", args.calibration_reps), ("test", args.test_reps)]
    }
    dump(out / "fixtures.json", fixtures)
    code_hashes = {str(f.relative_to(Path(__file__).parent)): sha(f) for f in Path(__file__).parent.glob("*.py")}
    protocol = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "configurations": [dataclasses.asdict(c) for c in CONFIGURATIONS],
        "calibration_reps": args.calibration_reps, "test_reps": args.test_reps,
        "fixture_sha256": sha(out / "fixtures.json"), "code_sha256": code_hashes,
        "memory_ceiling_gib": args.memory_ceiling_gib, "max_hours": args.max_hours,
        "objective": "Correct responses completed before the fixed per-profile deadline, per episode minute.",
        "deadline_rule": "1.25 times each profile's serial warmup episode duration, minimum 2 seconds; frozen before calibration.",
        "arrival_model": "Finite closed bursts: all requests queued together; queue time is included in TTFT and completion.",
        "timing_scope": "Warm resident models; tokenization and allocator cache clearing excluded; all inference, within-burst queueing and stream generation included.",
        "cache_policy": "Fresh request KV caches; allocator cache cleared between episodes; no prefix reuse.",
        "policy_scope": "Configuration changes only at empty-engine batch boundaries; no mid-generation cache conversion.",
        "test_selection": "Best fixed and per-state adaptive choices frozen from calibration; test scores are never used for policy selection.",
        "analysis": "Pair fixed/adaptive by held-out profile and replicate, counterbalance order; bootstrap paired blocks within profile.",
    }
    dump(out / "protocol.json", protocol)
    print("Loading local target and draft models...", flush=True)
    engine = Engine(args.target, args.draft)
    dump(out / "environment.json", engine.manifest())
    episodes = []

    def run_one(tasks, cfg, **kw):
        if time.monotonic() - started > args.max_hours * 3600:
            raise TimeoutError("Pilot wall-clock budget exhausted; partial JSONL remains saved.")
        result = engine.run(tasks, cfg, memory_ceiling_gib=args.memory_ceiling_gib, **kw)
        result["sequence"] = len(episodes)
        result["completed_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        episodes.append(result)
        with (out / "episodes.jsonl").open("a") as f:
            f.write(json.dumps(result) + "\n")
        print(
            f"{len(episodes):03d} {result['phase']:11} {result['profile']:12} "
            f"{cfg.name:18} {result['metrics']['elapsed_s']:6.2f}s "
            f"correct={result['metrics']['correct']}/{len(tasks)} "
            f"peak={result['peak_active_gib']:.2f}GiB"
            + (f" ERROR {result['error']}" if result["error"] else ""), flush=True,
        )
        return result

    # Warm each execution path and verify the capability claims against real GPU work.
    probes = {}
    for cfg in CONFIGURATIONS:
        probes[cfg.name] = run_one(make_tasks("warmup", 0, "burst_short"), cfg,
                                  phase="warmup", profile="burst_short")
        if probes[cfg.name]["error"]:
            raise RuntimeError(f"Execution-path probe failed for {cfg.name}: {probes[cfg.name]['error']}")
    serial_tokens = [r["tokens"] for r in probes["serial"]["records"]]
    gates = {
        "greedy_draft_matches_serial": serial_tokens == [r["tokens"] for r in probes["draft2"]["records"]],
        "plain_batch_outputs_correct": all(r["correct"] for c in CONFIGURATIONS if c.mode == "batch" for r in probes[c.name]["records"]),
        "quantized_cache_observed": all("QuantizedKVCache" in probes[n]["effective"]["cache_types"] for n in ["kv4", "kv8"]),
        "real_batch_observed": all(probes[c.name]["effective"]["max_observed_decode_batch"] > 1 for c in CONFIGURATIONS if c.mode == "batch"),
    }
    dump(out / "capability_checks.json", gates)
    if not all(gates.values()):
        raise RuntimeError(f"Capability/correctness checks failed: {gates}")
    deadlines = {}
    for profile in PROFILES:
        row = run_one(make_tasks("deadline", 0, profile), CONFIG_BY_NAME["serial"],
                      phase="deadline", profile=profile)
        if row["error"] or row["metrics"]["accuracy"] < 1:
            raise RuntimeError(f"Deadline baseline not correct for {profile}; inspect before continuing.")
        deadlines[profile] = max(2.0, row["metrics"]["elapsed_s"] * 1.25)
    dump(out / "deadline_freeze.json", deadlines)

    rng = random.Random(2818)
    # Every configuration sees the same tasks within a replicate.
    for rep in range(args.calibration_reps):
        profiles = list(PROFILES)
        rng.shuffle(profiles)
        for profile in profiles:
            configs = list(CONFIGURATIONS)
            rng.shuffle(configs)
            for cfg in configs:
                run_one(fixtures["calibration"][f"{rep}:{profile}"], cfg, deadline=deadlines[profile],
                        phase="calibration", profile=profile, replicate=rep)
    policy = train_policy(episodes, args.memory_ceiling_gib)
    policy["environment_sha256"] = sha(out / "environment.json")
    policy["calibration_episode_count"] = sum(e["phase"] == "calibration" for e in episodes)
    dump(out / "policy.json", policy)
    freeze = {"policy_sha256": sha(out / "policy.json"), "fixture_sha256": sha(out / "fixtures.json"),
              "protocol_sha256": sha(out / "protocol.json"), "deadline_sha256": sha(out / "deadline_freeze.json")}
    dump(out / "PRE_TEST_FREEZE.json", freeze)
    print("FROZEN POLICY", json.dumps({"fixed": policy["best_fixed"], "adaptive": policy["choices"]}), flush=True)
    for rep in range(args.test_reps):
        profiles = list(PROFILES)
        rng.shuffle(profiles)
        for profile in profiles:
            tasks = fixtures["test"][f"{rep}:{profile}"]
            lengths = [len(engine.tokenize(t["prompt"])) for t in tasks]
            configs = {"fixed": CONFIG_BY_NAME[policy["best_fixed"]],
                       "adaptive": select_configuration(policy, lengths)}
            order = ["fixed", "adaptive"] if rep % 2 == 0 else ["adaptive", "fixed"]
            for mode in order:
                row = run_one(tasks, configs[mode], deadline=deadlines[profile],
                              phase="test", profile=profile, replicate=rep)
                row["policy_mode"] = mode
                # Append the mode to a separate compact index; full traces stay append-only.
                with (out / "test_index.jsonl").open("a") as f:
                    f.write(json.dumps({"sequence": row["sequence"], "policy_mode": mode}) + "\n")
    if sha(out / "policy.json") != freeze["policy_sha256"]:
        raise RuntimeError("Frozen policy was modified.")
    result = summarize(episodes, policy)
    result["wall_seconds"] = time.monotonic() - started
    result["total_model_requests"] = sum(len(e["records"]) for e in episodes)
    result["total_generated_tokens"] = sum(r["output_tokens"] for e in episodes for r in e["records"])
    dump(out / "summary.json", result)
    dump(out / "COMPLETE.json", {"status": "complete", "summary_sha256": sha(out / "summary.json"), "freeze_verified": True})
    print(json.dumps(result, indent=2), flush=True)


def main():
    args = parser().parse_args()
    if args.command == "pilot":
        pilot(args)
    elif args.command == "serve":
        from .server import serve
        serve(args)


if __name__ == "__main__":
    main()
