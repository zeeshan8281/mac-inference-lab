"""Independent saved-data checks; does not load model weights or tune a policy."""
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_json_answer(text):
    def unique(items):
        keys = [k for k, _ in items]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate key")
        return dict(items)
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.partition("\n")[2].rpartition("```")[0].strip()
    try:
        return json.loads(text, object_pairs_hook=unique)
    except ValueError:
        return None


run = Path(sys.argv[1])
protocol = json.loads((run / "protocol.json").read_text())
fixtures = json.loads((run / "fixtures.json").read_text())
policy = json.loads((run / "policy.json").read_text())
freeze = json.loads((run / "PRE_TEST_FREEZE.json").read_text())
summary = json.loads((run / "summary.json").read_text())
complete = json.loads((run / "COMPLETE.json").read_text())
deadlines = json.loads((run / "deadline_freeze.json").read_text())
episodes = [json.loads(line) for line in (run / "episodes.jsonl").read_text().splitlines()]
index_rows = [json.loads(line) for line in (run / "test_index.jsonl").read_text().splitlines()]
test_index = {r["sequence"]: r["policy_mode"] for r in index_rows}
errors = []


def require(condition, message):
    if not condition:
        errors.append(message)


for filename, key in [
    ("policy.json", "policy_sha256"), ("fixtures.json", "fixture_sha256"),
    ("protocol.json", "protocol_sha256"), ("deadline_freeze.json", "deadline_sha256"),
]:
    require(digest(run / filename) == freeze[key], f"Freeze mismatch: {filename}")
for filename, expected in protocol["code_sha256"].items():
    require(digest(run / "code_snapshot" / filename) == expected, f"Snapshot hash mismatch: {filename}")
require(digest(run / "environment.json") == policy["environment_sha256"], "Policy environment hash mismatch")
require(digest(run / "summary.json") == complete["summary_sha256"], "Completed summary hash mismatch")
require(digest(run / "fixtures.json") == protocol["fixture_sha256"], "Protocol fixture hash mismatch")
require(len(test_index) == len(index_rows), "Duplicate test index sequence")
require([e["sequence"] for e in episodes] == list(range(len(episodes))), "Missing/duplicate trace sequence")
require(set(test_index) == {e["sequence"] for e in episodes if e["phase"] == "test"}, "Test index coverage mismatch")
phase_counts = Counter(e["phase"] for e in episodes)
require(phase_counts["calibration"] == protocol["calibration_reps"] * 4 * len(protocol["configurations"]), "Calibration count mismatch")
require(phase_counts["test"] == protocol["test_reps"] * 4 * 2, "Test count mismatch")
task_lookup = {}
for phase, blocks in fixtures.items():
    for block, tasks in blocks.items():
        for task in tasks:
            body = task["prompt"].split("REGISTER:\n", 1)[1]
            raw_records, raw_refs = body.split("\nREQUESTED REFS: ", 1)
            lookup = {r["ref"]: r for r in map(json.loads, raw_records.splitlines())}
            expected = [lookup[ref.strip()] for ref in raw_refs.split(",")]
            require(task["expected"] == expected, f"Bad fixture ground truth: {task['id']}")
            require(task["id"] not in task_lookup, f"Duplicate fixture ID: {task['id']}")
            task_lookup[task["id"]] = task
cal_prompts = {t["prompt"] for b in fixtures["calibration"].values() for t in b}
test_prompts = {t["prompt"] for b in fixtures["test"].values() for t in b}
require(not cal_prompts.intersection(test_prompts), "Calibration/test prompt overlap")
checked = 0
for episode in episodes:
    if episode["phase"] not in ["calibration", "test"]:
        continue
    require(episode["deadline_s"] == deadlines[episode["profile"]], "Episode deadline differs from frozen value")
    expected_tasks = fixtures[episode["phase"]][f"{episode['replicate']}:{episode['profile']}"]
    require([r["id"] for r in episode["records"]] == [t["id"] for t in expected_tasks], "Episode fixture coverage mismatch")
    correct = timely = 0
    for row in episode["records"]:
        task = task_lookup[row["id"]]
        parsed = strict_json_answer(row["text"])
        valid = parsed == task["expected"] and isinstance(parsed, list)
        if valid:
            valid = all(isinstance(r, dict) and type(r.get("units")) is int for r in parsed)
        if row.get("error") and row["finish_reason"] is None:
            valid = False
        require(bool(row["correct"]) == valid, f"Score mismatch: sequence {episode['sequence']} {row['id']}")
        require(len(row["tokens"]) == row["output_tokens"] == len(row["token_times_s"]), "Token trace length mismatch")
        require(all(a <= b for a, b in zip(row["token_times_s"], row["token_times_s"][1:])), "Nonmonotonic token times")
        require(row["finish_s"] <= episode["metrics"]["elapsed_s"] + .01, "Finish outside episode")
        correct += int(valid)
        timely += int(valid and row["finish_s"] <= episode["deadline_s"])
        checked += 1
    metrics = episode["metrics"]
    require(correct == metrics["correct"], "Incorrect episode correctness total")
    require(timely == metrics["on_time_correct"], "Incorrect deadline total")
    require(math.isclose(60 * timely / metrics["elapsed_s"], metrics["goodput_per_min"], rel_tol=1e-10), "Incorrect goodput")
    if episode["phase"] == "test":
        mode = test_index[episode["sequence"]]
        expected = policy["best_fixed"] if mode == "fixed" else policy["choices"][episode["feature"]]
        require(episode["configuration"] == expected, "Test execution differs from frozen policy")
pairs = Counter((e["profile"], e["replicate"]) for e in episodes if e["phase"] == "test")
require(all(count == 2 for count in pairs.values()), "Incomplete fixed/adaptive pair")
require(len(pairs) == summary["paired_result"]["blocks"], "Paired block total mismatch")
pair_modes = Counter((e["profile"], e["replicate"], test_index[e["sequence"]]) for e in episodes if e["phase"] == "test")
require(len(pair_modes) == len(pairs) * 2 and all(n == 1 for n in pair_modes.values()), "Duplicate/missing policy in pair")
fixed = [e["metrics"]["goodput_per_min"] for e in episodes if e["phase"] == "test" and test_index[e["sequence"]] == "fixed"]
adaptive = [e["metrics"]["goodput_per_min"] for e in episodes if e["phase"] == "test" and test_index[e["sequence"]] == "adaptive"]
require(math.isclose(sum(adaptive)/len(adaptive) - sum(fixed)/len(fixed), summary["paired_result"]["difference_goodput"], abs_tol=1e-9),
        "Summary difference mismatch")
result = {
    "status": "PASS" if not errors else "FAIL", "errors": errors,
    "calibration_and_test_requests_rescored": checked, "paired_held_out_blocks": len(pairs),
    "phase_episode_counts": dict(phase_counts),
    "frozen_inputs_and_code_snapshot_verified": True if not errors else None,
    "scope": "Independent fixture reconstruction, scoring, timestamp arithmetic, policy choices and file-hash checks. Does not independently reproduce GPU execution.",
}
(run / "AUDIT.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
sys.exit(bool(errors))
