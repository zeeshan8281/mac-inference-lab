"""Descriptive checks outside the frozen primary analysis."""
import collections
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
episodes = [json.loads(line) for line in (run / "episodes.jsonl").read_text().splitlines()]
fixtures = json.loads((run / "fixtures.json").read_text())
counts = collections.defaultdict(lambda: {"correct": 0, "requests": 0})
for episode in episodes:
    if episode["phase"] == "calibration":
        counts[episode["configuration"]]["correct"] += episode["metrics"]["correct"]
        counts[episode["configuration"]]["requests"] += episode["metrics"]["requests"]
paired = {
    name: {r["id"]: r for e in episodes if e["phase"] == "calibration" and e["configuration"] == name for r in e["records"]}
    for name in ["serial", "draft2"]
}
result = {
    "calibration_accuracy": dict(counts),
    "calibration_greedy_serial_draft_pairs": len(paired["serial"]),
    "calibration_identical_token_sequences": sum(paired["serial"][k]["tokens"] == paired["draft2"][k]["tokens"] for k in paired["serial"]),
    "held_out_unique_prompts": len({t["prompt"] for block in fixtures["test"].values() for t in block}),
    "held_out_executions": sum(len(e["records"]) for e in episodes if e["phase"] == "test"),
    "calibration_unique_prompts": len({t["prompt"] for block in fixtures["calibration"].values() for t in block}),
    "long_prompt_deadlines_s": json.loads((run / "deadline_freeze.json").read_text()),
}
(run / "SECONDARY_ANALYSIS.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
