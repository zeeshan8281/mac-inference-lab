"""Build the animation's data from the published pilot, never from a new run."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUN = ROOT / "runs" / "pilot-002"
names = ("summary.json", "SECONDARY_ANALYSIS.json", "environment.json")
source = {name: json.loads((RUN / name).read_text()) for name in names}
summary, secondary, environment = (source[name] for name in names)
profiles = summary["profiles"]
data = {
    "run": "pilot-002",
    "source_sha256": {
        f"runs/pilot-002/{name}": hashlib.sha256((RUN / name).read_bytes()).hexdigest()
        for name in names
    },
    "hardware": {
        "chip": environment["device"]["device_name"].removeprefix("Apple "),
        "memory_gib": environment["device"]["memory_size"] / 2**30,
        "mlx_lm": environment["versions"]["mlx-lm"],
    },
    "model": "Qwen2.5-Coder 7B",
    "weight_bits": 4,
    "total_model_requests": summary["total_model_requests"],
    "configs": len(secondary["calibration_accuracy"]),
    "paired_result": summary["paired_result"],
    "test_unique_prompts": secondary["held_out_unique_prompts"],
    "test_executions": secondary["held_out_executions"],
    "correct": {
        mode: sum(profile[mode]["correct"] for profile in profiles.values())
        for mode in ("fixed", "adaptive")
    },
    "on_time_correct": {
        mode: sum(profile[mode]["on_time_correct"] for profile in profiles.values())
        for mode in ("fixed", "adaptive")
    },
    "solo_short": profiles["solo_short"],
    "calibration": secondary["calibration_accuracy"],
    "policy_choices": summary["choices"],
    "best_fixed": summary["best_fixed"],
}
assert data["correct"] == {"fixed": 77, "adaptive": 77}
assert data["paired_result"]["difference_95ci"][0] < 0 < data["paired_result"]["difference_95ci"][1]
assert data["policy_choices"]["solo_short"] == "draft2"
assert all(choice == "batch4_chunk256" for state, choice in data["policy_choices"].items() if state != "solo_short")
(HERE / "src" / "data.json").write_text(json.dumps(data, indent=2) + "\n")
print("Generated video/src/data.json from three frozen result files.")
