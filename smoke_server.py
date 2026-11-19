"""Exercise a running localhost API, including a real concurrent GPU batch."""
import argparse
import concurrent.futures
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

from inference_lab.core import make_tasks, score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8766")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Smoke-test output already exists.")
    checks = []

    def request(path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(args.url + path, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as exc:
            return exc.code, json.load(exc)

    status, health = request("/health")
    checks.append({"name": "health", "pass": status == 200 and health["ready"], "status": status})
    invalid = [
        ("list_body", [], 400),
        ("boolean_tokens", {"prompt": "hello", "max_tokens": True}, 400),
        ("negative_budget", {"prompt": "hello", "memory_budget_gib": -1}, 400),
        ("infeasible_budget", {"prompt": "hello", "memory_budget_gib": 1}, 503),
        ("uncalibrated_length", {"prompt": "word " * 4500, "max_tokens": 1}, 503),
    ]
    for name, payload, expected_status in invalid:
        status, body = request("/generate", payload)
        checks.append({"name": name, "pass": status == expected_status and "error" in body,
                       "status": status, "response": body})
    status, body = request("/generate", {"prompt": "Return only the JSON array [1,2,3].", "max_tokens": 64})
    try:
        # The API returns raw model text, not grammar-constrained JSON.
        # Like the pilot scorer, accept optional Markdown fences.
        text = body.get("text", "").strip()
        if text.startswith("```") and text.endswith("```"):
            text = text.partition("\n")[2].rpartition("```")[0].strip()
        parsed = json.loads(text)
        single_correct = parsed == [1, 2, 3] and all(type(value) is int for value in parsed)
    except ValueError:
        single_correct = False
    checks.append({"name": "single_generation", "pass": status == 200 and single_correct,
                   "status": status, "response": body})
    tasks = make_tasks("warmup", 0, "burst_short")
    barrier = threading.Barrier(4)

    def concurrent_request(task):
        barrier.wait()
        status, body = request("/generate", {"prompt": task["prompt"], "max_tokens": 192})
        return {
            "status": status, "response": body,
            "pass": status == 200 and score(body.get("text", ""), task["expected"])
            and body.get("batch_size") == 4 and body.get("effective", {}).get("max_observed_decode_batch") == 4,
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        batch = list(pool.map(concurrent_request, tasks))
    checks.append({"name": "four_concurrent_requests", "pass": all(r["pass"] for r in batch), "records": batch})
    status, health = request("/health")
    checks.append({"name": "healthy_after_errors_and_generation", "pass": status == 200 and health["ready"],
                   "status": status, "response": health})
    result = {"status": "PASS" if all(c["pass"] for c in checks) else "FAIL", "checks": checks}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({c["name"]: c["pass"] for c in checks}, indent=2))
    raise SystemExit(result["status"] != "PASS")


if __name__ == "__main__":
    main()
