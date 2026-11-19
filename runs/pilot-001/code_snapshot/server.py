"""Local-only JSON API with one MLX owner thread and measured batch selection."""
from __future__ import annotations

import concurrent.futures
import json
import queue
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .core import select_configuration, sha


def serve(args):
    from .mlx_backend import Engine
    policy_path = Path(args.policy).resolve()
    policy = json.loads(policy_path.read_text())
    environment_path = policy_path.parent / "environment.json"
    if sha(environment_path) != policy["environment_sha256"]:
        raise ValueError("Policy environment manifest does not match its training run.")
    expected_environment = json.loads(environment_path.read_text())
    jobs = queue.Queue(maxsize=64)
    ready = threading.Event()
    state = {"ready": False, "completed": 0, "selected_counts": {}, "startup_error": None}
    lock = threading.Lock()

    def worker():
        try:
            engine = Engine(args.target, args.draft)
            environment = engine.manifest()
            for field in ["versions", "model_files", "tokenizer_mapping_sha256", "generation_source_sha256"]:
                if environment[field] != expected_environment[field]:
                    raise ValueError(f"Serving environment changed since calibration: {field}")
            state["ready"] = True
        except Exception as exc:
            state["startup_error"] = str(exc)
            ready.set()
            return
        ready.set()
        while True:
            group = [jobs.get()]
            until = time.monotonic() + max(0, args.microbatch_ms) / 1000
            while len(group) < 4:
                remaining = until - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    group.append(jobs.get(timeout=remaining))
                except queue.Empty:
                    break
            try:
                tasks = [job["task"] for job in group]
                lengths = [len(engine.tokenize(t["prompt"])) for t in tasks]
                budget = min(job["budget"] for job in group)
                config = select_configuration(policy, lengths, budget_gib=budget)
                run_started = time.monotonic()
                result = engine.run(tasks, config, memory_ceiling_gib=budget, phase="serve")
                with lock:
                    state["selected_counts"][config.name] = state["selected_counts"].get(config.name, 0) + 1
                    state["completed"] += len(group)
                for job, row in zip(group, result["records"]):
                    response = {
                        "id": row["id"], "text": row["text"], "configuration": config.name,
                        "batch_size": len(group), "prompt_tokens": row["prompt_tokens"],
                        "output_tokens": row["output_tokens"], "finish_reason": row["finish_reason"],
                        "queue_wait_s": run_started - job["submitted"],
                        "completion_s": run_started - job["submitted"] + row["finish_s"],
                        "ttft_s": run_started - job["submitted"] + row["ttft_s"] if row["ttft_s"] is not None else None,
                        "peak_active_gib": result["peak_active_gib"],
                        "effective": result["effective"], "error": row.get("error"),
                    }
                    job["future"].set_result(response)
            except Exception as exc:
                for job in group:
                    job["future"].set_exception(exc)
            finally:
                for _ in group:
                    jobs.task_done()

    owner = threading.Thread(target=worker, name="mlx-owner", daemon=True)
    owner.start()
    ready.wait()
    if state["startup_error"]:
        raise RuntimeError(state["startup_error"])

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status, value):
            data = json.dumps(value).encode()
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if self.path != "/health":
                return self.send_json(404, {"error": "Use GET /health or POST /generate"})
            with lock:
                snapshot = dict(state)
                snapshot["selected_counts"] = dict(state["selected_counts"])
            snapshot.update({"queue_depth": jobs.qsize(), "policy_choices": policy["choices"],
                             "best_fixed": policy["best_fixed"], "policy_sha256": sha(policy_path),
                             "scope": policy["scope"]})
            self.send_json(200, snapshot)

        def do_POST(self):
            if self.path != "/generate":
                return self.send_json(404, {"error": "Use POST /generate"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1024 * 1024:
                    raise ValueError("JSON body must be between 1 byte and 1 MiB.")
                body = json.loads(self.rfile.read(length))
                prompt = body.get("prompt")
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError("prompt must be a nonempty string.")
                tokens = body.get("max_tokens", 192)
                if type(tokens) is not int or not 1 <= tokens <= 192:
                    raise ValueError("max_tokens must be an integer from 1 to 192, the calibrated output bound.")
                budget = float(body.get("memory_budget_gib", policy["memory_ceiling_gib"]))
                if not 0 < budget <= policy["memory_ceiling_gib"]:
                    raise ValueError("memory_budget_gib must be positive and no larger than the calibrated ceiling.")
                future = concurrent.futures.Future()
                job = {"task": {"id": str(uuid.uuid4()), "prompt": prompt, "max_tokens": tokens},
                       "future": future, "submitted": time.monotonic(), "budget": budget}
                jobs.put_nowait(job)
            except queue.Full:
                return self.send_json(429, {"error": "Local queue is full."})
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                return self.send_json(400, {"error": str(exc)})
            try:
                response = future.result(timeout=600)
                self.send_json(503 if response.get("error") else 200, response)
            except Exception as exc:
                self.send_json(503, {"error": str(exc)})

    print(f"Local adaptive inference API ready: http://127.0.0.1:{args.port}", flush=True)
    print("GET /health; POST /generate {prompt, max_tokens?, memory_budget_gib?}", flush=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
