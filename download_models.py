"""Download the pinned public model snapshots and verify the measured files."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    from huggingface_hub import snapshot_download
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("models"))
    args = parser.parse_args()
    manifest = json.loads((Path(__file__).parent / "models.json").read_text())
    for role, model in manifest.items():
        destination = args.out / model["directory"]
        snapshot_download(
            repo_id=model["model_id"], revision=model["revision"], local_dir=destination,
        )
        for filename, expected in model["files"].items():
            path = destination / filename
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            if path.stat().st_size != expected["bytes"] or digest.hexdigest() != expected["sha256"]:
                raise ValueError(f"File does not match the measured snapshot: {path}")
        print(f"Verified {role}: {destination.resolve()}", flush=True)


if __name__ == "__main__":
    main()
