# Silent benchmark dashboard

An original React/SVG composition built with Remotion. The deliverables are
[`../media/mac-inference-dashboard.mp4`](../media/mac-inference-dashboard.mp4)
and [the matching PNG](../media/mac-inference-dashboard.png).

**1440 × 1080 · 30 fps · 480 frames · 16 seconds · H.264 · no audio.**
The PNG is frame 450, inside the final hold.

## Reproduce

From this directory, with Node.js/npm and Python 3:

```sh
npm ci
npm run data
npm run typecheck
npm run poster
npm run render
```

`npm run studio` opens the editable composition. On macOS the configuration
uses installed Google Chrome, when present, in a separate headless process
with a temporary profile. It does not access the user's logged-in profile.
Otherwise the first render may download Remotion's headless Chromium. Set
`REMOTION_BROWSER_EXECUTABLE` to use another compatible full Chrome binary.
The bundled Headless Shell timed out on the measured Mac; Chrome
158.0.8127.37 with `swangle` rendered successfully.

Model weights, MLX and a GPU benchmark run are not needed to render the video.
Rendering was checked using Node 26.13.1 and npm 11.20.0 on macOS.
All JavaScript dependency versions are captured in
`package-lock.json`; font files are bundled from the pinned Inter package.

## Timing and editing

`src/Dashboard.tsx` owns the layout, SVG artwork, typography and frame-based
animation. `src/index.tsx` owns frame rate, duration and canvas dimensions.
There is one persistent dashboard:

| Time | Motion |
| --- | --- |
| 0–2 s | Cards settle into place; the chip rotates slightly. |
| 1.5–5.2 s | Fixed/adaptive goodput bars rise from a zero baseline. |
| 6.2–9.5 s | Accuracy and the two latency comparisons fill. |
| 9.5–12.2 s | Calibration bar and uncertainty interval receive emphasis. |
| 13–16 s | Complete dashboard holds for reading and a clean poster. |

Numbers stay at their measured values throughout. Only chart geometry is
animated; intermediate bar lengths are reveal effects, not additional
observations. The headline uncertainty is visible throughout. The two latency
rows use different horizontal scales and compare the two policies **within
each row**. The chip is a generic SVG illustration of the processor, not a
photo or representation of the computer's enclosure.

## Evidence and scope

`prepare_data.py` reads three published result files from `runs/pilot-002` and
writes `src/data.json`, including their SHA-256 hashes:

- `summary.json`: paired mean episode goodput, confidence interval, workload
  latency, test correctness, policy choices and full-pilot request count.
- `SECONDARY_ANALYSIS.json`: calibration accuracy and unique prompt/execution
  counts.
- `environment.json`: actual device memory, processor and MLX-LM version.

The title calls this the first completed experiment (`EXPERIMENT 001`); the
source directory is `pilot-002` because `pilot-001` was aborted before
calibration. The footer identifies the exact completed run.

The apparent 3.6% overall gain is inconclusive: the paired difference interval
includes zero. Both policies scored 77/80; each achieved 76/80 correct within
the deadline. The goodput metric is a balanced mean over 32 paired finite-burst
workload blocks, not steady-state production throughput.

The short-request latency comparison contains eight paired held-out cases.
The KV result is a separate calibration comparison: ordinary KV 28/30 versus
4-bit KV 0/30 with quantization starting at token zero. It does not establish
that every 4-bit KV implementation fails. The target's 4-bit **weights** are a
separate setting from its KV cache.

The 374-request figure covers the full pilot. It must not be described as 374
unique test prompts: the held-out set contains 80 prompts replayed under two
policies (160 executions). Models were warm and resident; the task was
controlled structured extraction.

No raw results, frozen protocol, deadlines or measured policy are changed by
building these media files. See [NOTICE](../NOTICE.md) for visual-reference and
dependency attribution.
