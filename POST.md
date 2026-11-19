# One post, one measured graph

I built an inference selector on my M5 Pro.

160 held-out runs: +3.6% average goodput, but the 95% CI includes zero. Same 77/80 accuracy.

Single short requests finished sooner; their first token arrived later.

Code, raw traces & graph:
https://github.com/zeeshan8281/mac-inference-lab

Attach only: `runs/pilot-002/comparison.png`.

The 160 executions replay 80 unique prompts under two policies. “Goodput” means correct responses completed before a fixed deadline per episode minute. The full uncertainty calculation and scope are in RESULTS.md.
