# Fresh AI experiments for a 48 GiB Apple-silicon Mac

Research date: 19 November 2026. Hardware: Apple M5 Pro, 15 CPU cores, 16 GPU cores, 48 GiB unified memory.

**Evidence coverage: 473 distinct technical sources—215 general AI and 258 inference engineering—plus two directly inspected X posts.** The shortlist contains 25 ideas; the three supporting research files contain 60 detailed designs. Most documents were inspected through selected substantive sections, with 21 papers inspected at abstract level and four short technical documents read in full. Browser checks covered current implementation discussions and social posts; the rest used direct public text retrieval. See the source audit for exact depth labels.

**My recommendation: build an adaptive local inference controller.** First map which combinations of batching, speculative decoding and cache precision actually run well; then make a controller select a configuration from that measured map. This gives you an engineering artifact, a controlled experiment and a result people can reproduce on their own Macs.

The rankings below are editorial judgments about usefulness, feasibility and clarity of the resulting demonstration. They are not measured popularity scores. All proposed outcomes are hypotheses. None of these experiments was run for this report.

## The strongest inference-engineering ideas

| Rank | Experiment and artifact | What to change and measure | One graph for the post | Posting angle, to adapt to the actual result |
|---|---|---|---|---|
| 1 | **An adaptive inference controller for a Mac** | Build a measured configuration map, then select concurrency, draft depth and cache policy from queue length and memory headroom. Compare against the best fixed configuration chosen on separate calibration traces. Measure correct responses delivered before deadlines, latency and memory. | Useful responses per minute versus offered load, one curve per policy. | “I made my local AI change its inference settings as the workload changed.” |
| 2 | **Do speed optimizations actually compose?** | Cross batching, speculative decoding and KV precision. Instrument effective execution paths, serial fallbacks, memory and answer accuracy. | Configuration grid with speed and correctness annotations. | “I switched on three inference optimizations. Here’s which combinations actually worked together.” |
| 3 | **A scheduler that protects short interactive requests** | Interleave long-document prefills with short chats; compare FIFO, static chunking and a latency-aware policy in one engine. Count rejected and timed-out requests. | Short-request p95 first-token delay versus background load. | “One long prompt can make everyone else wait. I measured—and reduced—the delay.” |
| 4 | **A speculative-decoding autopilot** | Adapt draft length using accepted tokens and measured draft/verification cost. Compare with no drafting and the best fixed length on held-out workloads. | Net speedup versus workload type, including slowdowns. | “Speculative decoding is not always faster. My controller learns when to use it.” |
| 5 | **Make the cache survive changing questions** | Reuse one document cache for different question families. Compare full KV, quantized KV and a ported eviction policy; keep the source document identical. | Accuracy versus cache memory, split by whether the question changes topic. | “My AI remembered the document—until I asked a different kind of question.” |
| 6 | **An edit-aware prompt cache** | Change a fact near the beginning, middle or end of a cached prompt. Build exact invalidation/reuse boundaries and compare with recomputation. | Latency saved versus edit position, with stale-answer failures marked. | “Can a local AI reuse its cache after you edit the document?” |
| 7 | **Memory-aware admission and cache reclamation** | Limit simultaneous requests and reclaim reusable idle caches before pressure becomes severe. Preserve the actual task context. Compare static limits with a controller. | Completed valid requests and p95 latency across memory budgets. | “The model fits in RAM. The workload may not. I built a controller for that.” |
| 8 | **One useful custom Metal kernel, end to end** | Profile first, then fuse one demonstrated bottleneck—such as normalization plus adjacent work or quantized matrix-vector operations. Test numerical error and actual generation time. | End-to-end speedup versus batch/context size, showing the break-even point. | “A faster kernel does not always mean a faster model. Here is the crossover.” |
| 9 | **Quantization’s rare-detail failure boundary** | Quantize the same base weights at several supported precisions. Test exact identifiers, numbers, negations and structured arguments as well as ordinary text. | Error rate by detail type versus model/cache memory. | “I tested the details that ordinary quantization scores can hide.” |
| 10 | **Long prompts versus long answers: different bottlenecks** | Sweep prompt length and output length separately, fixing model and precision. Profile prefill, decode, tokenization and streaming independently. | A prompt-length × output-length heatmap of the dominant cost. | “There is no single tokens-per-second number for a local model.” |
| 11 | **The cost of abandoned generations** | Inject realistic cancellations and slow stream consumers. Measure stop-to-GPU-idle delay, retained cache memory and latency imposed on other clients; implement cleanup where needed. | Wasted compute after cancellation versus load, before and after. | “I closed the chat. How long did the GPU keep working?” |
| 12 | **Cold start, shape changes and compilation debt** | Compare uncached compilation, warmed repeated shapes and variable shapes. Test shape bucketing only where semantics and padding are controlled. | Cumulative time versus number of requests; mark when compilation pays back. | “My optimization was slower for the first few requests. Here’s when it paid off.” |
| 13 | **A four-hour sustained-inference stress test** | Repeat randomized workload blocks under a documented power/thermal setup. Track latency, memory, swap and errors through time. | Throughput and tail latency over four hours, with workload boundaries. | “The five-minute benchmark looked good. Here is the four-hour run.” |
| 14 | **Exactness under batching and kernel changes** | Use near-tie next-token cases and controlled prompts. Separate floating-point logit differences, greedy token flips and actual task failures. | Token divergence versus logit margin, split by execution path. | “Can changing the batch change a deterministic answer?” |
| 15 | **A smaller draft vocabulary—and its coverage cliff** | Physically reduce draft output-head work, preserve correct target verification, and increase the fraction of rare identifiers or multilingual tokens. Measure net latency and acceptance; masking an unchanged full head is not a compute reduction. | Net speedup versus rare-token frequency. | “I made the draft model cheaper. Then rare tokens ate the speedup.” |

### What the current implementation evidence changes

- The inspected [MLX-LM server source](https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/server.py) says KV quantization disables server batching, and its batchability condition excludes a draft model. Check the exact installed version before choosing a factorial experiment. A flag accepted by a command line is not proof that the intended path ran.
- [Ollama PR 19197](https://github.com/ollama/ollama/pull/19197) introduced concurrent MLX generation with chunked prompts. Its author-reported benchmarks illustrate a real tradeoff: a waiting client can get its first token earlier while total completion takes longer.
- [Ollama PR 19663](https://github.com/ollama/ollama/pull/19663), inspected in the browser and merged on 16 November, adds concurrent speculative verification. It also describes temporary-memory costs. The earlier “drafting pauses under concurrency” limitation is therefore stale as a blanket statement about current mainline. A merge does not establish that a feature is available in your installed release.
- [MLX-VLM issue 1567](https://github.com/Neywa-Labs/mlx-vlm/issues/1567) documents a multi-request quantized-KV failure and its fix. It is **closed and fixed**, not a claim of a current unfixed bug. It motivates testing interactions and mixed prompt lengths.
- [MLX compilation](https://ml-explore.github.io/mlx/build/html/usage/compile.html), [custom Metal kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html) and [unified memory](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html) provide practical local implementation paths. CUDA kernels and datacenter serving architectures in the bibliography are design references; their reported speedups do not transfer to this Mac.

### Technical foundations for the most promising tests

| Experiment | Inspected evidence to start from | What it supports |
|---|---|---|
| Adaptive scheduling and short-request protection | [Sarathi-Serve](https://arxiv.org/abs/2403.02310), [Ollama concurrent generation](https://github.com/ollama/ollama/pull/19197) | Prefill/decode scheduling and the throughput/latency tradeoff. Sarathi's implementation and hardware results are not assumed to run unchanged on Metal. |
| Adaptive speculative decoding | [Speculative decoding](https://arxiv.org/abs/2211.17192), [Recurrent Drafter](https://arxiv.org/abs/2403.09919), [Apple implementation](https://github.com/apple/ml-recurrent-drafter) | Target verification, acceptance and execution cost. The inspected Apple README warns that non-greedy decoding may not exactly match the target distribution; begin with greedy checks. |
| Cache fidelity when the question changes | [SCBench](https://arxiv.org/abs/2412.10319), [SnapKV](https://arxiv.org/abs/2404.14469), [KIVI](https://arxiv.org/abs/2402.02750) | Evaluate reused context and memory reduction beyond a single easy retrieval question. An eviction implementation may need a local port. |
| Edit-aware cache reuse | [CacheBlend](https://arxiv.org/abs/2405.16444), [Cache-Craft](https://arxiv.org/abs/2502.15734) | Cached chunks have context dependencies; naive reuse after an edit needs a recomputation control. These papers do not certify a proposed local cache implementation. |
| Rare-field precision boundary | [WKVQuant](https://arxiv.org/abs/2402.12065), [KIVI](https://arxiv.org/abs/2402.02750), [AWQ](https://arxiv.org/abs/2306.00978) | Separate weight precision and cache precision, and measure task-level effects alongside memory. Named quantization methods need explicit runtime support. |
| Draft-vocabulary coverage | [Vocabulary trimming study](https://arxiv.org/abs/2603.05210) | A draft can become cheaper while losing coverage of rare outputs. Physically trimming computation differs from merely masking logits. |
| Context-length stress tests | [RULER](https://arxiv.org/abs/2404.06654), [LongBench](https://arxiv.org/abs/2308.14508) | A model accepting a long input is not evidence that it uses all of it reliably. |

## Fresh applied experiments alongside inference engineering

| Rank | Experiment and artifact | The meaningful comparison | One graph | Posting angle |
|---|---|---|---|---|
| 1 | **An offline worker that produces finished files** | A local agent must edit spreadsheets, reconcile CSVs, query SQLite and create specified outputs. Compare direct prompting, a tool loop and tool execution with deterministic validation. | Completion rate by task family and time budget. | “I gave an offline model real work. Here are the files it actually finished.” |
| 2 | **A small specialist that learns without forgetting** | Locally train a task adapter; compare base, task-only training, replay-mixed training and a retrieval baseline at matched data budgets. Evaluate held-out target tasks and untouched general tasks. | Target improvement versus retained general ability. | “I trained a specialist on my Mac—and measured what it forgot.” |
| 3 | **Small model plus tools versus a larger model under a deadline** | Use document extraction, SQL, constrained planning or data-cleaning tasks with artifact-level checks. Compare equal wall time and report actual memory/energy separately. | Correct completed tasks versus deadline. | “Does a bigger model win when both get the same amount of time?” |
| 4 | **A document pipeline that reconciles totals** | Compare direct vision-language extraction, OCR plus parsing, and extraction plus arithmetic/schema checks. Use invoices, tables and deliberately difficult scans with known truth. | Correct complete documents versus scan degradation. | “Reading every number is not enough. Can the AI make the totals add up?” |
| 5 | **A local evidence assistant that handles changed facts** | Maintain a document collection with revisions and conflicting claims. Compare lexical retrieval, embeddings, hybrid search and reranking; judge both answer and cited version. | Correct supported answers versus revision/conflict difficulty. | “I changed the evidence. Did the local AI update its answer?” |
| 6 | **A scientific analyst that must verify its own calculations** | Give datasets and questions with known computations. Compare prose-only answers, generated code and tool-grounded execution with checks for units, leakage and numerical consistency. | Correct analyses by failure category. | “My local AI had to show calculations that actually ran.” |
| 7 | **A foundation model versus classical ML on small tabular data** | Compare a locally supported TabPFN configuration with tuned classical baselines under identical train/test splits and time budgets. Include a shifted test set. | Accuracy or calibrated loss versus dataset size and compute time. | “Do small datasets need a foundation model—or a good baseline?” |
| 8 | **An active-learning loop that buys fewer labels** | Compare random sampling, uncertainty sampling and diversity-aware sampling for a local classifier. Keep evaluation labels isolated and include label noise. | Held-out quality versus number of labels acquired. | “How much can the model learn from 100 carefully chosen labels?” |
| 9 | **Recover a physical law from messy measurements** | Give a local agent a damped oscillator or growth dataset with noise, missing metadata and unit errors. Compare an equation guess, numerical fitting and tool-assisted discovery with unit checks. | Predicted versus true dynamics outside the fitting interval. | “Could my local AI discover the equation—and notice the units were wrong?” |
| 10 | **A time-series foundation model versus a simple forecast** | Compare small CPU-runnable pretrained forecasters with seasonal naive and AutoARIMA using rolling-origin evaluation and shifted series. | Forecast error and interval coverage versus runtime. | “The foundation model had to beat ‘same as last week.’” |

Useful starting implementations and evaluation assets include [smolagents](https://github.com/huggingface/smolagents), [AppWorld](https://github.com/StonyBrookNLP/appworld), [MLX examples](https://github.com/ml-explore/mlx-examples), [PEFT](https://github.com/huggingface/peft), [Docling](https://github.com/docling-project/docling), [OmniDocBench](https://github.com/opendatalab/OmniDocBench), [BEIR](https://github.com/beir-cellar/beir) and [TabPFN](https://github.com/PriorLabs/TabPFN). A benchmark being available does not mean its full environment or model runs unmodified on macOS; the detailed plans identify local subsets and portability work.

For the scientific directions, [PySINDy](https://github.com/dynamicslab/pysindy) provides a concrete equation-discovery tool, while [Chronos-2](https://huggingface.co/amazon/chronos-2) and [StatsForecast](https://github.com/Nixtla/statsforecast) support a small local forecasting comparison. Preserve time-ordered splits and evaluate extrapolation; fitting the observed data alone is not success.

## The first experiment I would actually run

**Start with idea 2, the optimization-composition map. Use it to build idea 1, the adaptive controller.** This is a useful sequence because the controller needs measured evidence about the configurations it is allowed to choose.

### Deliverables

1. A version-pinned runner with an explicit configuration/capability manifest.
2. JSONL request traces containing enqueue, first-token and finish times; answer checks; effective batching/drafting/cache mode; and memory observations.
3. One graph showing the latency/quality tradeoff for every supported configuration.
4. A small configuration selector trained on calibration traces and evaluated on separate request traces.

### A bounded four-hour measurement pilot

This is a **measurement budget after the harness and a working model are ready**. Writing a robust scheduler or a Metal kernel can take longer. Model downloads, environment setup and an unsupported-port implementation do not fit automatically inside this budget.

| Time | Work |
|---|---|
| 0–30 minutes | Verify the execution path for each candidate configuration. Calibrate request duration and memory. Reject unsupported or unstable combinations from the executable matrix, but retain their status in the capability report. |
| 30–60 minutes | Freeze a small held-out workload: short structured extraction, longer document questions and mixed request lengths. Fix exact expected outputs where possible. Set deadlines using calibration data before the measured runs. |
| 60–180 minutes | Run counterbalanced blocks of supported configurations on identical arrival traces. Begin with a small model and expand the request count only if timing permits. Include both solo and concurrent workloads. |
| 180–220 minutes | Repeat the most consequential contrast and inspect correctness failures, fallback paths and memory retention. Treat this as confirmation only if its cases and analysis were reserved in advance. |
| 220–240 minutes | Produce the graph, source/version manifest and a short result summary with failures and uncertainty. |

The core matrix can be **two concurrency settings × drafting off/on × two supported cache modes**, reduced where the engine cannot execute a combination. Do not silently turn unsupported cells into another setting. A feasibility pilot need not complete a large exhaustive matrix.

Use a small 0.5–3B/4B-class compatible model pair to establish correctness and instrumentation, then move to a 7–14B or larger quantized target after a memory/latency probe. These are planning ranges, not a guarantee that every architecture, quantizer or context will fit. The detailed algorithm plan identifies a Qwen2.5 3B target plus 0.5B draft as a candidate whose config vocabulary dimensions match; still verify token IDs and runtime compatibility. A separate compatible checkpoint must be used if the experiment requires a native drafting head.

### What makes the result credible

- **Fix the model within a comparison.** Changing engine, model conversion and quantizer together is a system comparison, not attribution to a single optimization.
- **Measure useful work.** A request counts only if its answer passes the stated check and it finishes before its deadline. Keep failures, rejected admissions and timeouts in the denominator.
- **Include queueing.** Timestamp before submission; use scheduled arrivals for serving tests so a slow server does not make its own workload artificially easier.
- **Separate timing phases.** Report time to first token, generation intervals and end-to-end completion. Warm and cold caches answer different questions.
- **Verify semantic effects.** KV quantization/eviction can change answers; numerical drift is not automatically a meaningful task failure. Greedy equality tests are useful for paths intended to preserve greedy results. Distribution-preserving claims for stochastic speculative decoding need stronger tests.
- **Use paired workload blocks.** Replay the same tasks/arrival traces; randomize condition order; estimate uncertainty across independent tasks or workload blocks rather than treating adjacent generated tokens as independent samples.
- **Keep tuning and testing apart.** Select the fixed baseline and adaptive policy on calibration data, then compare on unseen workload mixes.
- **Show a negative result honestly.** A controller that fails to beat a well-chosen static configuration is still an informative outcome if its overhead and failure regime are visible.

## Practical Mac limits

The machine has **48 GiB shared by CPU, GPU and the operating system**. A rough weight-only lower bound is parameters × bits / 8; 32 billion parameters at four bits is about 16 GB decimal before scales, metadata, KV cache, temporary buffers and runtime overhead. That arithmetic is not a model-fitting measurement.

For immediately executable experiments, prefer MLX/MLX-LM, a verified Metal path in llama.cpp, or a specifically verified Ollama MLX build. Model architecture and version matter. A generic vLLM CPU build on macOS is not evidence that CUDA-oriented vLLM kernels run through Metal.

KV eviction policies, new draft heads, unusual quantizers and datacenter scheduling ideas may need a port. Start with a reference implementation and a correctness test; call a Python prototype a prototype. For the first kernel project, choose a bottleneck found in this exact workload, not the most fashionable operator.

A four-hour sustained test is useful for latency drift, memory retention and thermal behavior. A single time series is not enough to attribute a change to temperature alone: record workload phase, power mode and background activity and repeat counterbalanced conditions when making that causal claim.

## What the browser check says about posting potential

Two current X posts were inspected directly, including their visible engagement:

- [Local Mac inference optimization, 3 November](https://x.com/Jhaddix/status/2117324109185188281): **185 likes, 13,713 views and 103 bookmarks** observed on 19 November. The post presents concrete changes and before/after speed. Its 512 GB Mac configuration is not transferable to this 48 GiB machine.
- [Ollama concurrent generation, 12 November](https://x.com/TeksEdge/status/2120758496358732166): **14 likes and 1,256 views** observed on 19 November. Its useful angle is the tradeoff between responsiveness and finishing all work.

This is an anecdotal sample of active discussion, **not a representative trend ranking or a prediction of your impressions**. The strongest defensible posting format is a specific workload, a clear before/after result, one graph, exact hardware and a runnable artifact. Draft hooks above describe what to test; publish the measured outcome even when it is a slowdown or no improvement.

## Research evidence and detailed plans

The accompanying [source catalog](SOURCES.md), [machine-readable catalog](SOURCE_CATALOG.json) and [audit](SOURCE_AUDIT.json) document the source counts and inspection depth. Sources include papers, official technical documentation, implementation code, model cards, datasets and repository READMEs. Reading a substantive section or abstract is recorded as such; this report does not claim that every cited paper was read in full.

The three structured evidence files contain the full experiment designs, source-specific findings, tooling and limitations:

- [Applied experiments](evidence/Fresh_applied_local_AI_experiments.json)
- [Inference algorithms and memory](evidence/Inference_algorithms_and_memory_experiments.json)
- [Inference serving and systems](evidence/Inference_serving_and_systems_experiments.json)
