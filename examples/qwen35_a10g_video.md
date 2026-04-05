# Qwen3.5 Video Demo

This is the cleanest demo I would record on the current machine.

## Why this example

- It fits the current single-GPU A10G box without making the server setup look fragile.
- It produces an interesting `serve-report`: queue wait, TTFT, GPU phase breakdown, cache behavior, and prefix sharing should all show up.
- The prompts are all about ML systems theory, so the model output itself reinforces the exact ideas hotpath is measuring.
- The prompts still share a long repeated instruction prefix, so the report has something concrete to say about cacheability instead of looking random.

## What to run

```bash
chmod +x examples/start_qwen35_video_server.sh
chmod +x examples/stop_qwen35_video_server.sh
chmod +x examples/qwen35_a10g_video_demo.sh
./examples/qwen35_a10g_video_demo.sh
```

If you want to split the video into three explicit steps, run:

```bash
./examples/start_qwen35_video_server.sh
hotpath serve-profile --endpoint http://127.0.0.1:8000 --traffic examples/qwen35_a10g_video_traffic.jsonl --concurrency 4 --duration 60 --output .hotpath/qwen35-video
hotpath serve-report .hotpath/qwen35-video/serve_profile.db
hotpath disagg-config .hotpath/qwen35-video/serve_profile.db --format all
```

## What to show on camera

1. Show the hardware quickly.

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

2. Open the demo script and point out the three choices that matter:
   - model: `Qwen/Qwen3.5-4B`
   - bounded context: `--max-model-len 8192`
   - concurrency: `4`
   - startup flags: `--enforce-eager` and `--language-model-only`

3. Run the script.

4. When `hotpath serve-report` prints, pause on these sections:
   - `Latency`
   - `GPU Phase Breakdown`
   - `KV Cache`
   - `Prefix Sharing`
   - `Disaggregation Advisor`

5. If you want a quick content beat, open one or two generated answers from the model and point out that the requests themselves are about prefill, decode, KV cache, batching, and disaggregation.

6. End on `hotpath disagg-config` to show that the tool moves from observation to action.

## Suggested narration

Use this structure:

1. "We are profiling a live vLLM server, not just benchmarking tokens per second."
2. "The workload is intentionally shaped around systems-theory questions and repeated prefixes, so both the model outputs and the cache analysis are relevant."
3. "The report separates queue wait, prefill, and decode so you can see whether the latency problem is a traffic problem, a cache problem, or a GPU-compute problem."
4. "Then hotpath turns that analysis into a deployment recommendation instead of stopping at charts."

## Good recording defaults

- Keep the terminal full width.
- Use a 60-second profile instead of a 5-minute run.
- Do not enable `--nsys` in the first video. It slows the demo and distracts from the serving story.
- If the model download is slow, pre-pull it before recording and start from the script already on disk.
- For this machine and vLLM build, prefer the provided startup script over calling `vllm serve` directly.
