# Black-Box Inference Backend Failure Study

## Research Questions
1. How does a minimal llama.cpp baseline differ from vLLM and SGLang in performance and failure behavior across public and custom workloads?
2. Do inference optimizations introduce quality failures on certain long-context or workload-specific prompts, and can we characterize when optimized backends fail differently?

## Required Framing
We treat llama.cpp, vLLM, and SGLang as black-box inference backends. Because the experiment uses endpoint-level logs, we focus on observable behavior rather than low-level GPU internals.

## Tool Stack
{{TOOL_STACK}}

## Benchmarks and Workloads
{{WORKLOADS}}

## Setup
{{SETUP}}

## Correctness Summary
{{CORRECTNESS_SUMMARY}}

## Failure Bucket Summary
{{FAILURE_BUCKET_SUMMARY}}

## Performance Summary
{{PERFORMANCE_SUMMARY}}

## Representative Case Studies
{{CASE_STUDIES}}

## Limitations
We do not claim that any observed quality difference is definitively caused by KV-cache management, scheduling, or GPU memory behavior. Establishing that would require lower-level instrumentation, self-hosted servers, and controlled ablations. Our contribution is to identify and characterize failure patterns that can motivate deeper systems analysis.

{{LIMITATIONS_EXTRA}}

## Future Work
{{FUTURE_WORK}}

## Generated Artifacts
{{ARTIFACTS}}
