# Black-Box vLLM vs SGLang Failure Study

## Research Questions
1. How do vLLM and SGLang differ not only in performance, but in failure behavior across public and custom workloads?
2. Do inference optimizations introduce quality failures on certain long-context or workload-specific prompts, and can we characterize when vLLM and SGLang fail differently?

## Required Framing
We treat vLLM and SGLang as black-box optimized inference backends. Because the experiment uses managed endpoints and request-level logs, we focus on observable behavior rather than low-level GPU internals.

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
