# loopgym

RL post-training (GRPO) of a small coding model on tasks made from bugs that autonomous coding agents actually
shipped, plus the systems around it: a Rust sandbox server for grading, a fused log-prob kernel (C++/CUDA/Triton),
FSDP2 / DeepSpeed training checked against single-process math, vLLM rollouts, and host / AWS scripts for GPU runs.
Built by local agents from TODO.md; see CLAUDE.md for the rules and the known answers every number is checked against.
