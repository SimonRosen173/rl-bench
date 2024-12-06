# rl-bench: RL Benchmarker 🤖💻

RL benchmarker for large scale highly parallelised experiments specifically tailored for slurm based clusters. 

The main paradigm behind this code is **realibility** and handling an unreliable cluster. Due to the unreliability of WANDB on clusters manual logging is implemented, such that a user can  seamlessly switch between logging manually or via WANDB. 
