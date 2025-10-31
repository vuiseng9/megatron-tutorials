## Megatron-LM Tutorials
*On Replicating Empirical Scaling Trends in Distributed Training and Model Parallelism [DP(FSDP)/TP/SP/CP/PP/VPP/EP]*

Motivation:

Review of distributed and parallelism strategies is out of the scope of the, we divert the audience to papers which we have cited. Therefore we expect the audience to have basic understanding of distributed training and model parallelism techniques. The goal is to provide hands-on experience with canned configuration on how different parallelism strategies affect the empirical scaling trends in terms of memory usage and training speed.

i think we should put all the scripts and makefile here for reference.

-----
### Setup and Run

We rely on the out-of-the-box Megatron-LM's `pretrain_gpt.py` and use bash scripts to wrap it with different parallelism strategies. We also use `Makefile` extensively to organize and manage the experiments. 

Easiest way to get started is to use prebuilt docker image:
```
```
Or build the image yourself using `docker/Dockerfile` (TODOwhat is the file size)

**How to run?** 
* The docker entrypoint will lead to working directory, `/workspace/megatron-lm/examples/`
* Each experiment is defined as a Makefile target prefixed with a unique id, you can see the target corresponding to a row in the tables below. Our intent is to reduce the number of bash scripts, instruction steps/argument, making the runs less friction to reproduce, basically just type make then the id and finally a tab to complete the target. e.g. type `make 101<tab>` turn into `make 101-gpt2xl-dp1-gbs1-bf16`.
* Metrics can be found in std output which is also logged to `./outdir/<experiment label>/logs.txt`. To see GPU memory usage, run `monitor-gpu` on a seperate terminal. Most runs stop after 100 training steps.

**System requirements**: Our results are collected on a single node with 8x H100 80GB SXM5 (NVLink) GPUs. 8xA100 80GB GPUs should give similar trend.

-----



-----
### Tensor Parallelism
Weak Scaling
Strong Scaling

-----
### Context Parallelism 

| Row Id | `make <target>`                   | Model   | GBS | Seq. Len. | CP | Mem. Usage (GB/gpu) | step time (ms) |
|:------:|:----------------------------------|:-------:|----:|----------:|---:|:-------------------:|---------------:|
| r1     | `301-cp1-gpt2xl-gbs4-len4096-oom` | gpt2-xl |   4 |      4096 |  1 |      *oom*          |      -         |
| r2     | `302-cp2-gpt2xl-gbs4-len4096`     |   "     |   " |         " |  2 |        56           |     370        |
| r3     | `304-cp4-gpt2xl-gbs4-len4096`     |   "     |   " |         " |  4 |        46           |     410        |
| r4     | `308-cp8-gpt2xl-gbs4-len4096`     |   "     |   " |         " |  8 |        40           |     550        |
| r5     | `311-gpt2xl-gbs4-len8192-cp8`     |   "     |   4 |      8192 |  8 |        48           |     565        |
| r6     | `312-gpt2xl-gbs4-len16384-cp8`    |   "     |   4 |     16384 |  8 |        63           |     710        |
| r7     | `313-gpt2xl-gbs8-len8192-cp8`     |   "     |   8 |      8192 |  8 |        60           |     625        |
| r8     | `314-gpt2xl-gbs8-len8192-cp4-oom` |   "     |   8 |      8192 |  4 |      *oom*          |      -         |

1. [TODO] brief about context parallel, contrast to sequence parallel.
2. **r1-r4**: Let's start from a baseline, a case where training of gpt2-xl (1.5B) on a single H100 of 80GB can not fit a batch of sequences of 4096 length. Context parallelism allows us to split the sequence length dimension across multiple GPUs. From r2 to r4, we use 2, 4, 8 GPUs to split the sequence length of 4096 into smaller chunks (mention activation memory), resulting in decreasing memory usage per GPU at a cost of increased step time due to communication overhead.
3. **r5-r6**: With the surplus of memory at higher CP degree, we can handle longer sequences, we can scale the sequence length up to 16K and still only consuming slight over 3/4 of GPU memory. 
4. **r6-r7**: Alternative, we can also increase the batch size like.
4. **r7 vs r8** is to emphasize the importance of CP in overcoming memory challenge of attention with long sequences. The training job is oom by halfing the number of GPUs.
5. [todo] need to find a case where it help speed up, between r4-r5, use cp4, 8K, so see speed up