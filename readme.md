## Megatron, Transformed! 😎
*A Hands-on Megatron-LM Tutorial on Replicating Empirical Trends in Distributed Training and Model Parallelism*

[Megatron-LM][mlm-gh] is, without question, one of the most impressive frameworks advancing and enabling ultra-large model training. But just how many arguments are there in [`pretrain_gpt.py`][mlm-pretrain-gpt]? Over 500. 😅 

>*How many GPUs do we need to get started?
How and what should we set?
Which axis of parallelism should we scale first?
Why am I getting out-of-memory (OOM) all the time? Okay, it’s finally running… but is it even correct? is the performance as expected?*

This tutorial series is written to address those pain points. It provides a set of curated, ready-to-run (hack) experiments designed to reduce the friction of getting started with Megatron-LM. Each tutorial explains the core concept succinctly, ablates one parallelism strategy at a time. Wherever possible, we align with the main paper to reproduce the reported scaling trends to verify correctness and understanding.

All experiments are designed to run on a single node with 8×H100 80 GB GPUs. Performance and memory metrics are meticulously tabulated for your reference and verification. The ***goal*** is to *make Megatron-LM more accessible, reproducible, and tinker-friendly*. Let's get started!

Explore:

* [Data Parallelism & ZeRO-2](./01-dp-zero2.md): Scaling Up Batch and Model Size with Sublinear Memory Growth across Replicas.
* [Tensor Parallelism](./02-tp-sp.md): Intra-layer Parameter Sharding for Larger Models, per paper's Strong and Weak Scaling.
* [Sequence Parallelism](./02-tp-sp.md#sequence-parallelism): Turns Activation Duplication into Partitions, Bypassing Recomputation.
* [Context Parallelism](./03-cp.md): Extending Sequence Parallelism to Attention and Variants in Megatron.
* [(Virtual) Pipeline Parallelism](./04-pp-vpp.md): Inter-layer Model Sharding, Scheduling, and Layer Interleaving for Reduced Bubbles.
* [Expert Parallelism](./05-ep-moe.md): Mixture-of-Experts (MoE) for Scaling Model Capacity with Conditional Compute.

-----
### Setup and Run

We rely on the out-of-the-box Megatron-LM's `pretrain_gpt.py` and use bash scripts to wrap it with different parallelism strategies. We also use `Makefile` extensively to organize and manage the experiments. 

Easiest way to get started is to use prebuilt docker image:
```
docker run -d --gpus all -it --rm \
  --network=host --ipc=host \
  --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  vuiseng9/megatron-tutorials
```
Or build using [docker/Dockerfile](./docker/Dockerfile).

**How to run? Just `make <id>-tab-completion`** 
* The docker entrypoint will lead to working directory, `/workspace/megatron-lm/examples/gpt3`
* Each experiment is defined as a [Makefile](./Makefile) target prefixed with a unique id. Our intent is to reduce the number of bash scripts, steps, arguments, making the runs less friction to reproduce, basically just type make then the id and finally a tab to complete the target. e.g. type `make 101<tab>` turn into `make 101-gpt2xl-dp1-gbs1-bf16`. Each result in our tables also comes their corresponding unique id. 
* Metrics can be found in std output which is also logged to `./outdir/<experiment label>/logs.txt`. To see GPU memory usage, run `monitor-gpu` on a seperate terminal. Runs are preconfigured to stop after 100 training steps.

**System requirements**: Our results are collected on a single node with 8x H100 80GB SXM5 (NVLink) GPUs. 8xA100 80GB GPUs should give similar trend. PCIe GPUs will run, but unproductively slow.

```
$ make 
101-gpt2xl-dp1-gbs1-bf16                 318-cp8-gpt2-1.2B-gbs8-len4096-ag
104-gpt2xl-dp4-gbs4                      328-cp8-gpt2-1.2B-gbs8-len4096-a2a
111-gpt2xl-dp1-fit-80GB                  338-cp8-gpt2-1.2B-gbs8-len16k
112-gpt2xl-dp1-fit-80GB-GA4              348-cp8-gpt2-1.2B-gbs8-len16k-ag
114-gpt2xl-dp4-fit-4x80GB                358-cp8-gpt2-1.2B-gbs8-len16k-a2a
115-gpt2xl-dp4-gbs60-zero2               401-gpt2-8.3B-pp8-m1
118-gpt2xl-dp8-fit-8x80GB                402-gpt2-8.3B-pp8-m2
121-gpt2xl-dp1-gbs16-oom                 404-gpt2-8.3B-pp8-m4
122-gpt2xl-dp2-gbs32-zero2               408-gpt2-8.3B-pp8-m8
124-gpt2xl-dp4-gbs64-zero2               416-gpt2-8.3B-pp8-m16
128-gpt2xl-dp8-gbs128-zero2              420-gpt2-8.3B-tpsp8
129-gpt2xl-dp8-gbs168-zero2              424-gpt2-8.3B-tpsp2-pp4-m4
211-weak-scale-tp1-gpt2-1.2B-paper       432-gpt2-8.3B-pp8-m32
212-weak-scale-tp2-gpt2-2.5B-paper       438-gpt2-8.3B-pp8-vpp3-m8
214-weak-scale-tp4-gpt2-4.2B-paper       443-gpt2-8.3B-tpsp2-pp4-vpp3-m4
218-weak-scale-tp8-gpt2-8.3B-paper       446-gpt2-8.3B-tpsp2-pp4-vpp6-m4
221-weak-scale-tp1-gpt2-1.2B-gbs20       449-gpt2-8.3B-tpsp2-pp4-vpp9-m4
222-weak-scale-tp2-gpt2-2.5B-gbs20       458-gpt2-8.3B-tpsp2-pp4-vpp18-m4
224-weak-scale-tp4-gpt2-4.2B-gbs20       498-gpt2-8.3B-pp8-vpp9-m8
228-weak-scale-tp8-gpt2-8.3B-gbs20       500-olmoe-dp8
231-strong-scale-gpt2-1.2B-tp1-paper     501-olmoe-dp8-zero2
232-strong-scale-gpt2-1.2B-tp2-paper     502-olmoe-dp8-zero2-ep8
234-strong-scale-gpt2-1.2B-tp4-paper     503-olmoe-2xE-dp8-zero2
238-strong-scale-gpt2-1.2B-tp8-paper     504-olmoe-2xE-dp8-zero2-ep8
241-strong-scale-gpt2-1.2B-tp1-gbs20     505-olmoe-4xE-dp8-zero2
242-strong-scale-gpt2-1.2B-tp2-gbs20     506-olmoe-4xE-dp8-zero2-ep8
244-strong-scale-gpt2-1.2B-tp4-gbs20     511-olmoe-4xE-dp8-zero2-ep4-etp2
248-strong-scale-gpt2-1.2B-tp8-gbs20     512-olmoe-4xE-dp8-zero2-ep2-etp4
281-gpt-22B-tp8-gbs4-len2048-oom         count-arguments
282-gpt-22B-tp8-gbs4-len2048-sp          how-to-recompute-activation
283-gpt-22B-tp8-gbs4-len2048-ra          install-dependencies
300-cp1-gpt2-1.2B-gbs8-len4096-ra        prepare-ds-openwebtext-10k
301-cp1-gpt2-1.2B-gbs8-len4096-oom       profile-282-gpt-22B-tp8-gbs4-len2048-sp
302-cp2-gpt2-1.2B-gbs8-len4096           profile-283-gpt-22B-tp8-gbs4-len2048-ra
304-cp4-gpt2-1.2B-gbs8-len4096           show-arguments
308-cp8-gpt2-1.2B-gbs8-len4096           
```

[mlm-gh]: https://github.com/NVIDIA/Megatron-LM
[mlm-pretrain-gpt]: https://github.com/NVIDIA/Megatron-LM/blob/core_v0.14.0/pretrain_gpt.py

```
@misc{chua2025megatrontransformed,
  title        = {Megatron, Transformed! A Hands-on Megatron-LM Tutorial on Replicating Empirical Trends in Distributed Training and Model Parallelism},
  author       = {Chua, Vui Seng},
  year         = {2025},
  url          = {https://github.com/vuiseng9/megatron-tutorials},
}
```
![](assets/gemini-3-pro-banner.png)

> *Generated by Gemini 3 Pro conditioned on previous [image](assets/gemini_pro_2.5_generated.png)*
