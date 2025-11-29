# Not needed if have use prebuilt docker megatron-tuts
install-dependencies:
	pip install datasets==3.6.0
	pip install wandb

# Not needed if have use prebuilt docker megatron-tuts
prepare-ds-openwebtext-10k:
	rm -rf ./owt-ds
	python scripts/hf_ds_to_json.py
	wget -P ./owt-ds https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
	wget -P ./owt-ds https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt

	python ../../tools/preprocess_data.py \
		--input ./owt-ds/openwebtext-10k.jsonl \
		--output-prefix ./owt-ds/openwebtext-10k \
		--vocab-file ./owt-ds/gpt2-vocab.json \
		--tokenizer-type GPT2BPETokenizer \
		--merge-file ./owt-ds/gpt2-merges.txt \
		--workers $(shell nproc) \
		--append-eod

show-arguments:
	python ../../pretrain_gpt.py -h | grep -o '^  --[^ ]*' 

count-arguments:
	python ../../pretrain_gpt.py -h | grep -o '^  --[^ ]*' | wc -l

# DP 1: Single GPU, GBS 1
101-gpt2xl-dp1-gbs1-bf16:
	bash ./scripts/100_gpt2xl_ddp_gpuX_gbsX.sh

104-gpt2xl-dp4-gbs4:
	bash ./scripts/100_gpt2xl_ddp_gpuX_gbsX.sh 4 4

111-gpt2xl-dp1-fit-80GB:
	bash ./scripts/100_gpt2xl_ddp_gpuX_gbsX.sh 1 15

112-gpt2xl-dp1-fit-80GB-GA4:
	bash ./scripts/112_gpt2xl_ddp_gpu1_gbs60_mbs15_ga4.sh

114-gpt2xl-dp4-fit-4x80GB:
	bash ./scripts/100_gpt2xl_ddp_gpuX_gbsX.sh 4 60

115-gpt2xl-dp4-gbs60-zero2:
	bash ./scripts/100_gpt2xl_ddp_gpuX_gbsX.sh 4 64 1

118-gpt2xl-dp8-fit-8x80GB:
	bash ./scripts/100_gpt2xl_ddp_gpuX_gbsX.sh 8 120

121-gpt2xl-dp1-gbs16-oom:
	bash ./scripts/100_gpt2xl_ddp_gpuX_gbsX.sh 1 16 1
	
122-gpt2xl-dp2-gbs32-zero2:
	bash ./scripts/100_gpt2xl_ddp_gpuX_gbsX.sh 2 32 1

124-gpt2xl-dp4-gbs64-zero2:
	bash ./scripts/100_gpt2xl_ddp_gpuX_gbsX.sh 4 64 1

128-gpt2xl-dp8-gbs128-zero2:
	bash ./scripts/100_gpt2xl_ddp_gpuX_gbsX.sh 8 128 1

129-gpt2xl-dp8-zero2-fit-gbs:
	bash ./scripts/100_gpt2xl_ddp_gpuX_gbsX.sh 8 168 1

130-gpt2xl-dp8-zero2-fit-param:
	bash scripts/130_gpt2xl+12tx_dp8_zero2_gbs128.sh

# TP Weak Scaling
211-weak-scale-tp1-gpt2-1.2B-paper:
	bash ./scripts/2-1_weak_tp1_gpt2-1.2B.sh

212-weak-scale-tp2-gpt2-2.5B-paper:
	bash ./scripts/2-2_weak_tp2_gpt2-2.5B.sh

214-weak-scale-tp4-gpt2-4.2B-paper:
	bash ./scripts/2-4_weak_tp4_gpt2-4.2B.sh

218-weak-scale-tp8-gpt2-8.3B-paper:
	bash ./scripts/2-8_weak_tp8_gpt2-8.3B.sh

221-weak-scale-tp1-gpt2-1.2B-gbs20:
	bash ./scripts/2-1_weak_tp1_gpt2-1.2B.sh 20

222-weak-scale-tp2-gpt2-2.5B-gbs20:
	bash ./scripts/2-2_weak_tp2_gpt2-2.5B.sh 20

224-weak-scale-tp4-gpt2-4.2B-gbs20:
	bash ./scripts/2-4_weak_tp4_gpt2-4.2B.sh 20

228-weak-scale-tp8-gpt2-8.3B-gbs20:
	bash ./scripts/2-8_weak_tp8_gpt2-8.3B.sh 20

# TP Strong Scaling
231-strong-scale-gpt2-1.2B-tp1-paper:
	bash ./scripts/2--_strong_tpX_gpt2-1.2B.sh 1 8

232-strong-scale-gpt2-1.2B-tp2-paper:
	bash ./scripts/2--_strong_tpX_gpt2-1.2B.sh 2 8

234-strong-scale-gpt2-1.2B-tp4-paper:
	bash ./scripts/2--_strong_tpX_gpt2-1.2B.sh 4 8

238-strong-scale-gpt2-1.2B-tp8-paper:
	bash ./scripts/2--_strong_tpX_gpt2-1.2B.sh 8 8

241-strong-scale-gpt2-1.2B-tp1-gbs20:
	bash ./scripts/2--_strong_tpX_gpt2-1.2B.sh 1 20

242-strong-scale-gpt2-1.2B-tp2-gbs20:
	bash ./scripts/2--_strong_tpX_gpt2-1.2B.sh 2 20

244-strong-scale-gpt2-1.2B-tp4-gbs20:
	bash ./scripts/2--_strong_tpX_gpt2-1.2B.sh 4 20

248-strong-scale-gpt2-1.2B-tp8-gbs20:
	bash ./scripts/2--_strong_tpX_gpt2-1.2B.sh 8 20

# SP
281-gpt-22B-tp8-gbs4-len2048-oom:
	bash ./scripts/28-_gpt-22B_tp8_gbs4_len2048_spX.sh 0

282-gpt-22B-tp8-gbs4-len2048-sp:
	bash ./scripts/28-_gpt-22B_tp8_gbs4_len2048_spX.sh 1

283-gpt-22B-tp8-gbs4-len2048-ra:
	bash ./scripts/283_gpt-22B_tp8_gbs4_len2048_RA.sh 1

how-to-recompute-activation:
	diff --color scripts/28-_gpt-22B_tp8_gbs4_len2048_spX.sh scripts/283_gpt-22B_tp8_gbs4_len2048_RA.sh

profile-282-gpt-22B-tp8-gbs4-len2048-sp:
	bash ./scripts/28-_profile_gpt-22B_tp8_gbs4_len2048_spX.sh 1

profile-283-gpt-22B-tp8-gbs4-len2048-ra:
	bash ./scripts/283_profile_gpt-22B_tp8_gbs4_len2048_RA.sh 1

# CP
300-cp1-gpt2-1.2B-gbs8-len4096-ra:
	bash ./scripts/3--_gpt2-1.2B_gbsX_lenX_cpX.sh 8 4096 1 1

301-cp1-gpt2-1.2B-gbs8-len4096-oom:
	bash ./scripts/3--_gpt2-1.2B_gbsX_lenX_cpX.sh 8 4096 1

302-cp2-gpt2-1.2B-gbs8-len4096:
	bash ./scripts/3--_gpt2-1.2B_gbsX_lenX_cpX.sh 8 4096 2

304-cp4-gpt2-1.2B-gbs8-len4096:
	bash ./scripts/3--_gpt2-1.2B_gbsX_lenX_cpX.sh 8 4096 4

308-cp8-gpt2-1.2B-gbs8-len4096:
	bash ./scripts/3--_gpt2-1.2B_gbsX_lenX_cpX.sh 8 4096 8

318-cp8-gpt2-1.2B-gbs8-len4096-ag:
	bash ./scripts/3--_gpt2-1.2B_gbsX_lenX_cpX.sh 8 4096 8 0 all_gather

328-cp8-gpt2-1.2B-gbs8-len4096-a2a:
	bash ./scripts/3--_gpt2-1.2B_gbsX_lenX_cpX.sh 8 4096 8 0 a2a

338-cp8-gpt2-1.2B-gbs8-len16k:
	bash ./scripts/3--_gpt2-1.2B_gbsX_lenX_cpX.sh 8 16384 8

348-cp8-gpt2-1.2B-gbs8-len16k-ag:
	bash ./scripts/3--_gpt2-1.2B_gbsX_lenX_cpX.sh 8 16384 8 0 all_gather

358-cp8-gpt2-1.2B-gbs8-len16k-a2a:
	bash ./scripts/3--_gpt2-1.2B_gbsX_lenX_cpX.sh 8 16384 8 0 a2a


401-gpt2-8.3B-pp8-m1:
	bash ./scripts/4--_ppX_gpt2-8.3B.sh 8 32 32 1

402-gpt2-8.3B-pp8-m2:
	bash ./scripts/4--_ppX_gpt2-8.3B.sh 8 32 16 1

404-gpt2-8.3B-pp8-m4:
	bash ./scripts/4--_ppX_gpt2-8.3B.sh 8 32 8 1

408-gpt2-8.3B-pp8-m8:
	bash ./scripts/4--_ppX_gpt2-8.3B.sh 8 32 4 1

416-gpt2-8.3B-pp8-m16:
	bash ./scripts/4--_ppX_gpt2-8.3B.sh 8 32 2 1

432-gpt2-8.3B-pp8-m32:
	bash ./scripts/4--_ppX_gpt2-8.3B.sh 8 32 1 1

438-gpt2-8.3B-pp8-vpp3-m8:
	bash ./scripts/4--_ppX_gpt2-8.3B.sh

498-gpt2-8.3B-pp8-vpp9-m8:
	bash ./scripts/4--_tpspX_ppXvppX_gpt2-8.3B.sh

420-gpt2-8.3B-tpsp8:
	bash ./scripts/4--_tpspX_ppXvppX_gpt2-8.3B.sh

424-gpt2-8.3B-tpsp2-pp4-m4:
	bash ./scripts/4--_tpspX_ppXvppX_gpt2-8.3B.sh 2 1 4 1

443-gpt2-8.3B-tpsp2-pp4-vpp3-m4:
	bash ./scripts/4--_tpspX_ppXvppX_gpt2-8.3B.sh 2 1 4 3

446-gpt2-8.3B-tpsp2-pp4-vpp6-m4:
	bash ./scripts/4--_tpspX_ppXvppX_gpt2-8.3B.sh 2 1 4 6

449-gpt2-8.3B-tpsp2-pp4-vpp9-m4:
	bash ./scripts/4--_tpspX_ppXvppX_gpt2-8.3B.sh 2 1 4 9

458-gpt2-8.3B-tpsp2-pp4-vpp18-m4:
	bash ./scripts/4--_tpspX_ppXvppX_gpt2-8.3B.sh 2 1 4 18