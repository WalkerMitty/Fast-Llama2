CUDA_VISIBLE_DEVICES=3 python inference.py \
	--model_name /data/hfmodel/PLMs/llama27b_hf \
	--peft_model loras/decisioner-100-epoch40 \
	--max_new_tokens 8 \
	--do_sample false \
	--num_beams 1 \
	--start 0 \
	--end -1 \
	--eval_file ./data/demo_infer.json \
	--bsz 16 \
	--max_length 256 \
	--generate_file './record/conflict_2-baseline-decision.file'

