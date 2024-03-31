CUDA_VISIBLE_DEVICES=4 python inference.py \
	--model_name /data/pretrained_models/llama27b_hf \
	--peft_model loras/checker-sample \
	--max_new_tokens 4 \
	--num_beams 1 \
	--start 0 \
	--end -1 \
	--eval_file /data/train_file/conflict_checker_4-h.json \
	--bsz 16 \
	--output_logits \
	--max_length 256 \
	--token_k 3 \
	--generate_file './record/conflict_checker_answer_4-h.json'

