**Note:** These codes are for instruction tuning with Llama2, adapted from the official repository. Unnecessary features have been removed for ease of use, and some practical features have been added.

### Added Features:
    - Load pre-trained Lora for continued training
    - Output logits during inference
    - Modified scheduler logic to decrease learning rate only when loss increases

## step1: Data Preparation
SFT dataset consists of a series of question-answer pairs. Simply fill in the questions and answers in the template below (see /data/demo*.json).
```json
[{"conversations": [{"from": "human", "value": "QUESTION1"}, {"from": "gpt", "value": "ANSWER1"}]}]
```

## step2: Fine tuning

```shell
bash run_test.sh
```

## step3: Inference

```shell
bash run_infer.sh  # without logits
bash run_infer_logit.sh  # with logits
```

## details

Here are the details of run_test.sh:

- By default, it's Lora fine-tuning. Remove ``use_peft`` and ``peft_method`` for full-parameters tuning.
- No need to change the ``dataset`` parameter, grammar_dataset is just a template.
- ``lr`` is quite important. I used 1e-3, 1e-4, 1e-5, 1e-6 on different datasets.
- ``output_dir`` is where Lora weights are stored.
- If ``lora_path`` is an empty string, weights will be automatically initialized. Otherwise, it will load weights from this Lora path for continued training.
- ``step size`` controls the frequency of lr changes. If ``step size`` is 1, lr will be evaluated for decrease after each epoch.
```shell
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup torchrun --nnodes 1 --nproc_per_node 4 --master_port 29504 finetuning.py \
	--enable_fsdp  \
	--model_name /data/hfmodel/PLMs/llama27b_hf \
	--peft_method lora \
	--use_peft true \
	--dataset grammar_dataset \
	--save_model \
	--dist_checkpoint_root_folder model_checkpoints \
	--dist_checkpoint_folder fine-tuned \
	--fsdp_config.pure_bf16 \
	--lr 5e-6 \
	--output_dir loras/decisioner-100-epoch60-prompt \
	--train_split ./data/demo_train.json \
	--batch_size_training 128 \
	--lora_path '' \
	--step_size 1 \
	--num_epochs 10  > logs/decisioner-100-epoch60-prompt.log 2>&1 &

```

Here are the details of run_infer.sh:

- Supports single-GPU inference only. For multi-GPU, manually parallelize with ``start`` and ``end`` parameters, which indicate the starting and ending indices of data for inference. Default parameters infer all data.
- ``eval_file`` is the dataset to be inferred.
- ``generate_file`` stores the generated LLM answer dataset (each line corresponds to an answer).

```shell
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
```

Here are the details of run_infer_logit.sh:

- ``token_k`` outputs the top k logits for each token (before softmax).
- ``generate_file`` must be in JSON format, storing both answer and logits information.

```shell
CUDA_VISIBLE_DEVICES=4 python inference.py \
	--model_name /data/pretrained_models/llama27b_hf \
	--peft_model loras/checker-sample \
	--max_new_tokens 4 \
	--num_beams 1 \
	--start 0 \
	--end -1 \
	--eval_file ./data/demo_infer.json \
	--bsz 16 \
	--output_logits \
	--max_length 256 \
	--token_k 3 \
	--generate_file './record/conflict_checker_answer_4-h.json'


```
## Reference

https://github.com/meta-llama/llama-recipes