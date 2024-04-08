[English](./README_en.md)

**Note:** 这些代码用于Llama2指令微调，适配自官方仓库。删掉了不必要功能，方便上手，添加了一些实用功能。

### 添加的部分：
    - 加载训练好的lora继续训练
    - 推理输出文本的同时输出权重
    - 修改了scheduler逻辑，只有当loss增加时才减小lr

## step1: 数据构建与环境准备
环境为python3.9，其余环境见requirements.txt

指令微调数据集是一系列的question, answer 对，只需要将question和answer填入下面模板（见/data/demo*.json)
```json
[{"conversations": [{"from": "human", "value": "QUESTION1"}, {"from": "gpt", "value": "ANSWER1"}]}]
```

## step2: 微调

```shell
bash run_test.sh
```

## step3: 推理

```shell
bash run_infer.sh  #不输出logit
bash run_infer_logit.sh  #输出logit
```

## details
下面是run_test.sh的细节
- 默认是Lora微调，如果是全参数微调，则删掉use_peft和peft_method
- dataset参数不用改，grammar_dataset只是一个模板
- lr比较重要，我在不同的数据集上采用的是1e-3,1e-4,1e-5,1e-6
- output_dir loras权重存储位置
- train_split 训练集的路径
- batch_size_training 根据自己的显存改，注意数据量必须  >= batch_size_training* num_gpus
- lora_path 如果为空字符串，则自动初始化权重。否则将加载这个lora路径继续训练
- step size 控制改变lr的频率的，如果step size为1，则每个epoch结束后判断是否需要减小lr
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

下面是 run_infer.sh的细节

- 仅支持单卡推理，多卡请结合start, end参数手动并行。start和end分别是需要推理数据的起始index和结束index。默认参数表示推理全量数据
- eval_file 需要推理的数据集 
- generate_file 生成的LLM answer数据集（每一行对应一个answer）
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

下面是 run_infer_logit.sh的细节

- token_k 每个token输出前k个最大的logits (这里是没有softmax的)
- generate_file 只能是json格式，存储有answer和logits信息

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
