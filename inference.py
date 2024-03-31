# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch

import fire
import os
import sys
import time
import json
import torch
from transformers import LlamaTokenizer
from tqdm import tqdm
# from llama_recipes.inference.safety_utils import get_safety_checker, AgentType
from llama_recipes.inference.model_utils import load_model, load_peft_model
# from IPython import embed
# import pdb
import jsonlines
def write_answers(file, answers):
    with open(file, 'a') as f:
        for id, answer in enumerate(answers):
            # output = '\t'.join([start_id+id,answer])
            answer = answer.replace('\n', ' ')
            f.write(answer + '\n')

def write_answers_json(generate_file, batch_answers,scores,topk_index,index_str):
    with jsonlines.open(generate_file, mode='a') as writer:
        for index,answer in enumerate(batch_answers):
            # split_list= []
            # for sent in all_splits[index]:
            #     split_list.append(sent.split('/'))
            answer = answer.replace('\n', ' ')
            json_item = {'answer':answer,'score':scores[index],'topk_index':topk_index[index],'topk_token':index_str[index]}
            writer.write(json_item)
def main(
        model_name,
        num_beams: int = 1,
        generate_file: str = '',
        bsz: int = 1,
        eval_file: str = '',
        start: int = 0,
        end: int = -1,
        max_length: int = 128,
        token_k:int= 10, #表示输出前k个token
        peft_model: str = None,
        quantization: bool = False,
        max_new_tokens=100,  # The maximum numbers of tokens to generate
        prompt_file: str = None,
        seed: int = 42,  # seed value for reproducibility
        do_sample: bool = False,  # Whether or not to use sampling ; use greedy decoding otherwise.
        min_length: int = None,  # The minimum length of the sequence to be generated, input prompt + min_new_tokens
        use_cache: bool = True,
        # [optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
        top_p: float = 1.0,
        # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
        temperature: float = 1.0,  # [optional] The value used to modulate the next token probabilities.
        top_k: int = 50,  # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
        repetition_penalty: float = 1.0,  # The parameter for repetition penalty. 1.0 means no penalty.
        length_penalty: int = 1,
        # [optional] Exponential penalty to the length that is used with beam-based generation.
        enable_azure_content_safety: bool = False,  # Enable safety check with Azure content safety api
        enable_sensitive_topics: bool = False,  # Enable check for sensitive topics using AuditNLG APIs
        enable_salesforce_content_safety: bool = False,  # Enable safety check with Salesforce safety flan t5
        enable_llamaguard_content_safety: bool = False,
        llamaguard_model_name: str = None,
        max_padding_length: int = None,  # the max padding length to be used with tokenizer padding the prompts.
        use_fast_kernels: bool = False,
        output_logits:bool=False,
        # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
        **kwargs
):

    # Set the seeds for reproducibility
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    if output_logits:
        print('Output logits !!')
    else:
        print('Do not output logits!!')

    if os.path.exists(generate_file):
        # 删除文件
        os.remove(generate_file)
        print(f"{generate_file} exists already, but has been removed")
    questions = []
    with open(eval_file) as f:
        text = json.load(f)
    if end == -1:
        text = text[start:]
    else:
        text = text[start:end]
    for item in text:
        questions.append(item['conversations'][0]['value'])
    if num_beams == 1:
        print('greedy search...')
    else:
        print('beam search...')

    model = load_model(model_name, quantization)
    if peft_model:
        model = load_peft_model(model, peft_model)

    model.eval()

    if use_fast_kernels:
        """
        Setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels 
        based on the hardware being used. This would speed up inference when used for batched inputs.
        """
        try:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model)
        except ImportError:
            print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")

    tokenizer = LlamaTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    #
    def evaluate(instructions):
        # while True:
        question = input('please input:')
        instructions = [question]
        batch = tokenizer(instructions, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        batch = {k: v.to("cuda") for k, v in batch.items()}

        with torch.no_grad():
            generation_output = model.generate(     #当output scores时，该元素有两个属性有值，分别是sequences和scores，都是tuple，前者大小是2，后者是3
                # input_ids=input_ids,
                **batch,
                pad_token_id=tokenizer.eos_token_id,
                # num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                top_p=top_p,
                temperature=temperature,
                min_length=min_length,
                use_cache=use_cache,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                output_scores= output_logits,
                return_dict_in_generate=output_logits,
                **kwargs
            )
            if output_logits:
                logits = generation_output.scores
        all_answers = []
        all_scores = [] #每一个case的大小是[3, topk]   3换成N，表示输出N个token
        all_topk_index = []# 每一个case的大小是[3, topk]
        all_topk_index_str = [] #与上述列表一一对应，表示decode之后的字符
        if output_logits:
            batch_size = len(generation_output.sequences)
        else:
            batch_size = generation_output.size()[0]

         #取最后输出的N个token,N是len(logits)，表示每个case生成了几个token
        if output_logits:
            topks = []
            all_strs = []
            for i in range(len(logits)):
                topks.append(torch.topk(logits[i], token_k, dim=-1))

                i_strs = []
                for j in range(batch_size):
                    temp_list = []
                    for k in range(token_k):

                        temp_list.append(tokenizer.decode(topks[i][1][j][k], skip_special_tokens=False))
                    i_strs.append(temp_list)
                all_strs.append(i_strs)
            for j in range(batch_size):
                item_score = []
                item_index = []
                item_index_str = []
                for i in range(len(logits)):
                    item_score.append(topks[i][0].tolist()[j])
                    item_index.append(topks[i][1].tolist()[j])
                    item_index_str.append(all_strs[i][j])
                all_scores.append(item_score)
                all_topk_index.append(item_index)
                all_topk_index_str.append(item_index_str)

        for i in range(batch_size):
            if output_logits:
                s = generation_output.sequences[i]
            else:
                s = generation_output[i]
            output = tokenizer.decode(s, skip_special_tokens=True)  # including instruction

            answer = output.replace(instructions[i],'')
            all_answers.append(answer)
        # print(all_answers[0])
        if output_logits:
            return all_answers,all_scores,all_topk_index,all_topk_index_str
        else:
            return all_answers

    temp_count = 0
    batch = []
    if output_logits:
        for id, question in enumerate(tqdm(questions)):
            if id < (len(questions) - 1):

                if temp_count < bsz:
                    batch.append(question)
                else:
                    batch_answers,scores,topk_index,index_str = evaluate(batch)
                    write_answers_json(generate_file, batch_answers,scores,topk_index,index_str)
                    batch = []
                    temp_count = 0
                    batch.append(question)
            else:
                batch.append(question)
                batch_answers,scores,topk_index,index_str = evaluate(batch)
                write_answers_json(generate_file, batch_answers,scores,topk_index,index_str)
            temp_count += 1
    else:
        for id, question in enumerate(tqdm(questions)):
            if id < (len(questions) - 1):

                if temp_count < bsz:
                    batch.append(question)
                else:
                    batch_answers = evaluate(batch)
                    write_answers(generate_file, batch_answers)
                    batch = []
                    temp_count = 0
                    batch.append(question)
            else:
                batch.append(question)
                batch_answers = evaluate(batch)
                write_answers(generate_file, batch_answers)
            temp_count += 1




if __name__ == "__main__":
    fire.Fire(main)
