from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from dataset_class import SupervisedDataset, DataCollatorForSupervisedDataset
from config import *
import math
import os
import torch
import time
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score
from accelerate import load_checkpoint_and_dispatch, init_empty_weights, infer_auto_device_map
from tqdm import tqdm
from my_arguments import DataTrainingArguments
from deepspeed.accelerator import get_accelerator

get_accelerator().empty_cache()

def map_to_int(res):
    if 'Yes' in res:
        return 1
    elif 'No' in res:
        return 0
    else:
        return -1

def my_compute_metric(eval_pred):
    Acc=[]
    for p, a in zip(eval_pred['predictions'], eval_pred['labels']):
        Acc.append(p==a)

    f1 = recall = None
    if not any(i == -1 for i in eval_pred['predictions']):
        f1 = f1_score(eval_pred['labels'], eval_pred['predictions'])
        recall = recall_score(eval_pred['labels'], eval_pred['predictions'])

    return {
        'Acc': np.mean(Acc),
        'F1': f1,
        'Recall': recall
    }


tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=bf16_check_point_dir,use_fast=True)
model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path=bf16_check_point_dir)

with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=bf16_check_point_dir)
    

model = load_checkpoint_and_dispatch(
    model,checkpoint=bf16_check_point_dir, device_map='auto',no_split_module_classes=['LlamaDecoderLayer',]
)

model.hf_device_map['lm_head'] = model.hf_device_map['model.embed_tokens']

model.eval()

data_args = DataTrainingArguments()
test_dataset = SupervisedDataset(args=data_args, config=model_config, tokenizer=tokenizer, mode='test')
eval_pred = {
                'predictions':[],
                'labels':[]
            }

if os.path.exists(os.path.join(model_output_dir,'predictions.txt')):
    with open(os.path.join(model_output_dir,'predictions.txt'), 'r') as f:
        for line in f:
            idx, p, a = line.strip().split()
            eval_pred['predictions'].append(int(p))
            eval_pred['labels'].append(int(a))
    f.close()
    print(f'continue from {idx}')

pred_result_writer = open(os.path.join(model_output_dir,'predictions.txt'),'a+')

with torch.no_grad():
    for i in range(len(test_dataset)):
        if i < len(eval_pred['predictions']):
            continue
        example = test_dataset[i]
        input_id, label = example   
        print(input_id.size())
        if input_id.size(-1) > data_args.max_seq_len:
            eval_pred['predictions'].append(-1)
            eval_pred['labels'].append(map_to_int(label))
            pred_result_writer.write(f"{i} {eval_pred['predictions'][-1]} {eval_pred['labels'][-1]}\n")
            continue

        input_id = input_id.to('cuda')
        
        stop_id = tokenizer.convert_tokens_to_ids(EOT_TOKEN)
        assert isinstance(stop_id, int), "Invalid tokenizer, EOT id not found"

        outputs = model.generate(
            input_id,
            max_new_tokens=3,
            pad_token_id=stop_id,
            eos_token_id=stop_id
        )

        output = outputs[0][input_id.size(-1):]
        decode_str = tokenizer.decode(output,skip_special_tokens=True)
        answer = decode_str.split('\n')[0]

        eval_pred['predictions'].append(map_to_int(answer))
        eval_pred['labels'].append(map_to_int(label))

        pred_result_writer.write(f"{i} {eval_pred['predictions'][-1]} {eval_pred['labels'][-1]}\n")

        print(f'{len(eval_pred["predictions"])}: {answer} {eval_pred["predictions"][-1]}')


print(my_compute_metric(eval_pred))
