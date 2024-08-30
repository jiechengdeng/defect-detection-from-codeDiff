import deepspeed
import deepspeed.comm as dist
import argparse
import os
import gc
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.deepspeed import HfDeepSpeedConfig
from my_arguments import DataTrainingArguments
from dataset_class import SupervisedDataset, DataCollatorForSupervisedDataset
from config import *
from deepspeed.accelerator import get_accelerator

def my_compute_metric(eval_pred):
    Acc=[]
    for p, a in zip(eval_pred['predictions'], eval_pred['labels']):
        Acc.append(p==a)
    
    return {
        'Acc': np.mean(Acc),
    }


def get_ds_model(model_config):
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_dir,torch_dtype=torch.bfloat16,trust_remote_code=True)
    hidden_size = model_config.hidden_size
    
    ds_config = {
        "fp16": {
            "enabled": model_config.torch_dtype == torch.float16,
        },
        "bf16": {
            "enabled": model_config.torch_dtype == torch.bfloat16,
        },
        "zero_optimization": {
            "stage": 3,
            "stage3_prefetch_bucket_size": 2 * hidden_size * hidden_size, # 0, 
            "stage3_param_persistence_threshold": hidden_size,
            "stage3_max_live_parameters": 2 * hidden_size * hidden_size,
        },
        "steps_per_print": 2000,
        "train_batch_size": 2,
        "wall_clock_breakdown": False,
    }

    dschf = HfDeepSpeedConfig(
        ds_config
    )  # this tells from_pretrained to instantiate directly on gpus

    get_accelerator().empty_cache()

    ds_engine = deepspeed.init_inference(model=model, 
                                        tensor_parallel={"tp_size": 2},
                                        dtype=torch.bfloat16,
                                        checkpoint=None,
                                        replace_with_kernel_inject=False)
   
    model = ds_engine.module

    return model
    
def run_generation():
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=bf16_check_point_dir,use_fast=True)
    model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path=bf16_check_point_dir)
    data_args = DataTrainingArguments()
    accelerator = Accelerator()
    test_dataset = SupervisedDataset(args=data_args, config=model_config, tokenizer=tokenizer, mode='test')
    model = get_ds_model(model_config)
    dataloader = DataLoader(test_dataset)
    dataloader = accelerator.prepare(dataloader)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, help="local rank for distributed inference")
    args = parser.parse_args()

    deepspeed.init_distributed()    
    num_gpus_per_node = get_accelerator().device_count()
    num_nodes = dist.get_world_size() // num_gpus_per_node

    run_generation()