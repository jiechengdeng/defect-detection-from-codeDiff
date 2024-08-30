import sys
import os

#PROJECT_NAME = "hy-tmp"
PROJECT_NAME = "model_evaluation_on_global_defects"
if not os.getcwd().endswith(PROJECT_NAME):
    os.chdir(os.path.join(os.getcwd()[:os.getcwd().find(PROJECT_NAME)], PROJECT_NAME))
sys.path.append(os.getcwd())

import datasets
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from transformers.trainer import *

from transformers import (
    TrainingArguments, 
    DataCollatorWithPadding, 
    PreTrainedTokenizer,
    HfArgumentParser,
    set_seed,
    AutoTokenizer,
    LlamaForCausalLM,
    AutoModel,
    AutoModelForCausalLM,
    AutoConfig,
    LlamaConfig,
    GenerationConfig,

)
from transformers.modeling_utils import PreTrainedModel
from torch.utils.data import Dataset
import torch 
from torch import nn

IGNORE_INDEX = -100 
EOT_TOKEN = "<|EOT|>"

root_dir = os.path.join(os.getcwd(),'defect_prediction')

project_output_dir = os.path.join(root_dir, "project_output")

model_output_dir = os.path.join(project_output_dir, "model_outputs")

check_point_dir = os.path.join(project_output_dir, "model_weights")

bf16_check_point_dir = os.path.join(project_output_dir, "checkpoint-18675")

pytrace_data_dir = os.path.join(os.getcwd(), 'all_datasets', 'processed_pytracebugs')

codeXglue_dataset_dir = os.path.join(root_dir, 'dataset')

dataset_dir = os.path.join(root_dir, 'dataset')

weight_path = "models--deepseek-ai--deepseek-coder-6.7b-base//snapshots//ce2207a8bfef3ee92bd7dd4cc31c52cfa0046912"

cached_model_path = os.path.join(root_dir,'weights',weight_path)


