from config import *
from my_arguments import ModelArguments, DataTrainingArguments, SelfTrainingArguments
from dataset_class import SupervisedDataset, DataCollatorForSupervisedDataset
from model import MyModel
from trainer import MyTrainer
from huggingface_hub import login
from transformers.models.llama import LlamaForCausalLM
import numpy as np
import accelerate

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=cached_model_path,use_fast=True,padding_side='left')
model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path=cached_model_path)
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=cached_model_path,torch_dtype=torch.bfloat16)

def safe_save_model_for_hf_trainer(trainer, output_dir):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def my_compute_metric(eval_pred):
    Acc=[]
    for p, a in zip(eval_pred['predictions'], eval_pred['labels']):
        Acc.append(p==a)
    
    return {
        'Acc': np.mean(Acc),
    }

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, SelfTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    model.config.use_cache = False
    model_config._attn_implementation = 'flash_attention_2'
    model_config.pad_token_id = 32021

    
    if training_args.local_rank > 0: 
        torch.distributed.barrier()

    train_dataset = SupervisedDataset(args=data_args, config=model_config, tokenizer=tokenizer, mode='train')
    test_dataset = SupervisedDataset(args=data_args, config=model_config, tokenizer=tokenizer, mode='test')
    dataCollator = DataCollatorForSupervisedDataset(model_config, tokenizer)

    if training_args.local_rank == 0:
        torch.distributed.barrier()

    trainer = MyTrainer(
        model=model,
        args=training_args,
        data_collator=dataCollator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=my_compute_metric,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(bf16_check_point_dir)
    trainer.tokenizer.save_pretrained(bf16_check_point_dir)
    trainer.save_state()

if __name__ == "__main__":
    main()

 




