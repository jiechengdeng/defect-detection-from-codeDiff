from config import *

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models"}
    )
    lora_r: int = field(
        default=8, metadata={"help": "the rank of the low-rank approximated matrices"}
    )
    lora_alpha: int = field(
        default=16, metadata={"help": "the scaling factor"}
    )
    lora_dropout: float = field(
        default=0.05, metadata={"help": "the dropout rate in low-rank matrices"}
    )
    lora_target_modules: List[str] = field(
        default=None, metadata={"help": "the target modules to apply low-rank matrices, have to check the selected model's target modules' name first"}
    )


@dataclass
class DataTrainingArguments:
    train_data: str = field(
        default='processed_train.jsonl', metadata = {"help": "The path to the input training data file (a jsonl file)."}
    )

    test_data: str = field(
        default='processed_test.jsonl', metadata = {"help": "The path to the input test data file (a jsonl file)."}
    )

    data_dir: str = field(
        default=codeXglue_dataset_dir, metadata = {"help": "The path to the input data directory."}
    )

    max_seq_len: int = field(
        default=2048
    )


"""
Some useful arguments for TrainingArguments:

 - group_by_length (`bool`, *optional*, defaults to `False`):
            Whether or not to group together samples of roughly the same length in the training dataset (to minimize
   this will activate trainer.py - _get_train_sampler - LengthGroupedSampler / DistributedLengthGroupedSampler

 - report_to (`str` or `List[str]`, *optional*, defaults to `"all"`):
            The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,
            `"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. Use `"all"` to report to
            all integrations installed, `"none"` for no integrations.
    disable it otherwise it ask user to select an option when running trainer.train()


"""


@dataclass
class SelfTrainingArguments(TrainingArguments):
    output_dir: str = field(
        default=project_output_dir, metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )

    per_device_train_batch_size: int = field(
        default=1, metadata={"help": "Batch size per GPU/TPU core for training."}
    )

    per_device_eval_batch_size: int = field(
        default=1, metadata={"help": "Batch size per GPU/TPU core for evaluation."}
    )

    gradient_checkpointing: bool = field(
        default=True,
        metadata={
            "help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass."
        },
    )

    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )

    #do_eval: bool = field(default=True, metadata={"help": "Whether to run eval on the dev set."})

    num_train_epochs: float = field(default=5.0, metadata={"help": "Total number of training epochs to perform."})
    
    learning_rate: float = field(default=2e-5, metadata={"help": "The initial learning rate for AdamW."})

    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})

    warmup_ratio: float = field(
        default=0.01, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    )
    save_strategy: Union[IntervalStrategy, str] = field(
        default="epoch",
        metadata={"help": "The checkpoint save strategy to use."},
    )

    save_total_limit: Optional[int] = field(
        default=10,
        metadata={
            "help": (
                "If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in"
                " `output_dir`. When `load_best_model_at_end` is enabled, the 'best' checkpoint according to"
                " `metric_for_best_model` will always be retained in addition to the most recent ones. For example,"
                " for `save_total_limit=5` and `load_best_model_at_end=True`, the four last checkpoints will always be"
                " retained alongside the best model. When `save_total_limit=1` and `load_best_model_at_end=True`,"
                " it is possible that two checkpoints are saved: the last one and the best one (if they are different)."
                " Default is unlimited checkpoints"
            )
        },
    )

    max_steps: int = field(
        default=-1,
        metadata={"help": "Default is -1 will set the max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)\
                  Else set num_train_epochs = args.max_steps // num_update_steps_per_epoch which means you tell huggingface the total steps \
                  in the whole training process."},
    )
    do_eval: bool = field(default=True, metadata={"help": "Whether to run eval on the dev set."})

    evaluation_strategy: Union[IntervalStrategy, str] = field(
        default="epoch",
        metadata={"help": "The evaluation strategy to use."},
    )

    lr_scheduler_type: str = field(
        default="cosine",
        metadata={"help": "The scheduler type to use."},
    )

    optim: str = field(
        default="adamw_torch",
        metadata={"help": "The optimizer to use: adamw_hf, adamw_torch, adamw_torch_fused, adamw_apex_fused, adamw_anyprecision or adafactor"},
    )

    seed: int = field(default=1337, metadata={"help": "Random seed that will be set at the beginning of training."})

    bf16: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA"
                " architecture or using CPU (use_cpu) or Ascend NPU. This is an experimental API and it may change."
            )
        },
    )

    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use fp16 (mixed) precision instead of 32-bit"},
    )

    logging_steps: float = field(
        default=1,
        metadata={
            "help": (
                "Log every X updates steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )

    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Enable deepspeed and pass the path to deepspeed json config file (e.g. `ds_config.json`) or an already"
                " loaded json file as a dict"
            )
        },
    )

    report_to: Union[None, str, List[str]] = field(
        default="tensorboard", metadata={"help": "The list of integrations to report the results and logs to."}
    )

    log_level: Optional[str] = field(
        default="info",
    )
