from config import *
from torch.nn.utils.rnn import pad_sequence
from my_arguments import DataTrainingArguments
import math
import copy


parital_prompt = "\nignore the following lines of codes...\nIf the code snippet is vulnerable, answer Yes else answer No.\n### Response:\n"

class SupervisedDataset(Dataset):
    
    def __init__(self, args: DataTrainingArguments, config, tokenizer: PreTrainedTokenizer, mode: str):
        self.args = args
        self.config = config
        self.mode = mode

        if mode == 'train':
            self.data = datasets.load_dataset(path=args.data_dir,data_files=args.train_data,split='train')
        else:
            self.data = datasets.load_dataset(path=args.data_dir,data_files=args.test_data,split='train')
        print(f'Loaded {self.data}')
        self.tokenizer = tokenizer
    
        self._tokenize()

    def _tokenize(self):
        tokenized_data = []
        if self.mode == 'train':
            for i in range(len(self.data)):
                source = self.data[i]['source'].lstrip()
                output = self.data[i]['target'] + '\n' + EOT_TOKEN
                example = source + output

                example_tokenized = self.tokenizer(
                    example,
                    return_tensors="pt",
                )
                
                source_tokenized = self.tokenizer(
                    source,
                    return_tensors="pt"
                )
                
                example_len = example_tokenized['input_ids'].shape[-1]
                source_len = source_tokenized['input_ids'].shape[-1]

                if example_len > self.args.max_seq_len:
                    continue
    

                input_id = example_tokenized['input_ids']
                label = copy.deepcopy(input_id)
                label[:,:source_len] = IGNORE_INDEX
                tokenized_data.append([input_id,label])

        else:
            for i in range(len(self.data)):
               
                source = self.data[i]['source'].lstrip()
                output = self.data[i]['target']
                
                source_tokenized = self.tokenizer(
                    source,
                    return_tensors="pt"
                )

                ques_tokenized = self.tokenizer(
                    parital_prompt,
                    return_tensors="pt"
                )

                # output_tokenized = self.tokenizer(
                #     output,
                #     return_tensors="pt"
                # )
                
                source_len = source_tokenized['input_ids'].shape[-1]
  
                if source_len > self.args.max_seq_len:
                    source = self.data[i]['source'].lstrip().split('If the code snippet is vulnerable, answer Yes else answer No')[0]
                    source_tokenized = self.tokenizer(
                        source,
                        truncation=True,
                        max_length=self.args.max_seq_len - ques_tokenized['input_ids'].shape[-1],
                        return_tensors="pt"
                    )

                    truncated_source = self.tokenizer.decode(source_tokenized['input_ids'][0],skip_special_tokens=True)
                    truncated_source += parital_prompt
                    

                    source_tokenized = self.tokenizer(
                        truncated_source,
                        return_tensors="pt"
                    )
       
                else:
                    source = self.data[i]['source'].lstrip()
                    source_tokenized = self.tokenizer(
                        source,
                        return_tensors="pt"
                    )
                

                input_id = source_tokenized['input_ids']
                tokenized_data.append([input_id,output])
          
        print(f'number of samples tokenized: {len(tokenized_data)}/{len(self.data)}')
        self.data = tokenized_data
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
        

@dataclass
class DataCollatorForSupervisedDataset(DataCollatorWithPadding):
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

    def __call__(self, features: List):
        d = {
            'input_ids': [],
            'labels':[],
        }

        for input_id, label in features:
            d['input_ids'].append(input_id.squeeze(0))
            if type(label) == str:
                d['labels'].append(label)
            else:
                d['labels'].append(label.squeeze(0))


        pad_token_id = None
        if self.config.architectures != 'LlamaForCausalLM':
            pad_token_id = self.tokenizer.pad_token_id
        else:
            pad_token_id = 32021

        d['input_ids'] = pad_sequence(d['input_ids'], batch_first=True, padding_value=pad_token_id)
        if type(d['labels'][0]) != str:
            d['labels'] = pad_sequence(d['labels'], batch_first=True, padding_value=pad_token_id)

        return d
