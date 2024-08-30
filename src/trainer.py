from config import *
import matplotlib.pyplot as plt
from torch.nn.functional import softmax

class MyTrainer(Trainer):
    """
    override:
           def compute_loss(self, model, inputs, return_outputs=False) -> Union[float, Tuple[float, Dict[str, torch.Tensor]]]
    
    """
    # def compute_loss(self, model, inputs, return_outputs=False):
    #     outputs = model(**inputs)
    #     if self.args.local_rank == 0:
    #         logits = outputs['logits']  # Shape: (1, sequence_length, vocab_size)

    #         #logits = logits[:,logits.size(1) - 5:,:]
        
    #         # Step 1: Convert logits to probabilities
    #         probabilities = softmax(logits, dim=-1)
            
    #         # Step 2: Get the predicted token indices
    #         predicted_indices = torch.argmax(probabilities, dim=-1)
            
    #         # Step 3: Convert indices to tokens
    #         predicted_tokens = self.tokenizer.convert_ids_to_tokens(predicted_indices[0])

    #         print(''.join(predicted_tokens))
    #     return (outputs['loss'], outputs) if return_outputs else outputs['loss']
        
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args
    
        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Will be useful when we have an iterable dataset so don't know its length.
        observed_num_examples = 0
        all_preds = []
        all_labels = []
        all_inputs = []
        stop_id = self.tokenizer.convert_tokens_to_ids(EOT_TOKEN)

        # Main evaluation loop
        with torch.no_grad():
            for step, inputs in enumerate(dataloader):
                # Update the observed num examples
                observed_batch_size = find_batch_size(inputs)
                if observed_batch_size is not None:
                    observed_num_examples += observed_batch_size
                    # For batch samplers, batch_size is not known by the dataloader in advance.
                    if batch_size is None:
                        batch_size = observed_batch_size

                input_id = inputs['input_ids']
                label = inputs['labels']
                
                input_ids = self.accelerator.pad_across_processes(input_id, dim=1, pad_index=stop_id,pad_first=True)
                input_ids = self.accelerator.gather_for_metrics((input_ids))

                if self.args.local_rank == 0:
                    print(f'{input_ids.size()}')

                for input_id in input_ids:
                    if self.args.local_rank == 0:
                        print(f'{input_id.size()}')
                    input_id = input_id[input_id != stop_id]
                    output = model.generate(
                                input_id.unsqueeze(0),
                                max_new_tokens=3,
                                pad_token_id=stop_id,
                                eos_token_id=stop_id,
                            )
                    if self.args.local_rank == 0:
                        print(f'{input_id.unsqueeze(0).size()=}')
                        print(f'{output.size()=}')

                    output = output[0][input_id.size(-1):]

                    if self.args.local_rank == 0:
                        print(f'after {output.size()=}')

                    output = self.tokenizer.decode(output,skip_special_tokens=True)
                    if self.args.local_rank == 0:
                        print(f'{output=}')
                    pred_num = -1
                    if 'Yes' in output:
                        pred_num = 1
                    elif 'No' in output:
                        pred_num = 0

                    all_preds.append(pred_num)
                    
                labels = self.accelerator.gather_for_metrics((label))
                labels = [1 if 'Yes' in l else 0 for l in labels]
                all_labels.extend(labels)
      

                if self.args.local_rank == 0:
                    print(f'{len(all_preds)=}\t{len(all_labels)=}')
    

            # stop_id = self.tokenizer.convert_tokens_to_ids(EOT_TOKEN)
            # outputs = model.generate(
            #     input_id,
            #     max_new_tokens=3,
            #     pad_token_id=stop_id,
            #     eos_token_id=stop_id
            # )

            # output = outputs[0][input_id.size(-1):]
            # decode_str = self.tokenizer.decode(output,skip_special_tokens=True)
            # decode_label = self.tokenizer.decode(label[0],skip_special_tokens=True)
            # answer = decode_str.split('\n')[0]
            # pred_num = -1
            
            # if 'Yes' in answer:
            #     pred_num = 1
            # elif 'No' in answer:
            #     pred_num = 0

            
            # all_preds.append(pred_num)
            # all_labels.append(1 if 'Yes' in answer else 0)

            # if self.args.local_rank == 0:
            #     print(f'{len(all_preds)=}\t{decode_str=}\t{decode_label=}')
            
        metrics = self.compute_metrics({'predictions': all_preds,'labels':all_labels})
        if self.args.local_rank == 0:
            with open(os.path.join(model_output_dir,'eval_result.txt'), 'a+') as f:
                f.write(f'Epoch: {self.state.epoch}\tGlobal_Step: {self.state.global_step}\n{metrics}\n')
            f.close()

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples) 


    def get_training_curve(self):
        y_train_loss = []
        #y_grad_norm = []
        for d in self.state.log_history[:-1]:
            y_train_loss.append(d['loss'])
            #y_grad_norm.append(d['grad_norm'])

        x_total_steps = range(len(y_train_loss))

        self._compute_curve(x_total_steps,y_train_loss,'steps','loss','training_loss')
        #self._compute_curve(x_total_steps,y_grad_norm,'steps','grad_norm','Grad_Norm')
    
    def _compute_curve(self,x,y,xLabel,yLabel,title):
        plt.figure()
        ax = plt.axes()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.xlabel(xLabel)
        plt.ylabel(yLabel)

        plt.plot(x,y, linewidth=1, linestyle="solid",label="train_loss")
        plt.legend()
        plt.title(title)
        plt.savefig(os.path.join(log_dir,f"{title}.png"), format='png', dpi=300)
    
        
