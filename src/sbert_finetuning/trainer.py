from transformers import AutoTokenizer, AutoModel
import logging
from sbert_finetuning.models import ModelType
from sbert_finetuning.data import Dataset
import torch

from transformers import TrainingArguments, Trainer as HFTrainer
import numpy as np 
# import evaluate


class Trainer:
    def __init__(self, l_rate, batch_size, num_epoch, model_type, device, export_dir, train_data_path, valid_data_path, test_data_path):
        self.logger = logging.getLogger(__class__.__name__)

        self._l_rate = l_rate
        self._batch_size = batch_size
        self._num_epoch = num_epoch
        self._model_type = model_type

        self.model, self.tokenizer = self._init_model(self._model_type)
        self._init_datasets(train_data_path, valid_data_path, test_data_path)
        self.training_args = TrainingArguments(
                                               output_dir=export_dir,
                                               learning_rate=l_rate,
                                               num_train_epochs=num_epoch,
                                               save_strategy="epoch",
                                               evaluation_strategy="epoch",
                                               use_cpu=False,
                                               report_to=None,
                                               per_device_train_batch_size=batch_size,
                                               per_device_eval_batch_size=batch_size,
                                               label_names=[],
                                               )
        

        # self.clf_metrics = evaluate.combine(["accuracy", "f1"])

        
        self.huggingface_trainer = HFTrainer(
                                    model=self.model,
                                    args=self.training_args,
                                    train_dataset=self.train_dataset,
                                    eval_dataset=self.valid_dataset,
                                    compute_metrics=self.compute_metrics,
                                    data_collator=self.data_collator,
                                )
        self.huggingface_trainer.compute_loss = self.compute_loss
        self.loss_fn = torch.nn.TripletMarginLoss()

    def data_collator(self, input):
        new_input = {"anchor": input[0][0],
                     "pos": input[0][1],
                     "neg": input[0][2]}
        return new_input
    
    def _init_model(self, model_type):
        if model_type == ModelType.SbertLargeNluRu:
            model = AutoModel.from_pretrained("ai-forever/sbert_large_nlu_ru")
            tokenizer = AutoTokenizer.from_pretrained("ai-forever/sbert_large_nlu_ru")
        else:
            raise ValueError(f"Model type: {model_type} doesn't supported!")
        
        return model, tokenizer

    def _init_datasets(self, train_data_path, valid_data_path, test_data_path):
        self.train_dataset = Dataset(train_data_path, self.tokenizer)
        self.valid_dataset = Dataset(valid_data_path, self.tokenizer)
        self.test_dataset = Dataset(test_data_path, self.tokenizer)
        self.logger.info(f"{len(self.train_dataset)=}")
        self.logger.info(f"{len(self.valid_dataset)=}")
        self.logger.info(f"{len(self.test_dataset)=}")

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def compute_loss(self, model, inputs, return_outputs=False):
        anchor, pos, neg = inputs["anchor"], inputs["pos"], inputs["neg"]
        # anchor_labels = anchor.pop("labels")
        # pos_labels = pos.pop("labels")
        # neg_labels = neg.pop("labels")

        anchor_outputs = self.mean_pooling(model(**anchor), anchor['attention_mask']) 
        pos_outputs = self.mean_pooling(model(**pos), anchor['attention_mask']) 
        neg_outputs = self.mean_pooling(model(**neg), anchor['attention_mask'])

        loss = self.loss_fn(anchor_outputs, pos_outputs, neg_outputs)

        return (loss, anchor_outputs) if return_outputs else loss

        
    def compute_metrics(self, eval_pred):
        return {"Mock": 1.0}
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return self.clf_metrics.compute(predictions=predictions, references=labels)

    
    def train(self):
        self.huggingface_trainer.train()
