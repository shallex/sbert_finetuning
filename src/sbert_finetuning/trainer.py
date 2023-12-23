from transformers import AutoTokenizer, AutoModel
import logging
from sbert_finetuning.models import ModelType
from sbert_finetuning.data import Dataset
from torch.utils.data import DataLoader
from sentence_transformers import losses
from sbert_finetuning.loss import LossType


class Trainer:
    def __init__(self, l_rate, batch_size, loss_type, num_epoch, model_type, device, export_dir, train_data_path, valid_data_path, test_data_path):
        self.logger = logging.getLogger(__class__.__name__)

        self._l_rate = l_rate
        self._batch_size = batch_size
        self._loss_type = loss_type
        self._num_epoch = num_epoch
        self._model_type = model_type

        self.model, self.tokenizer = self._init_model(self._model_type)
        self._init_dataloaders(train_data_path, valid_data_path, test_data_path)
        self._init_loss(self._loss_type)
    
    def _init_model(self, model_type):
        if model_type == ModelType.SbertLargeNluRu:
            model = AutoModel.from_pretrained("ai-forever/sbert_large_nlu_ru")
            tokenizer = AutoTokenizer.from_pretrained("ai-forever/sbert_large_nlu_ru")
        else:
            raise ValueError(f"Model type: {model_type} doesn't supported!")
        
        return model, tokenizer

    def _init_dataloaders(self, train_data_path, valid_data_path, test_data_path):
        train_dataset = Dataset(train_data_path)
        valid_dataset = Dataset(valid_data_path)
        test_dataset = Dataset(test_data_path)
        self.logger.info(f"{len(train_dataset)=}")
        self.logger.info(f"{len(valid_dataset)=}")
        self.logger.info(f"{len(test_dataset)=}")

        self.train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=self._batch_size)
        self.valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=1)
        self.test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1)

    def _init_loss(self, loss_type):
        if loss_type == LossType.BatchAllTripletLoss:
            self._loss_fn = losses.BatchAllTripletLoss(model=self.model)
        elif loss_type == LossType.BatchHardSoftMarginTripletLoss:
            self._loss_fn = losses.BatchHardSoftMarginTripletLoss(model=self.model)
        elif loss_type == LossType.BatchHardTripletLoss:
            self._loss_fn = losses.BatchHardTripletLoss(model=self.model)
        elif loss_type == LossType.BatchSemiHardTripletLoss:
            self._loss_fn = losses.BatchSemiHardTripletLoss(model=self.model)
        else:
            raise ValueError(f"Loss type: {loss_type} doesn't supported!")
        
    def fit(self):
        self.model.fit(train_objectives=[(self.train_dataloader, self._loss_fn)], epochs=self._num_epoch) 




    
