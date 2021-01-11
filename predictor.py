import torch
import numpy as np
from typing import List, Dict
from torch.nn import Module
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from data.instance import InputInstance
from data.processor import DataProcessor
from utils.utils import collate_fn, xlnet_collate_fn, convert_batch_to_bert_input_dict


class Predictor:
    def __init__(self, model: Module, data_processor: DataProcessor, model_type: str):
        self._model = model
        self._data_processor = data_processor
        self._model_type = model_type

    def _forward(self, batch: Dict[str, torch.Tensor]) -> np.ndarray:
        batch = tuple(t.to(self._model.device) for t in batch)
        inputs = convert_batch_to_bert_input_dict(batch, self._model_type)
        self._model.eval()
        with torch.no_grad():
            logits = self._model(**inputs)[0]
        probs = F.softmax(logits,dim=-1)
        return probs.cpu().numpy()

    def _forward_on_multi_batches(self, dataset: Dataset, batch_size: int = 300) -> np.ndarray:
        data_loader = DataLoader(dataset, shuffle=False, batch_size=batch_size,
                                collate_fn=xlnet_collate_fn if self._model_type in ['xlnet'] else collate_fn)

        all_probs = []
        for batch in data_loader:
            all_probs.append(self._forward(batch))
        all_probs = np.concatenate(all_probs, axis=0)
        return all_probs

    def get_label_idx(self, label: str) -> int:
        return self._data_processor.data_reader.get_label_to_idx(label)

    @torch.no_grad()
    def predict(self, example: InputInstance) -> np.ndarray:
        return self.predict_batch([example])[0]

    @torch.no_grad()
    def predict_batch(self, examples: List[InputInstance]) -> np.ndarray:
        dataset = self._data_processor.convert_instances_to_dataset(examples,use_tqdm=False)
        return self._forward_on_multi_batches(dataset)

    @property
    def pad_token(self) -> str:
        return self._data_processor.tokenizer.pad_token

    @property
    def unk_token(self) -> str:
        return self._data_processor.tokenizer.unk_token

    @property
    def sep_token(self) -> str:
        return self._data_processor.tokenizer.sep_token