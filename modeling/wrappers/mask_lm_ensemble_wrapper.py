"""
HuggingFace Model Wrapper
--------------------------
"""

import torch
import transformers

import textattack

from textattack.models.wrappers.pytorch_model_wrapper import PyTorchModelWrapper


class MaskLMEnsembleModelWrapper(PyTorchModelWrapper):
    """Loads a HuggingFace ``transformers`` model and tokenizer."""

    def __init__(self, model, twokenizer, batch_size=32):
        self.model = model.to(textattack.shared.utils.device)
        if isinstance(tokenizer, transformers.PreTrainedTokenizer):
            tokenizer = textattack.models.tokenizers.AutoTokenizer(tokenizer=tokenizer)
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def _model_predict(self, inputs):
        """Turn a list of dicts into a dict of lists.

        Then make lists (values of dict) into tensors.
        """
        outputs = self.model(inputs)

        if isinstance(outputs[0], str):
            # HuggingFace sequence-to-sequence models return a list of
            # string predictions as output. In this case, return the full
            # list of outputs.
            return outputs
        else:
            # HuggingFace classification models return a tuple as output
            # where the first item in the tuple corresponds to the list of
            # scores for each input.
            return outputs[0]

    def __call__(self, text_input_list):
        """Passes inputs to HuggingFace models as keyword arguments.

        (Regular PyTorch ``nn.Module`` models typically take inputs as
        positional arguments.)
        """
        # ids = self.encode(text_input_list)

        with torch.no_grad():
            outputs = textattack.shared.utils.batch_model_predict(
                self._model_predict, text_input_list, batch_size=self.batch_size
            )

        return outputs


    def _tokenize(self, inputs):
        """Helper method that for `tokenize`
        Args:
            inputs (list[str]): list of input strings
        Returns:
            tokens (list[list[str]]): List of list of tokens as strings
        """
        return [
            self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(x)["input_ids"])
            for x in inputs
        ]
