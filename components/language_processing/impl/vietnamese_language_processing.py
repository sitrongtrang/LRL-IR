from functools import reduce
import os
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import Callable
from ..language_processing import LanguageProcessing
import py_vncorenlp


class VietnameseLanguageProcessing(LanguageProcessing):
    def __init__(
            self, 
            pre_trained_tokenizer_model: PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
            tokenizer: Callable[[str], list[str]] | None = None,
            encoder: Callable[[str | list[str]], list[int]] | None = None
    ):
        self._pre_trained_tokenizer_model = AutoTokenizer.from_pretrained("vinai/phobert-base-v2") if \
            pre_trained_tokenizer_model is None else pre_trained_tokenizer_model
        self._text_preprocessing = self._load_text_preprocessing()
        self._tokenizer = self._pre_trained_tokenizer_model.tokenize if tokenizer is None else tokenizer
        self._encoder = self._pre_trained_tokenizer_model.encode if encoder is None else encoder
    
    def _load_text_preprocessing(self):
        if os.path.isdir("/vncorenlp/models") == False or os.path.exists('/vncorenlp/VnCoreNLP-1.2.jar') == False:
            py_vncorenlp.download_model("/vncorenlp")
        return py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir="/vncorenlp").word_segment
    
    def text_preprocessing(self, text):
        return self._text_preprocessing(text)
        
    def pre_trained_tokenizer_model(self):
        return self._pre_trained_tokenizer_model
    
    def tokenizer(self, text: str):
        return self._tokenizer(text)
    
    def encoder(self, text: str | list[str]):
        return self._encoder(text)
    
    
        
    
        
        