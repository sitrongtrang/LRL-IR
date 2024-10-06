import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torch.nn import functional as F
from qrann import QRANN

class SentenceTransformer(nn.Module):
    """
    Args:
        model_path (string): Path to the fine-tuned model
            default model: bert-base-uncased.
    Attributes:
        base (BertModel): Base model
        tokenizer (BertTokenizer): Tokenizer
        qrann (QRANN): QRANN for calculating relevance score
    """

    def __init__(self, model_path='bert-base-uncased'):
        super(SentenceTransformer, self).__init__()
        self.base = BertModel.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.fc = nn.Linear(self.bert.config.hidden_size, 256) 
        self.qrann = QRANN()

    def get_embedding(self, text):
        tokens = self.tokenizer.tokenize(text)
        tokens = ['[CLS]'] + tokens + ['[SEP]']

        attn_mask = [0] * len(tokens)
        sent_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        token_ids = torch.tensor(sent_ids).unsqueeze(0) 
        attn_mask = torch.tensor(attn_mask).unsqueeze(0) 
        
        outputs = self.base(input_ids=token_ids, attention_mask=attn_mask)
        hidden_state = outputs[0]
        cls_token = hidden_state[:, 0, :]  
        embedding = self.fc(cls_token)
        return embedding
    
    def forward(self, query, chunk):

        query_embedding = self.get_embedding(query)
        chunk_embedding = self.get_embedding(chunk)

        relevance_scores = self.qrann(query_embedding, chunk_embedding)

        return relevance_scores
