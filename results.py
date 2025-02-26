import heapq
import os
from dotenv import load_dotenv
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch

from components.ot_solver import OTSolver
from models.knowledge_distillation import KnowledgeDistillation
from utils.utils import compute_cosine_cost_matrix, l1_normalize, pad_sentences, uniform_dist

load_dotenv()


teacher_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "cpu", token=os.getenv("HUGGINGFACE_TOKEN"))
student_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "cpu", token=os.getenv("HUGGINGFACE_TOKEN"))

source_sentence = "These headless bodies can live for about a day, but they don't do much."
target_sentence = "Cơ thể không có đầu kia có thể sống được trong một ngày, nhưng chúng không thể hoạt động được nhiều."

source_tokens = source_sentence.split(' ')
target_tokens = target_sentence.split(' ')

source_tokens, target_tokens = pad_sentences(source_tokens, target_tokens, teacher_model.tokenizer.pad_token)
    
source_dist = uniform_dist(source_tokens, 'cpu')
target_dist = uniform_dist(target_tokens, 'cpu')

source_dist = l1_normalize(source_dist)
target_dist = l1_normalize(target_dist)  

source_ids = teacher_model.tokenizer.convert_tokens_to_ids(source_tokens)
source_ids = torch.tensor([source_ids], device='cpu')
attention_mask = [1 if token != teacher_model.tokenizer.pad_token else 0 for token in source_tokens]
attention_mask = torch.tensor([attention_mask], device='cpu')
source_encoded = {
    'input_ids': source_ids,
    'attention_mask': attention_mask
}

target_ids = student_model.tokenizer.convert_tokens_to_ids(target_tokens)
target_ids = torch.tensor([target_ids], device='cpu')
attention_mask = [1 if token != student_model.tokenizer.pad_token else 0 for token in target_tokens]
attention_mask = torch.tensor([attention_mask], device='cpu')
target_encoded = {
    'input_ids': target_ids,
    'attention_mask': attention_mask
}

with torch.no_grad():
    source_embeddings = teacher_model.forward(source_encoded)['token_embeddings'].squeeze(0)

target_embeddings = student_model.forward(target_encoded)['token_embeddings'].squeeze(0)

ot_solver = OTSolver('cpu')
cost = compute_cosine_cost_matrix(source_embeddings, target_embeddings)
plan, loss = ot_solver(source_dist, target_dist, cost)

for i, token in enumerate(source_tokens):
    temp = plan[i].tolist()
    largest_values = heapq.nlargest(3, temp)
    mapped_indices = [temp.index(value) for value in largest_values]
    mapped_tokens = [target_tokens[j] for j in mapped_indices]
    print(token, mapped_tokens)
