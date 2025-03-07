import heapq
import os
from dotenv import load_dotenv
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch

from components.language_processing.impl.vietnamese_language_processing import VietnameseLanguageProcessing
from components.ot_solver import OTSolver
from models.knowledge_distillation import KnowledgeDistillation
from utils.utils import compute_cosine_cost_matrix, l1_normalize, pad_sentences, uniform_dist

load_dotenv()


teacher_model = SentenceTransformer("distiluse-base-multilingual-cased-v2", "cpu", token=os.getenv("HUGGINGFACE_TOKEN"))
student_model = SentenceTransformer("distiluse-base-multilingual-cased-v2", "cpu", token=os.getenv("HUGGINGFACE_TOKEN"))

vi_language_processing = VietnameseLanguageProcessing()

source_sentence = "Today I want to talk to you about ethnic conflict and civil war"
target_sentence = "Hôm nay, tôi muốn nói chuyện với các bạn về vấn đề xung đột sắc tộc và nội chiến"

# source_sentence = "Thank you so much"
# target_sentence = "Cảm ơn rất nhiều"

source_tokens = source_sentence.split(' ')
target_tokens = vi_language_processing.text_preprocessing(target_sentence)

new_source_tokens = [token for token in source_tokens if token not in teacher_model.tokenizer.get_vocab()]
if new_source_tokens:
    teacher_model.tokenizer.add_tokens(new_source_tokens)
    teacher_model[0].auto_model.resize_token_embeddings(len(teacher_model.tokenizer))

new_target_tokens = [token for token in target_tokens if token not in student_model.tokenizer.get_vocab()]
if new_target_tokens:
    student_model.tokenizer.add_tokens(new_target_tokens)
    student_model[0].auto_model.resize_token_embeddings(len(student_model.tokenizer))


source_tokens, target_tokens = pad_sentences(source_tokens, target_tokens, teacher_model.tokenizer.pad_token, student_model.tokenizer.pad_token)
print(target_tokens)

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

print(source_ids, target_ids)

source_result = teacher_model.forward(source_encoded)
source_token_embeddings = source_result['token_embeddings'].squeeze(0)
source_sentence_embedding = source_result['sentence_embedding']

target_result = student_model.forward(target_encoded)
target_token_embeddings = target_result['token_embeddings'].squeeze(0)
target_sentence_embedding = target_result['sentence_embedding']

ot_solver = OTSolver('cpu')
cost = compute_cosine_cost_matrix(source_token_embeddings, target_token_embeddings)
plan, loss = ot_solver(source_dist, target_dist, cost)

print(source_token_embeddings, target_token_embeddings)
# print("\ncost: ", cost)
# print("\nplan: ", plan)
print("\nloss: ", loss)

for i, token in enumerate(source_tokens):
    temp = plan[i].tolist()
    largest_values = heapq.nlargest(1, temp)
    mapped_indices = [temp.index(value) for value in largest_values]
    mapped_tokens = [target_tokens[j] for j in mapped_indices]
    print(token, mapped_tokens)

dist = compute_cosine_cost_matrix(source_sentence_embedding, target_sentence_embedding)
print(dist)


