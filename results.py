import heapq
import os
from dotenv import load_dotenv
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch

from components.language_processing.impl.english_language_processing import EnglishLanguageProcessing
from components.language_processing.impl.vietnamese_language_processing import VietnameseLanguageProcessing
from components.ot_solver import OTSolver
from models.knowledge_distillation import KnowledgeDistillation
from utils.utils import compute_cosine_cost_matrix, l1_normalize, pad_sentences, uniform_dist

load_dotenv()


teacher_model = SentenceTransformer("distiluse-base-multilingual-cased-v2", "cpu", token=os.getenv("HUGGINGFACE_TOKEN"))
student_model = SentenceTransformer(os.getenv("PROJECT_DIR") + "sentence_transformer_multilingual_padded_uniform", "cpu", token=os.getenv("HUGGINGFACE_TOKEN"))

vi_language_processing = VietnameseLanguageProcessing()
en_language_processing = EnglishLanguageProcessing()

source_sentence = "Dong was a currency of the Republic of Vietnam (South Vietnam) from 1953 to May 2, 1978"
target_sentence = "Đồng đã từng là tiền tệ của Việt Nam Cộng hòa (Nam Việt Nam) từ năm 1953 đến ngày 2 tháng 5 năm 1978"

source_tokens = en_language_processing.text_preprocessing(source_sentence)
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

source_result = teacher_model.forward(source_encoded)
source_token_embeddings = source_result['token_embeddings'].squeeze(0)
source_sentence_embedding = source_result['sentence_embedding']

target_result = student_model.forward(target_encoded)
target_token_embeddings = target_result['token_embeddings'].squeeze(0)
target_sentence_embedding = target_result['sentence_embedding']

ot_solver = OTSolver('cpu')
cost = compute_cosine_cost_matrix(source_token_embeddings, target_token_embeddings)
plan, loss = ot_solver(source_dist, target_dist, cost)

# print("\ncost: ", cost)
# print("\nplan: ", plan)
# print("\nloss: ", loss)

for i, token in enumerate(source_tokens):
    temp = plan[i].tolist()
    largest_values = heapq.nlargest(1, temp)
    mapped_indices = [temp.index(value) for value in largest_values]
    mapped_tokens = [target_tokens[j] for j in mapped_indices]
    # if (token == "Vietnam"): print(source_token_embeddings[i], target_token_embeddings[mapped_indices[0]])
    print(token, mapped_tokens)

dist = compute_cosine_cost_matrix(source_sentence_embedding, target_sentence_embedding)
print(dist)

def visualize_embeddings(en_tokens, vi_tokens, en_embeddings, vi_embeddings, perplexity=30, n_iter=1000, random_state=42):
    """
    Visualize token embeddings from English and Vietnamese using t-SNE.
    
    Parameters:
    -----------
    en_tokens : list
        List of English tokens
    vi_tokens : list
        List of Vietnamese tokens
    en_embeddings : numpy.ndarray
        Embedding vectors for English tokens (shape: [n_tokens, embedding_dim])
    vi_embeddings : numpy.ndarray
        Embedding vectors for Vietnamese tokens (shape: [n_tokens, embedding_dim])
    perplexity : int, optional
        Perplexity parameter for t-SNE (default: 30)
    n_iter : int, optional
        Number of iterations for t-SNE optimization (default: 1000)
    random_state : int, optional
        Random seed for reproducibility (default: 42)
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure containing the visualization
    """
    en_embeddings = en_embeddings.detach().numpy() 
    vi_embeddings = vi_embeddings.detach().numpy()

    # Combine all embeddings for t-SNE
    combined_embeddings = np.vstack([en_embeddings, vi_embeddings])
    
    # Apply t-SNE to reduce dimensions to 2D
    print(f"Running t-SNE on {combined_embeddings.shape[0]} embeddings of dimension {combined_embeddings.shape[1]}...")
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=random_state, metric="cosine")
    reduced_embeddings = tsne.fit_transform(combined_embeddings)
    
    # Split the reduced embeddings back into English and Vietnamese
    n_en = len(en_tokens)
    en_reduced = reduced_embeddings[:n_en]
    vi_reduced = reduced_embeddings[n_en:]
    
    # Create figure for plotting
    plt.figure(figsize=(12, 10))
    
    # Plot English tokens (teacher model)
    plt.scatter(
        en_reduced[:, 0], en_reduced[:, 1], 
        c='blue', alpha=0.7, marker='o', s=100, label='English (Teacher)'
    )
    
    # Plot Vietnamese tokens (student model)
    plt.scatter(
        vi_reduced[:, 0], vi_reduced[:, 1], 
        c='red', alpha=0.7, marker='x', s=100, label='Vietnamese (Student)'
    )
    
    # Add token labels
    for i, token in enumerate(en_tokens):
        plt.annotate(token, (en_reduced[i, 0], en_reduced[i, 1]), 
                     fontsize=9, alpha=0.8, color='blue')
    
    for i, token in enumerate(vi_tokens):
        plt.annotate(token, (vi_reduced[i, 0], vi_reduced[i, 1]), 
                     fontsize=9, alpha=0.8, color='red')
    
    # Add title and legend
    plt.title('t-SNE Visualization of Token Embeddings', fontsize=16)
    plt.legend(fontsize=12)
    
    # Improve layout
    plt.tight_layout()
    
    # Add grid
    plt.grid(linestyle='--', alpha=0.6)
    
    # Return the figure
    plt.show()

# visualize_embeddings(source_tokens, target_tokens, source_token_embeddings, target_token_embeddings)

