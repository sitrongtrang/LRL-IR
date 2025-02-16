import torch
from torch import Tensor
import numpy as np


def pos_neg_samples_gen_first_round(
    document_id_answer: str,
    id_relevant_score_pairs: list[tuple[str, float]],
    negative_samples_limit: int = 35
):
    negative_samples_limit = max(0, negative_samples_limit)
    samples: list[tuple[str, float]] = []
    negative_samples_count: int = 0
    for id, relevant_score in id_relevant_score_pairs:
        if id == document_id_answer:
            samples.append((id, relevant_score))
        elif negative_samples_count < negative_samples_limit:
            samples.append((id, relevant_score))
            negative_samples_count += 1
    return samples


def pos_neg_samples_gen_later_round(
    previous_round_output: Tensor,
    previous_round_label_list: list[float],
    previous_round_similarity_score_list: list[float],
    previous_round_doc_chunk_list: list[list[str]],
    negative_samples_limit: int = 0
):
    # Separate streams by label
    pos_label_indices: list[int] = [i for i, label in enumerate(
        previous_round_label_list) if label == 1.0]
    neg_label_indices: list[int] = [i for i, label in enumerate(
        previous_round_label_list) if label == 0.0]

    # Get all label-1 streams and their indices
    pos_doc_chunk_list: list[list[str]] = [
        previous_round_doc_chunk_list[i] for i in pos_label_indices]
    pos_label_list: list[float] = [previous_round_label_list[i]
                                   for i in pos_label_indices]
    pos_similarity_score_list: list[float] = [
        previous_round_similarity_score_list[i] for i in pos_label_indices]

    # Get top-k label-0 streams
    neg_output: Tensor = previous_round_output[neg_label_indices]
    topk_values, topk_indices = torch.topk(neg_output, negative_samples_limit)

    # Map topk_indices back to original indices
    topk_original_indices = [neg_label_indices[i] for i in topk_indices]

    # Get top-k streams and labels for label-0
    neg_doc_chunk_list = [previous_round_doc_chunk_list[i]
                          for i in topk_original_indices]
    neg_label_list = [previous_round_label_list[i]
                      for i in topk_original_indices]
    neg_similarity_score_list: list[float] = [
        previous_round_similarity_score_list[i] for i in topk_original_indices]

    # Combine label-1 and top-k label-0 streams
    doc_chunk_list: list[list[str]] = pos_doc_chunk_list + neg_doc_chunk_list
    label_list: list[float] = pos_label_list + neg_label_list
    similarity_score_list = pos_similarity_score_list + neg_similarity_score_list

    return doc_chunk_list, label_list, similarity_score_list

def min_max_scale(
    doc_list: list[tuple[str, float]], 
    a: float = 0.1, 
    b: float = 1.0
) -> list[tuple[str, float]]:
    scores = [score for _, score in doc_list]
    min_score = min(scores)
    max_score = max(scores)
    scaled_doc_list = [
        (id, a + (score - min_score) * (b - a) / (max_score - min_score) if max_score != min_score else a)
        for id, score in doc_list
    ]
    return scaled_doc_list


def combine_doc_list(
    doc_list_original_query: list[tuple[str, float]],
    doc_list_extended_query: list[tuple[str, float]]
):
    scaled_original_query = min_max_scale(doc_list_original_query)
    scaled_extended_query = min_max_scale(doc_list_extended_query)

    combined_doc_dict: dict[str, float] = {}
    for id, relevant_score in scaled_original_query:
        combined_doc_dict[id] = relevant_score
    for id, relevant_score in scaled_extended_query:
        if id not in combined_doc_dict or combined_doc_dict[id] < relevant_score:
            combined_doc_dict[id] = relevant_score
    combined_doc_list: list[tuple[str, float]] = [
        (id, relevant_score) for id, relevant_score in combined_doc_dict.items()]
    return combined_doc_list


def term_frequency(term, doc):
    return doc.count(term)


def inverse_doc_frequency(term, doc_list):
    N = len(doc_list)
    df = sum([1 for doc in doc_list if term in doc])
    return np.log((N - df + 0.5) / (df + 0.5) + 1)


def bm25(query, doc, doc_list, avgdl, k1=1.5, b=0.75, delta=1.0):
    score = 0
    doc_len = len(doc)

    for term in query:
        tf = term_frequency(term, doc)
        idf = inverse_doc_frequency(term, doc_list)

        numerator = (tf + delta) * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * (doc_len / avgdl))

        score += idf * (numerator / denominator)

    return score

def compute_cosine_cost_matrix(source_emb: Tensor, target_emb: Tensor) -> Tensor:
    source_emb_norm = torch.nn.functional.normalize(source_emb, p=2, dim=1)
    target_emb_norm = torch.nn.functional.normalize(target_emb, p=2, dim=1)
    
    cosine_sim = torch.mm(source_emb_norm, target_emb_norm.t())
    
    cosine_dist = 1 - cosine_sim
    
    return cosine_dist

def pad_sentences(source: list[str], target: list[str]) -> tuple[list[str], list[str], dict]:
    """
    Pad the shorter sentence with mask tokens to match the length of the longer sentence.
    
    Args:
        source (list[str]): List of tokens for source sentence
        target (list[str]): List of tokens for target sentence
        
    Returns:
        Tuple of (padded source tokens, padded target tokens, attention mask)
    """
    max_len = max(len(source), len(target))
    
    padded_source = source + ['[MASK]'] * (max_len - len(source))
    padded_target = target + ['[MASK]'] * (max_len - len(target))
    
    source_attention_mask = [1] * len(source) + [0] * (max_len - len(source))
    target_attention_mask = [1] * len(target) + [0] * (max_len - len(target))
    
    features = {
        'source_attention_mask': torch.tensor([source_attention_mask]),
        'target_attention_mask': torch.tensor([target_attention_mask])
    }
    
    return padded_source, padded_target, features