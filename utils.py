import torch  
import numpy as np 

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



