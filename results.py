import os
from dotenv import load_dotenv
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

load_dotenv()


teacher_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "cpu", token=os.getenv("HUGGINGFACE_TOKEN"))
student_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "cpu", token=os.getenv("HUGGINGFACE_TOKEN"))

sentence1 = "The knight defended the castle from the enemy"
sentence2 = "El caballero defendio el castillo del enemigo"

# Tokenize the sentence (naive tokenization, you can use a tokenizer from Hugging Face)
tokens1 = teacher_model.tokenizer.tokenize(sentence1)
tokens2 = student_model.tokenizer.tokenize(sentence2)

# Compute embeddings
sentence_embedding1 = teacher_model.encode(sentence1, convert_to_numpy=True, output_value='sentence_embedding')
sentence_embedding2 = student_model.encode(sentence2, convert_to_numpy=True, output_value='sentence_embedding')

token_embeddings1 = teacher_model.encode(sentence1, convert_to_numpy=True, output_value='token_embeddings')
token_embeddings2 = teacher_model.encode(sentence1, convert_to_numpy=True, output_value='token_embeddings')

# Reduce dimensionality
dim_reducer_sentence = PCA(n_components=2)
sentence_embeddings_2d = dim_reducer_sentence.fit_transform(
    np.vstack([sentence_embedding1, sentence_embedding2])
)

dim_reducer_tokens = TSNE(n_components=2, perplexity=5, random_state=42)
token_embeddings1_2d = dim_reducer_tokens.fit_transform(token_embeddings1)
token_embeddings2_2d = dim_reducer_tokens.fit_transform(token_embeddings2)

# ---- PLOT 1: Sentence-Level Embeddings ----
plt.figure(figsize=(6, 6))
plt.scatter(sentence_embeddings_2d[0, 0], sentence_embeddings_2d[0, 1], color="blue", label="Model 1")
plt.scatter(sentence_embeddings_2d[1, 0], sentence_embeddings_2d[1, 1], color="red", label="Model 2")
plt.text(sentence_embeddings_2d[0, 0], sentence_embeddings_2d[0, 1], "Sentence", fontsize=10, color="blue")
plt.text(sentence_embeddings_2d[1, 0], sentence_embeddings_2d[1, 1], "Sentence", fontsize=10, color="red")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.title("Sentence Embedding Comparison")
plt.legend()
plt.savefig("sentence_embeddings.png")
plt.show()

# ---- PLOT 2: Token-Level Embeddings ----
plt.figure(figsize=(8, 6))
for i, token in enumerate(tokens1):
    plt.scatter(token_embeddings1_2d[i, 0], token_embeddings1_2d[i, 1], color="blue", label="Model 1" if i == 0 else "")
    plt.text(token_embeddings1_2d[i, 0], token_embeddings1_2d[i, 1], token, fontsize=9, color="blue")
    
for i, token in enumerate(tokens2):
    plt.scatter(token_embeddings2_2d[i, 0], token_embeddings2_2d[i, 1], color="red", label="Model 2" if i == 0 else "")
    plt.text(token_embeddings2_2d[i, 0], token_embeddings2_2d[i, 1], token, fontsize=9, color="red")

plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.title("Token Embedding Comparison")
plt.legend()
plt.savefig("token_embeddings.png")
plt.show()