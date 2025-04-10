import torch
from torch import nn, Tensor
from sentence_transformers import SentenceTransformer
from torch.nn.functional import cosine_similarity, relu
from typing import List
import concurrent.futures
from functools import partial


class CustomSentenceTransformer(nn.Module):
    def __init__(
            self,
            pretrained_model_name_or_path: str,
            device: str = "cpu",
            batch_size: int = 32,
            is_multilingual_retrival: bool = False,
            pretrained_model_name_or_path_for_query: str = "",
            max_workers: int = 35,  # Threads for document stream processing
            encoding_workers: int = 2  # Threads for batch encoding
    ):
        """
        Args:
            pretrained_model_name_or_path (str): Pretrained model path for SentenceTransformer.
            device (str): Device for computation (e.g., "cuda", "cpu").
            batch_size (int): Batch size for encoding.
            is_multilingual_retrival (bool): Whether to use separate models for query and documents.
            pretrained_model_name_or_path_for_query (str): Pretrained model path for query encoding.
            max_workers (int): Maximum threads for processing document streams.
            encoding_workers (int): Maximum threads for batch encoding.
        """
        super(CustomSentenceTransformer, self).__init__()

        self.document_sentence_transformer = SentenceTransformer(
            pretrained_model_name_or_path, device=device)
        self.query_sentence_transformer = self.document_sentence_transformer \
            if not is_multilingual_retrival \
            else SentenceTransformer(pretrained_model_name_or_path_for_query, device=device)
        
        self.device = device
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.encoding_workers = encoding_workers

        self.linear_sigmoid_stack = nn.Sequential(
            nn.Linear(in_features=2, out_features=1, bias=True),
            nn.Sigmoid()
        ).to(device=device)

    def _encode_batch(self, sentences: List[str], model=None):
        """
        Encode a batch of sentences.
        
        Args:
            sentences (List[str]): The sentences to encode
            model: The model to use (query or document model)
            
        Returns:
            Tensor: Embeddings for the sentences
        """
        if not sentences:
            return torch.tensor([], device=self.device)
            
        if model is None:
            model = self.document_sentence_transformer
            
        return model.encode(sentences, convert_to_tensor=True, device=self.device)

    def _process_document_batch(self, batch_data):
        """
        Process a single batch from a document.
        
        Args:
            batch_data: Tuple of (batch of sentences, query_embedding)
            
        Returns:
            Tensor: Similarities for this batch
        """
        batch, query_embedding = batch_data
        
        if not batch:
            return torch.tensor([0.0], device=self.device)
            
        # Encode the batch
        sentence_embeddings = self._encode_batch(batch)
        
        if sentence_embeddings.numel() == 0:
            return torch.tensor([0.0], device=self.device)
            
        # Calculate similarities
        similarities = cosine_similarity(
            sentence_embeddings, query_embedding.unsqueeze(0)).squeeze()
        similarities = relu(similarities)
        return similarities

    def _process_document_stream_multithreaded(self, query_embedding: Tensor, sentence_stream: List[str]) -> float:
        """
        Process a document stream with multithreaded batch processing.
        
        Args:
            query_embedding (Tensor): The query embedding
            sentence_stream (List[str]): Sentences in the document
            
        Returns:
            float: Semantic similarity score
        """
        if not sentence_stream:
            return 0.0
            
        # Prepare batches
        batches = []
        for i in range(0, len(sentence_stream), self.batch_size):
            batch = sentence_stream[i:i + self.batch_size]
            batches.append((batch, query_embedding))
        
        # Process batches in parallel
        all_similarities = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.encoding_workers) as executor:
            # Submit all batches for processing
            futures = [executor.submit(self._process_document_batch, batch_data) 
                      for batch_data in batches]
            
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                try:
                    similarities = future.result()
                    if similarities.numel() > 0:
                        if similarities.dim() == 0:  # Single value
                            all_similarities.append(similarities.unsqueeze(0))
                        else:
                            all_similarities.append(similarities)
                except Exception as e:
                    print(f"Error in batch processing: {e}")
        
        # If no valid similarities were found
        if not all_similarities:
            return 0.0
            
        # Combine all similarities
        combined_similarities = torch.cat(all_similarities)
        
        # Calculate final similarity score
        one_minus_similarities = 1 - combined_similarities
        total_semantic_similarity = 1 - torch.prod(one_minus_similarities).item()
        
        return total_semantic_similarity

    def forward(self, preprocessed_query: str, lexical_or_topic_similarities: List[float], sentence_streams: List[List[str]]) -> Tensor:
        """
        Calculate combined similarity scores using multithreading at both document and batch levels.
        
        Args:
            preprocessed_query (str): User query
            lexical_or_topic_similarities (List[float]): Lexical or topic similarity scores
            sentence_streams (List[List[str]]): List of document streams
            
        Returns:
            Tensor: Combined similarity scores
        """
        # Encode query once
        query_embedding = self.query_sentence_transformer.encode(
            preprocessed_query, convert_to_tensor=True, device=self.device)
        
        # Use multithreading to process document streams in parallel
        all_semantic_similarities = []
        
        # Create a partial function with the query_embedding already set
        process_func = partial(self._process_document_stream_multithreaded, query_embedding)
        
        # Process streams in parallel using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_stream = {
                executor.submit(process_func, stream): i 
                for i, stream in enumerate(sentence_streams)
            }
            
            # Collect results in order
            all_semantic_similarities = [None] * len(sentence_streams)
            for future in concurrent.futures.as_completed(future_to_stream):
                stream_index = future_to_stream[future]
                try:
                    similarity = future.result()
                    all_semantic_similarities[stream_index] = similarity
                except Exception as e:
                    print(f"Error processing stream {stream_index}: {e}")
                    all_semantic_similarities[stream_index] = 0.0
        
        # Convert to tensors
        all_semantic_similarities_tensor = torch.tensor(
            all_semantic_similarities, device=self.device)
        lexical_or_topic_similarities_tensor = torch.tensor(
            lexical_or_topic_similarities, device=self.device)
        
        # Combine similarities
        combined_similarities_tensor = torch.stack(
            (all_semantic_similarities_tensor, lexical_or_topic_similarities_tensor), dim=1)
        
        # Apply linear layer and sigmoid
        output = self.linear_sigmoid_stack(combined_similarities_tensor)
        return output.squeeze()