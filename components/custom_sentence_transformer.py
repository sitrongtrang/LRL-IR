import torch
from torch import nn, Tensor
from sentence_transformers import SentenceTransformer
from torch.nn.functional import cosine_similarity, relu, sigmoid


class CustomSentenceTransformer(nn.Module):
    def __init__(
            self,
            pretrained_model_name_or_path: str,
            device: str = "cpu",
            batch_size: int = 32
    ):
        super(CustomSentenceTransformer, self).__init__()

        self.sentence_transformer: SentenceTransformer = SentenceTransformer(
            pretrained_model_name_or_path).to(device=device)
        self.device: str = device
        self.batch_size: int = batch_size

        self.linear_sigmoid_stack = nn.Sequential(
            nn.Linear(in_features=2, out_features=1, bias=True),
            nn.Sigmoid()
        ).to(device=device)

    def _process_sentence_stream(self, query: str, sentence_stream: list[str]) -> float:
        total_semantic_similarity: float = 1.0

        query_embedding: Tensor = self.sentence_transformer.encode(
            query, convert_to_tensor=True, device=self.device)

        batch: list[str] = []

        for sentence in sentence_stream:
            batch.append(sentence)

            if len(batch) == self.batch_size:
                encoded_batch: Tensor = self.sentence_transformer.encode(
                    batch, convert_to_tensor=True, device=self.device)

                similarities: Tensor = cosine_similarity(
                    encoded_batch, query_embedding.unsqueeze(0)).squeeze()
                similarities = relu(similarities)
                one_minus_similarities: Tensor = 1 - similarities
                total_semantic_similarity *= one_minus_similarities.prod().item()

                batch = []

        # Process any remaining sentences in the final, smaller batch
        if batch:
            encoded_batch = self.sentence_transformer.encode(
                batch, convert_to_tensor=True, device=self.device)
            similarities = cosine_similarity(
                encoded_batch, query_embedding.unsqueeze(0)).squeeze()
            similarities = relu(similarities)
            one_minus_similarities = 1 - similarities
            total_semantic_similarity *= one_minus_similarities.prod().item()

        total_semantic_similarity = 1 - total_semantic_similarity
        return total_semantic_similarity

    def forward(self, query: str, lexical_similarities: list[float], sentence_streams: list[list[str]]) -> Tensor:
        all_semantic_similarities: list[float] = []
        for sentence_stream in sentence_streams:
            stream_similarity: float = self._process_sentence_stream(
                query, sentence_stream)
            all_semantic_similarities.append(stream_similarity)

        all_semantic_similarities_tensor: Tensor = torch.tensor(
            all_semantic_similarities, device=self.device)

        lexical_similarities_tensor: Tensor = torch.tensor(
            lexical_similarities, device=self.device)

        combined_similarities_tensor: Tensor = torch.stack(
            (all_semantic_similarities_tensor, lexical_similarities_tensor), dim=1)
        
        output: Tensor = self.linear_sigmoid_stack(combined_similarities_tensor)
        return output.squeeze()

        
