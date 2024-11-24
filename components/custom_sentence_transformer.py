import torch
from torch import nn, Tensor
from sentence_transformers import SentenceTransformer
from torch.nn.functional import cosine_similarity, relu, sigmoid


class CustomSentenceTransformer(nn.Module):
    def __init__(
            self,
            pretrained_model_name_or_path: str,
            device: str = "cpu",
            batch_size: int = 32,
            is_multilingual_retrival: bool = False,
            pretrained_model_name_or_path_for_query: str = ""
    ):
        """
        Args:
            pretrained_model_name_or_path (str): This param will be pass to the initailization of \
            SentenceTransformer instance to indicate which model that SentenceTransformer should use.\
            If it is a filepath on disc, it loads the model from that path.\
            If it is not a path, it first tries to download a pre-trained SentenceTransformer model.\
            If that fails, tries to construct a model from the Hugging Face Hub with that name.

            device (str): Device (like "cuda", "cpu", "mps", "npu") that indicate where the SentenceTransformer model \
            and all other computations of this class itself run.

            batch_size (int): Determine how many sentence chunks should be encode at once.

            is_multilingual_retrival (bool): Indicator variable representing that we are using this class in multilingual mode.\
            This mode will required `pretrained_model_name_or_path_for_query` to be passed, because we will need the second model \
            to processes the query which is in another language to the documents.

            pretrained_model_name_or_path_for_query (str): Similar to `pretrained_model_name_or_path`,\
            but the SentenceTransformer instance that is initialize with this is only used to encode the query.
        """
        super(CustomSentenceTransformer, self).__init__()

        self.document_sentence_transformer: SentenceTransformer = SentenceTransformer(
            pretrained_model_name_or_path, device=device)
        self.query_sentence_transformer: SentenceTransformer = self.document_sentence_transformer \
            if not is_multilingual_retrival \
            else SentenceTransformer(pretrained_model_name_or_path_for_query, device=device)
        self.device: str = device
        self.batch_size: int = batch_size

        self.linear_sigmoid_stack = nn.Sequential(
            nn.Linear(in_features=2, out_features=1, bias=True),
            nn.Sigmoid()
        ).to(device=device)

    def _process_sentence_stream(self, query_segmented: str, sentence_stream: list[str]) -> float:
        """
        Calculating the semantic similarity score of the query and all chunks from a single document only.

        Args:
            query_segmented (str): The query comes from users, which has been word-segmented by applying\
            `word_sentence_segment` and then `chunk_combiner` to recombine the segmented-sentences\
            to a single word-segmented sentence only.

            sentence_stream (list[str]): The list of all chunks that comes from only one document.

        Returns:
            float: The semantic similarity score between the query and the document.
        """
        total_semantic_similarity: float = 1.0

        query_embedding: Tensor = self.query_sentence_transformer.encode(
            query_segmented, convert_to_tensor=True, device=self.device)

        batch: list[str] = []

        for sentence in sentence_stream:
            batch.append(sentence)

            if len(batch) == self.batch_size:
                encoded_batch: Tensor = self.document_sentence_transformer.encode(
                    batch, convert_to_tensor=True, device=self.device)

                similarities: Tensor = cosine_similarity(
                    encoded_batch, query_embedding.unsqueeze(0)).squeeze()
                similarities = relu(similarities)
                one_minus_similarities: Tensor = 1 - similarities
                total_semantic_similarity *= one_minus_similarities.prod().item()

                batch = []

        # Process any remaining sentences in the final, smaller batch
        if batch:
            encoded_batch = self.document_sentence_transformer.encode(
                batch, convert_to_tensor=True, device=self.device)
            similarities = cosine_similarity(
                encoded_batch, query_embedding.unsqueeze(0)).squeeze()
            similarities = relu(similarities)
            one_minus_similarities = 1 - similarities
            total_semantic_similarity *= one_minus_similarities.prod().item()

        total_semantic_similarity = 1 - total_semantic_similarity
        return total_semantic_similarity

    def forward(self, query_segmented: str, lexical_or_topic_similarities: list[float], sentence_streams: list[list[str]]) -> Tensor:
        """
        Calculating the combined similarity score between the query and each documents from the input.

        Args:
            query_segmented (str): The query comes from users, which has been word-segment by applying\
            `word_sentence_segment` and then `chunk_combiner` to recombine the segmented-sentences\
            to a single word-segmented sentence only.

            lexical_or_topic_similarities (list[float]): The list holds the lexical (for monolingual) or topic (for multiligual)\
            similarity scores between the query and the correspond document from the input.

            sentence_streams (list[list[str]]): This list represents the input documents. Each element in this list is the\
            chunk list seperated from a single document that is considered relating to the query by lexical similarity or topic.

        Returns:
            Tensor: The tensor holds the combined similarity score between the query and each documents from the input
        """
        all_semantic_similarities: list[float] = []
        for sentence_stream in sentence_streams:
            stream_similarity: float = self._process_sentence_stream(
                query_segmented, sentence_stream)
            all_semantic_similarities.append(stream_similarity)

        all_semantic_similarities_tensor: Tensor = torch.tensor(
            all_semantic_similarities, device=self.device)

        lexical_or_topic_similarities_tensor: Tensor = torch.tensor(
            lexical_or_topic_similarities, device=self.device)

        combined_similarities_tensor: Tensor = torch.stack(
            (all_semantic_similarities_tensor, lexical_or_topic_similarities_tensor), dim=1)

        output: Tensor = self.linear_sigmoid_stack(
            combined_similarities_tensor)
        return output.squeeze()
