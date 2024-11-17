from functools import reduce
import torch
from torch import nn, Tensor
from components.dataset import LanguageProcessing, DocumentDataset
from components.query_expansion import QueryExpansion
from components.lexical_matching import LexicalMatching
from components.chunk_seperator import ChunkSeperator
from components.custom_sentence_transformer import CustomSentenceTransformer
from utils import combine_doc_list


class MonoLingualRetrival(nn.Module):
    """
    ATTENTION: This class is designed to be used in production only and not for training.
    If you want to train the model, please use the Trainer class to train the sub-model 
    and then pass the directory saving the trained-model into the constructor of this class.
    """
    def __init__(
        self,
        document_dir: str,
        processed_doc_store_dir: str,
        custom_sentence_transformer_pretrained_or_save_path: str,
        language: str = 'vie',
        language_processing: LanguageProcessing = LanguageProcessing('vie'),
        original_query_doc_count: int = 30,
        extended_query_doc_count: int = 30,
        chunk_length_limit: int = 128,
        relevant_threshold: float = 0.1,
        relevant_default_lowerbound: float = 0.25,
        device: str = "cpu",
        batch_size: int = 32
    ) -> None:
        """
        Args:
            document_dir (str): ABSOLUTE path to the directory containing documents with title, topic, and content xml-tag.

            processed_doc_store_dir (str): processed_doc_store_dir (str): ABSOLUTE path to the directory where you want to store preprocessed-documents.

            custom_sentence_transformer_pretrained_or_save_path (str): a path to a directory containing the trained\
            `CustomSentenceTransformer` model. 

            language (str): Language of the query and documents. Since this is class for monolingual retrival, the language\
            of the query and documents must be the same.

            language_processing (LanguageProcessing): Language processing object for the language.

            original_query_doc_count (int): Define how many documents should be retrived from the lexical-retrival phase,
            using the original query. The higher this number is, the more documents will be passed to the\
            semantic-retrival phase, which used SentenceTransformer model that can be very resource-consuming.\
            So please choose an appropriate number, based on how many documents at max that you expect should be relevant for a queries.

            extended_query_doc_count (int): Define how many documents should be retrived from the lexical-retrival phase,
            using the extended query. The higher this number is, the more documents will be passed to the\
            semantic-retrival phase, which used SentenceTransformer model that can be very resource-consuming.\
            So please choose an appropriate number, based on how many documents at max that you expect should be relevant for a queries.

            chunk_length_limit (int): The limit length of each chunk. Representing the max number of tokens in each chunk\
            when seperating the document.

            relevant_threshold (float): The threshold for choosing documents. The documents would be chosen if their \
            similarity scores in range `[max_relevant_score - relevant_threshold, max_relevant_score]`.

            relevant_default_lowerbound (float): The minimum similarity score required for a document to be chosen.\
            If this value is higher than `max_relevant_score - relevant_threshold`, this value will be chosen\
            as the lower bound instead of `max_relevant_score - relevant_threshold`.

            device (str): Device (like "cuda", "cpu", "mps", "npu") that indicate where all models and computations run.

            batch_size (int): Determine how many sentence chunks should be encode at once in SentenceTransformer.
        """
        super().__init__()
        self.language = language
        self.document_dataset: DocumentDataset = DocumentDataset(
            document_dir,
            processed_doc_store_dir,
            language,
            language_processing
        )
        self.query_expansion: QueryExpansion = QueryExpansion(
            self.document_dataset)
        self.lexical_matching: LexicalMatching = LexicalMatching(self.document_dataset)
        self.chunk_seperator: ChunkSeperator = ChunkSeperator(
            self.document_dataset,
            chunk_length_limit
        )

        # For monolingual, we just need to load one model, because both
        # `document_sentence_transformer` and `query_sentence_transformer` are the same instance
        # and will have with the same model's parameters.
        checkpoint = torch.load(
            custom_sentence_transformer_pretrained_or_save_path)
        self.custom_sentence_transformer = CustomSentenceTransformer(
            checkpoint['sentence_transformer_save_path'],
            device=device,
            batch_size=batch_size
        ).to(device=device)
        self.custom_sentence_transformer.linear_sigmoid_stack.load_state_dict(
            checkpoint['linear_sigmoid_stack'])
        
        self.original_query_doc_count: int = original_query_doc_count
        self.extended_query_doc_count: int = extended_query_doc_count
        self.device: str = device
        self.language_processing: LanguageProcessing = language_processing
        self.relevant_threshold: float = relevant_threshold
        self.relevant_default_lowerbound: float = relevant_default_lowerbound

    def forward(self, query: str) -> list[str]:
        """
        Retrive a list of file paths to the documents that this model considered to be related to the query.

        Args:
            query (str): The user's query.

        Returns:
            list[str]: a list of file paths to the documents which are considered to be related to the query.
        """
        query_segmented: list[str] = self.language_processing.word_sentence_segment(
            query)
        tokenized_query: list[str] = reduce(
            lambda prev, curr: prev + self.language_processing.tokenizer(curr), query_segmented, [])
        extended_query: list[str] = tokenized_query + \
            self.query_expansion.get_expansion_term(tokenized_query)
        original_query_doc_ranking: list[tuple[str, float]] = self.lexical_matching.get_documents_ranking(
            tokenized_query)
        extended_query_doc_ranking: list[tuple[str, float]] = self.lexical_matching.get_documents_ranking(
            extended_query)
        combine_lexical_relevant_doc_list: list[tuple[str, float]] = combine_doc_list(
            original_query_doc_ranking[:self.original_query_doc_count],
            extended_query_doc_ranking[:self.extended_query_doc_count]
        )

        lexical_relevant_doc_chunk_list: list[list[str]] = [self.chunk_seperator.get_chunks_of_document(
            pair[0]) for pair in combine_lexical_relevant_doc_list]
        lexical_similarity_score_list: list[float] = [
            pair[1] for pair in combine_lexical_relevant_doc_list]

        self.custom_sentence_transformer.eval()

        with torch.no_grad():
            output: Tensor = self.custom_sentence_transformer(
                query_segmented,
                lexical_similarity_score_list,
                lexical_relevant_doc_chunk_list
            )
            max_relevant = output.max()
            lower_bound = max(max_relevant - self.relevant_threshold, self.relevant_default_lowerbound)
            upper_bound = max_relevant

            mask = (output >= lower_bound) & (output <= upper_bound)

            indices: list[int] = torch.nonzero(mask, as_tuple=False).squeeze().tolist()
            result: list[str] = [combine_lexical_relevant_doc_list[i][0] for i in indices]
            return result
