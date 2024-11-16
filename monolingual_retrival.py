from functools import reduce
import torch
from torch import nn, Tensor
from torch.optim import Adam
from components.dataset import LanguageProcessing, DocumentDataset, QueryDocDataset
from components.query_expansion import QueryExpansion
from components.lexical_matching import LexicalMatching
from components.chunk_seperator import ChunkSeperator
from components.custom_sentence_transformer import CustomSentenceTransformer
from components.fine_tune_language_model import FineTuneLanguageModel
from utils import pos_neg_samples_gen_first_round, pos_neg_samples_gen_later_round, combine_doc_list


class MonoLingualRetrival(nn.Module):
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
        self.lexical_matching: LexicalMatching = LexicalMatching(
            self.document_dataset,
            training=False
        )
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
