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

FIRST_ROUND_NEGATIVE_SAMPLE_COUNT = 35
SECOND_ROUND_NEGATIVE_SAMPLE_COUNT = 20
THIRD_ROUND_NEGATIVE_SAMPLE_COUNT = 15


class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, outputs: Tensor, labels: Tensor) -> Tensor:
        """
        outputs: Tensor of shape (N,)
        labels: Tensor of shape (N,), where 1 indicates relevant and 0 indicates non-relevant
        """
        # Compute the positive pair loss (y * d^2)
        positive_loss: Tensor = labels * outputs ** 2

        # Compute the negative pair loss ((1 - y) * max(0, margin - d)^2)
        margin_diff: Tensor = torch.relu(self.margin - outputs)
        negative_loss: Tensor = (1 - labels) * margin_diff ** 2

        # Combine positive and negative losses
        loss: Tensor = 0.5 * (positive_loss + negative_loss).mean()
        return loss


class MonolingualRetrivalTrainer:
    def __init__(
        self,
        document_dir: str,
        processed_doc_store_dir: str,
        qd_dir: str,
        pretrained_model_name_or_path: str,
        do_mlm_fine_tune: bool = True,
        language: str = 'vie',
        language_processing: LanguageProcessing = LanguageProcessing('vie'),
        lexical_number_to_choose: int = 30,
        chunk_length_limit: int = 128,
        device: str = "cpu",
        batch_size: int = 32,
        margin: float = 1.0,
        learning_rate: float = 1e-5,
        epochs: int = 4
    ) -> None:
        super().__init__()
        self.language = language
        self.document_dataset: DocumentDataset = DocumentDataset(
            document_dir,
            processed_doc_store_dir,
            language,
            language_processing
        )
        self.query_doc_dataset = QueryDocDataset(
            qd_dir,
            language,
            language_processing
        )
        self.query_expansion: QueryExpansion = QueryExpansion(
            self.document_dataset)
        self.lexical_matching: LexicalMatching = LexicalMatching(
            self.document_dataset,
            training=True,
            number_to_choose=lexical_number_to_choose
        )
        self.chunk_seperator: ChunkSeperator = ChunkSeperator(
            self.document_dataset,
            chunk_length_limit
        )
        self.base_language_model: str = pretrained_model_name_or_path
        if do_mlm_fine_tune:
            fine_tune_language_model = FineTuneLanguageModel(
                self.document_dataset, pretrained_model_name_or_path)
            self.base_language_model = fine_tune_language_model.train()
        self.custom_sentence_transformer = CustomSentenceTransformer(
            self.base_language_model,
            device,
            batch_size
        ).to(device=device)
        self.contrastive_loss_fn: ContrastiveLoss = ContrastiveLoss(margin).to(device)
        self.optimizer = Adam(
            self.custom_sentence_transformer.parameters(), lr=learning_rate)
        self.epochs: int = epochs
        self.device: str = device
        self.language_processing: LanguageProcessing = language_processing

    def train(self) -> tuple[str, str]:
        for epoch in range(self.epochs):
            for query_segmented, document_file_path_list in self.query_doc_dataset:
                tokenized_query: list[str] = reduce(
                    lambda prev, curr: prev + self.language_processing.tokenizer(curr), query_segmented, [])
                extended_query: list[str] = tokenized_query + \
                    self.query_expansion.get_expansion_term(tokenized_query)
                original_query_doc_ranking: list[tuple[str, float]] = self.lexical_matching.get_documents_ranking(
                    tokenized_query)
                extended_query_doc_ranking: list[tuple[str, float]] = self.lexical_matching.get_documents_ranking(
                    extended_query)
                original_query_relevant_doc_list: list[tuple[str, float]] = pos_neg_samples_gen_first_round(
                    document_file_path_list, original_query_doc_ranking, FIRST_ROUND_NEGATIVE_SAMPLE_COUNT)
                extended_query_relevant_doc_list: list[tuple[str, float]] = pos_neg_samples_gen_first_round(
                    document_file_path_list, extended_query_doc_ranking, FIRST_ROUND_NEGATIVE_SAMPLE_COUNT)
                combine_lexical_relevant_doc_list: list[tuple[str, float]] = combine_doc_list(
                    original_query_relevant_doc_list, extended_query_relevant_doc_list)

                lexical_relevant_doc_chunk_list: list[list[str]] = [self.chunk_seperator.get_chunks_of_document(
                    pair[0]) for pair in combine_lexical_relevant_doc_list]
                lexical_similarity_score_list: list[float] = [
                    pair[1] for pair in combine_lexical_relevant_doc_list]

                self.custom_sentence_transformer.train()

                # First round
                self.optimizer.zero_grad()
                first_round_label_list: list[float] = [(1.0 if pair[0] in document_file_path_list else 0.0)
                                                       for pair in combine_lexical_relevant_doc_list]
                first_round_label_tensor: Tensor = torch.tensor(
                    first_round_label_list, device=self.device)
                first_round_output: Tensor = self.custom_sentence_transformer(
                    query_segmented,
                    lexical_similarity_score_list,
                    lexical_relevant_doc_chunk_list
                )
                first_round_loss: Tensor = self.contrastive_loss_fn(
                    first_round_output, first_round_label_tensor)
                first_round_loss.backward()
                self.optimizer.step()

                # Second round
                self.optimizer.zero_grad()
                (second_round_doc_chunk_list,
                 second_round_label_list,
                 second_round_lexical_similarity_score_list) = pos_neg_samples_gen_later_round(
                    first_round_output,
                    first_round_label_list,
                    lexical_similarity_score_list,
                    lexical_relevant_doc_chunk_list,
                    SECOND_ROUND_NEGATIVE_SAMPLE_COUNT
                )
                second_round_label_tensor: Tensor = torch.tensor(
                    second_round_label_list, device=self.device)
                second_round_output: Tensor = self.custom_sentence_transformer(
                    query_segmented,
                    second_round_lexical_similarity_score_list,
                    second_round_doc_chunk_list
                )
                second_round_loss: Tensor = self.contrastive_loss_fn(
                    second_round_output, second_round_label_tensor)
                second_round_loss.backward()
                self.optimizer.step()

                # Third round
                self.optimizer.zero_grad()
                (third_round_doc_chunk_list,
                 third_round_label_list,
                 third_round_lexical_similarity_score_list) = pos_neg_samples_gen_later_round(
                    second_round_output,
                    second_round_label_list,
                    second_round_lexical_similarity_score_list,
                    second_round_doc_chunk_list,
                    THIRD_ROUND_NEGATIVE_SAMPLE_COUNT
                )
                third_round_label_tensor: Tensor = torch.tensor(
                    third_round_label_list, device=self.device)
                third_round_output: Tensor = self.custom_sentence_transformer(
                    query_segmented,
                    third_round_lexical_similarity_score_list,
                    third_round_doc_chunk_list
                )
                third_round_loss: Tensor = self.contrastive_loss_fn(
                    third_round_output, third_round_label_tensor)
                third_round_loss.backward()
                self.optimizer.step()

        sentence_transformer_save_path: str = f"/sentence_transformer_finetune/{self.language}"
        custom_sentence_transformer_save_path: str = f"/custom_sentence_transformer_trained/{self.language}"

        self.custom_sentence_transformer.sentence_transformer.save(
            sentence_transformer_save_path)
        check_point = {
            'sentence_transformer_save_path': sentence_transformer_save_path,
            'linear_sigmoid_stack': self.custom_sentence_transformer.linear_sigmoid_stack.state_dict()
        }
        torch.save(
            check_point, custom_sentence_transformer_save_path + '/model.pth')

        return sentence_transformer_save_path, custom_sentence_transformer_save_path
