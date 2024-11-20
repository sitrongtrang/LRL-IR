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
    """
    This class represents the contrastive loss function and is used to calcualate the loss value.
    """
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
        chunk_length_limit: int = 128,
        device: str = "cpu",
        batch_size: int = 32,
        margin: float = 1.0,
        learning_rate: float = 1e-5,
        epochs: int = 4
    ) -> None:
        """
        Args:
            document_dir (str): ABSOLUTE path to the directory containing documents with title, topic, and content xml-tag.

            processed_doc_store_dir (str): processed_doc_store_dir (str): ABSOLUTE path to the directory where you want to store preprocessed-documents.

            qd_dir (str): Path to the folder containing CSV files with queries and corresponding answer document file paths (ABSOLUTE path).

            pretrained_model_name_or_path (str): A string - the model id of a pretrained language model hosted\
            inside a model repo on huggingface.co (e.g: `vinai/phobert-base-v2`, `FacebookAI/roberta-base`,...).\
            OR a path to a directory containing your own language model. This model should be based on a transformer model\
            such as BERT, RoBERTa, or other Hugging Face models.

            do_mlm_fine_tune (bool): Indicator variable telling whether or not to do mlm fine-tune task for the language model.

            language (str): Language of the query and documents. Since this is class for monolingual-training, the language\
            of the query and documents must be the same.

            language_processing (LanguageProcessing): Language processing object for the language.

            chunk_length_limit (int): The limit length of each chunk. Representing the max number of tokens in each chunk\
            when seperating the document.

            device (str): Device (like "cuda", "cpu", "mps", "npu") that indicate where all models and computations run.

            batch_size (int): Determine how many sentence chunks should be encode at once in SentenceTransformer.

            margin (float): The value of the margin in the contrastive-loss equation.

            learning_rate (float): The learning rate of the traning process.

            epochs (int): Indicate total number of iterations of all the training data.
        """
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
        self.lexical_matching: LexicalMatching = LexicalMatching(self.document_dataset)
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
        """
        Training entry point.

        Returns:
            tuple[str, str]: The first string is the path to the fine-tuned `SentenceTransformer` model.\
            This can be loaded seperately to do Knowledge Distillation later.\
            The second string is the path to the whole `CustomSentenceTransformer` model. Because `CustomSentenceTransformer`\
            has an attribute of class `SentenceTransformer`, which have different way to load the model,\
            we save it as a dictionary of two keys represents two part of `CustomSentenceTransformer`.
        """
        for epoch in range(self.epochs):
            for query_segmented, document_file_path_list in self.query_doc_dataset:
                tokenized_query: list[str] = self.language_processing.tokenizer(query_segmented)
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

        # For monoligual retrival, we will not save `query_sentence_transformer` 
        # cause both `document_sentence_transformer` and `query_sentence_transformer` are the same instance.
        sentence_transformer_save_path: str = f"/sentence_transformer_finetune/{self.language}"
        custom_sentence_transformer_save_path: str = f"/custom_sentence_transformer_trained/{self.language}"

        self.custom_sentence_transformer.document_sentence_transformer.save(
            sentence_transformer_save_path)
        check_point = {
            'sentence_transformer_save_path': sentence_transformer_save_path,
            'linear_sigmoid_stack': self.custom_sentence_transformer.linear_sigmoid_stack.state_dict()
        }
        torch.save(
            check_point, custom_sentence_transformer_save_path + '/model.pth')

        return sentence_transformer_save_path, custom_sentence_transformer_save_path
