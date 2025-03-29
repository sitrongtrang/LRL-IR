from functools import reduce
import torch
from torch import nn, Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader
from components import LanguageProcessing, VietnameseLanguageProcessing
from components import DocumentDataset
from components import QueryDocDataset
from components import QueryExpansion
from components import LexicalMatching
from components import ChunkSeperator
from components import CustomSentenceTransformer
from components import FineTuneLanguageModel
from utils.utils import pos_neg_samples_gen_first_round, pos_neg_samples_gen_later_round, combine_doc_list

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
        language_processing: LanguageProcessing = VietnameseLanguageProcessing(),
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
            tuple[str,str]: The first string is the path to the fine-tuned `SentenceTransformer` model.\
            This can be loaded separately to do Knowledge Distillation later.\
            The second string is the path to the whole `CustomSentenceTransformer` model. Because `CustomSentenceTransformer`\
            has an attribute of class `SentenceTransformer`, which have different way to load the model,\
            we save it as a dictionary of two keys represents two part of `CustomSentenceTransformer`.
        """
        query_doc_dataloader = DataLoader(self.query_doc_dataset, batch_size=32, shuffle=True)
        print("Training started...")
        for epoch in range(self.epochs):
            print("-----------------------------")
            print(f"Epoch {epoch + 1}/{self.epochs}:")
            print("-----------------------------")
            for batch_count, sample_batch in enumerate(query_doc_dataloader):
                print(f"Batch {batch_count + 1} started.")
                loss = self._run_training_sample_batch(sample_batch)
                print(f"Batch {batch_count + 1} completed.")
                print(f"Loss: {loss.item()}")

        sentence_transformer_save_path: str = f"/sentence_transformer_finetune/{self.language}"
        custom_sentence_transformer_save_path: str = f"/custom_sentence_transformer_trained/{self.language}"

        self.custom_sentence_transformer.document_sentence_transformer.save(sentence_transformer_save_path)
        check_point = {
            'sentence_transformer_save_path': sentence_transformer_save_path,
            'linear_sigmoid_stack': self.custom_sentence_transformer.linear_sigmoid_stack.state_dict()
        }
        torch.save(check_point, custom_sentence_transformer_save_path + '/model.pth')
        
        print("Training completed.")
        print(f"Fine-tuned SentenceTransformer model saved at: {sentence_transformer_save_path}")
        print(f"CustomSentenceTransformer model saved at: {custom_sentence_transformer_save_path}")

        return sentence_transformer_save_path, custom_sentence_transformer_save_path
    
    def _run_training_sample_batch(
        self, 
        sample_batch: tuple[list[list[str]], list[list[str]], list[str]]
    ) -> Tensor:
        """
        Runs a single training batch for the monolingual retrieval model.
        Args:
            sample_batch (tuple[list[list[str]], list[list[str]], list[str]]): \
                A tuple containing the list of preprocessed query, list of tokenized query, and list of document id.
        Returns:
            final_loss (Tensor): The final loss value for the batch.
        """
        sample_count: int = len(sample_batch[0])
        final_loss: Tensor = 0.0
        for i in range(sample_count):
            print(f"Processing sample {i} in batch...")
            query_preprossed: list[str] = sample_batch[0][i]
            tokenized_query: list[str] = sample_batch[1][i]
            document_id: str = sample_batch[2][i]
            query_segmented: str = ""
            for part in query_preprossed:
              query_segmented += part
            print(f"Sample {i} in batch: data loaded...")
            # extended_query: list[str] = tokenized_query + self.query_expansion.get_expansion_term(tokenized_query)
            original_query_doc_ranking: list[tuple[str, float]] = self.lexical_matching.get_documents_ranking(tokenized_query)
            print(f"Sample {i} in batch: lexical matching completed...")
            # extended_query_doc_ranking: list[tuple[str, float]] = self.lexical_matching.get_documents_ranking(extended_query)
            original_query_relevant_doc_list: list[tuple[str, float]] = pos_neg_samples_gen_first_round(
                document_id, original_query_doc_ranking, FIRST_ROUND_NEGATIVE_SAMPLE_COUNT)
            print(f"Sample {i} in batch: positive/negative pair generated...")
            # extended_query_relevant_doc_list: list[tuple[str, float]] = pos_neg_samples_gen_first_round(
            #     document_id, extended_query_doc_ranking, FIRST_ROUND_NEGATIVE_SAMPLE_COUNT)
            # combine_lexical_relevant_doc_list: list[tuple[str, float]] = combine_doc_list(
            #     original_query_relevant_doc_list, extended_query_relevant_doc_list)

            lexical_relevant_doc_chunk_list: list[list[str]] = [self.chunk_seperator.get_chunks_of_document(pair[0]) 
                                                                for pair in original_query_relevant_doc_list]
            print(f"Sample {i} in batch: chunk list created...")                                                  
            lexical_similarity_score_list: list[float] = [pair[1] for pair in original_query_relevant_doc_list]

            self.custom_sentence_transformer.train()
            print(f"Sample {i} in batch: Sentence Trandformer started #1...")

            first_round_label_list: list[float] = [(1.0 if pair[0] == document_id else 0.0) for pair in original_query_relevant_doc_list]
            first_round_output, _ = self._run_training_custom_sentence_transformer_round(
                query_segmented,
                first_round_label_list, 
                lexical_relevant_doc_chunk_list, 
                lexical_similarity_score_list
            )

            print(f"Sample {i} in batch: Sentence Trandformer started #2...")
            (second_round_doc_chunk_list, 
             second_round_label_list, 
             second_round_lexical_similarity_score_list) = pos_neg_samples_gen_later_round(
                 first_round_output, 
                 first_round_label_list, 
                 lexical_similarity_score_list, 
                 lexical_relevant_doc_chunk_list, 
                 SECOND_ROUND_NEGATIVE_SAMPLE_COUNT)
            second_round_output, _ = self._run_training_custom_sentence_transformer_round(
                query_segmented,
                second_round_label_list, 
                second_round_doc_chunk_list, 
                second_round_lexical_similarity_score_list)

            print(f"Sample {i} in batch: Sentence Trandformer started #3...")
            (third_round_doc_chunk_list, 
             third_round_label_list, 
             third_round_lexical_similarity_score_list) = pos_neg_samples_gen_later_round(
                    second_round_output, second_round_label_list, 
                    second_round_lexical_similarity_score_list, 
                    second_round_doc_chunk_list, 
                    THIRD_ROUND_NEGATIVE_SAMPLE_COUNT)
            _, third_round_loss = self._run_training_custom_sentence_transformer_round(
                query_segmented,
                third_round_label_list, 
                third_round_doc_chunk_list, 
                third_round_lexical_similarity_score_list)
            
            final_loss = third_round_loss
            
        return final_loss
    
    def _run_training_custom_sentence_transformer_round(
        self, 
        query_segmented: str,
        label_list: list[float], 
        doc_chunk_list: list[list[str]], 
        similarity_score_list: list[float]
    ) -> tuple[Tensor, Tensor]:
        """
        Runs a single training round for the sentence-transformer monolingual retrieval model.
        Args:
            query_segmented (str): The segmented query string.
            label_list (list[float]): A list of float labels corresponding to the relevance of each document chunk.
            doc_chunk_list (list[list[str]]): A list of document chunks, where each chunk is a list of strings.
            similarity_score_list (list[float]): A list of similarity scores for each document chunk.
        Returns:
            round_output (Tensor): The output tensor from the custom sentence transformer.
        """
        self.optimizer.zero_grad()
        label_tensor: Tensor = torch.tensor(label_list, device=self.device)
        round_output: Tensor = self.custom_sentence_transformer(
            query_segmented,
            similarity_score_list,
            doc_chunk_list
        )
        round_loss: Tensor = self.contrastive_loss_fn(round_output, label_tensor)
        round_loss.backward()
        self.optimizer.step()

        return round_output, round_loss
