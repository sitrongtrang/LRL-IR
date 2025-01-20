import torch
from torch import nn, Tensor
from sentence_transformers import SentenceTransformer
from ..components.ot_solver import OTSolver
from ..components.dataset import *
from ..utils.utils import compute_cosine_cost_matrix, pad_sentences

class KnowledgeDistillation:
    def __init__(
        self, 
        parallel_dir: str,
        teacher_language_processing: LanguageProcessing,
        student_language_processing: LanguageProcessing,
        student_model_language: str,
        teacher_model_language: str="vie",
        device: str = "cpu",
        batch_size: int = 32,
        epochs: int = 4,
        learning_rate: float = 1e-5,
        epsilon: float = 0.1,
    ) -> None:
        """
        Args:
            parallel_dir (str): Path to the folder containing CSV files with parallel sentences.
            teacher_language_processing (LanguageProcessing): Language processing for teacher model
            student_language_processing (LanguageProcessing): Language processing for student model
            teacher_model_language (str): The language used in the monolingual training phase.
            student_model_language (str): The language to be learned by the student model.

            device (str): Device (like "cuda", "cpu", "mps", "npu") that indicate where all models and computations run.

            batch_size (int): Determine how many sentence chunks should be encode at once in SentenceTransformer.

            epochs (int): Indicate total number of iterations of all the training data.
  
            learning_rate (float): The learning rate of the traning process.

            epsilon (float): Regularization parameter for optimal transport
        """
        self.parallel_dir: str = parallel_dir
        self.teacher_model_language: str = teacher_model_language
        self.student_model_language: str = student_model_language
        self.bitext_data: ParallelDataset = ParallelDataset(self.parallel_dir, teacher_language_processing, student_language_processing)
        self.teacher_model: str = f"/sentence_transformer_finetune/{self.teacher_model_language}"
        self.student_model: str = f"/sentence_transformer_multilingual"

        self.device: str = device
        self.batch_size: int = batch_size
        self.epochs: int = epochs
        self.learning_rate: float = learning_rate
        self.epsilon: float = epsilon

        self.teacher: SentenceTransformer = SentenceTransformer(self.teacher_model, self.device, self.batch_size)
        self.student: SentenceTransformer = SentenceTransformer(self.student_model, self.device, self.batch_size)
        for param in self.teacher.parameters():
            param.requires_grad = False
    
        self.teacher.to(self.device)
        self.student.to(self.device)

        self.optimizer = torch.optim.AdamW(self.student.parameters(), lr=self.learning_rate)

        self.ot_solver: OTSolver = OTSolver()

    def train(self) -> str:
        """
        Train the student model using knowledge distillation.

        Returns:
            str: The path to the multilingual sentence transformer
        """
        for epoch in range(self.epochs):
            for source_sentence, target_sentence in self.bitext_data:    
                padded_source, padded_target, features = pad_sentences(source_sentence, target_sentence)

                self.optimizer.zero_grad()      
                with torch.no_grad():
                    teacher_embeddings: Tensor = self.teacher.forward(
                        features={
                            'input_ids': torch.tensor([[self.teacher.tokenizer.convert_tokens_to_ids(padded_source)]]),
                            'attention_mask': features['source_attention_mask']
                        }
                    )['token_embeddings'].squeeze(0)
                
                student_embeddings_source: Tensor = self.student.forward(
                    features={
                        'input_ids': torch.tensor([[self.student.tokenizer.convert_tokens_to_ids(padded_source)]]),
                        'attention_mask': features['source_attention_mask']
                    }
                )['token_embeddings'].squeeze(0)

                student_embeddings_target: Tensor = self.student.forward(
                    features={
                        'input_ids': torch.tensor([[self.student.tokenizer.convert_tokens_to_ids(padded_target)]]),
                        'attention_mask': features['target_attention_mask']
                    }
                )['token_embeddings'].squeeze(0)
                
                uniform_dist: Tensor = torch.ones(len(padded_target))/len(padded_target)

                cost_source: Tensor = compute_cosine_cost_matrix(teacher_embeddings, student_embeddings_source)
                _, source_loss = self.ot_solver(uniform_dist, uniform_dist, cost_source)

                cost_target: Tensor = compute_cosine_cost_matrix(teacher_embeddings, student_embeddings_target) 
                _, target_loss = self.ot_solver(uniform_dist, uniform_dist, cost_target)
                
                loss: Tensor = source_loss + target_loss
                
                loss.backward()
                self.optimizer.step()


        self.student.save(self.student_model)
        check_point = {
            'student_sentence_transformer_save_path': self.student_model
        }
        torch.save(check_point, self.student_model + '/model.pth')

        return self.student_model