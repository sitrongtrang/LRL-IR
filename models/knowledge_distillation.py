from pandas import read_csv
import torch
from torch import nn, Tensor
from sentence_transformers import SentenceTransformer
from components.ot_solver import OTSolver
from components.dataset.document_dataset import *
from utils.utils import compute_cosine_cost_matrix, pad_sentences, tf_idf_dist, uniform_dist, l1_normalize
from dotenv import load_dotenv
import os

load_dotenv()

class KnowledgeDistillation:
    def __init__(
        self, 
        # parallel_dir: str,
        # teacher_language_processing: LanguageProcessing,
        # student_language_processing: LanguageProcessing,
        student_model_language: str,
        teacher_model_language: str,
        teacher_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        distribution: str = "padded_uniform", 
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
            teacher_model (str): The base model for the teacher

            distribution (str): The distribution for tokens in the sentences

            device (str): Device (like "cuda", "cpu", "mps", "npu") that indicate where all models and computations run.

            batch_size (int): Determine how many sentence chunks should be encode at once in SentenceTransformer.

            epochs (int): Indicate total number of iterations of all the training data.
  
            learning_rate (float): The learning rate of the traning process.

            epsilon (float): Regularization parameter for optimal transport
        """
        # self.parallel_dir: str = parallel_dir
        self.teacher_model_language: str = teacher_model_language
        self.student_model_language: str = student_model_language
        # self.bitext_data: ParallelDataset = ParallelDataset(self.parallel_dir, teacher_language_processing, student_language_processing)
        
        self.distribution = distribution

        self.device: str = device
        self.batch_size: int = batch_size
        self.epochs: int = epochs
        self.learning_rate: float = learning_rate
        self.epsilon: float = epsilon

        self.teacher: SentenceTransformer = SentenceTransformer(teacher_model, device=self.device, token=os.getenv("HUGGINGFACE_TOKEN"))
        self.student: SentenceTransformer = SentenceTransformer(teacher_model, device=self.device, token=os.getenv("HUGGINGFACE_TOKEN"))
        for param in self.teacher.parameters():
            param.requires_grad = False
    
        self.teacher.to(self.device)
        self.student.to(self.device)

        self.optimizer = torch.optim.AdamW(self.student.parameters(), lr=self.learning_rate)

        self.ot_solver: OTSolver = OTSolver()

    def train_loop(self, source_sentence: str, target_sentence: str):
        self.optimizer.zero_grad()    

        source_sentence = self.teacher.tokenizer.tokenize(source_sentence)
        target_sentence = self.teacher.tokenizer.tokenize(target_sentence)
        if "padded" in self.distribution:
            source_sentence, target_sentence = pad_sentences(source_sentence, target_sentence, self.teacher.tokenizer.pad_token)

        if self.distribution == "tf-idf":
            # source_dist = tf_idf_dist(source_sentence)
            pass
        elif "uniform" in self.distribution:
            source_dist = uniform_dist(source_sentence)
            target_dist = uniform_dist(target_sentence)

        source_dist = l1_normalize(source_dist)
        target_dist = l1_normalize(target_dist)  

        source_sentence = self.teacher.tokenizer.convert_tokens_to_string(source_sentence)
        target_sentence = self.teacher.tokenizer.convert_tokens_to_string(target_sentence)

        with torch.no_grad():
            teacher_embeddings: Tensor = self.teacher.encode(
                source_sentence, 
                output_value='token_embeddings', 
                convert_to_tensor=True, 
                device=self.device
            )
        
        student_embeddings_source: Tensor = self.student.encode(
            source_sentence, 
            output_value='token_embeddings', 
            convert_to_tensor=True, 
            device=self.device
        )
        
        student_embeddings_target: Tensor = self.student.encode(
            target_sentence, 
            output_value='token_embeddings', 
            convert_to_tensor=True, 
            device=self.device
        )

        cost_source: Tensor = compute_cosine_cost_matrix(teacher_embeddings, student_embeddings_source)
        _, source_loss = self.ot_solver(source_dist, source_dist, cost_source)

        cost_target: Tensor = compute_cosine_cost_matrix(teacher_embeddings, student_embeddings_target) 
        _, target_loss = self.ot_solver(source_dist, target_dist, cost_target)
        
        loss: Tensor = source_loss + target_loss
        
        loss.backward()
        self.optimizer.step()

    def train(self) -> str:
        """
        Train the student model using knowledge distillation.

        Returns:
            str: The path to the multilingual sentence transformer
        """
        print("Start training")
        for epoch in range(self.epochs):
            # for source_sentence, target_sentence in self.bitext_data:
            #     self.train_loop(source_sentence, target_sentence)

            df = read_csv("test_bitext.csv")
            bitext_data = list(zip(df["source"], df["target"]))
            for source_sentence, target_sentence in bitext_data:
                self.train_loop(source_sentence, target_sentence)

        self.student.save(f"/sentence_transformer_multilingual")
        check_point = {
            'student_sentence_transformer_save_path': f"/sentence_transformer_multilingual"

        }
        torch.save(check_point, f"/sentence_transformer_multilingual" + '/model.pth')

        return f"/sentence_transformer_multilingual"