import csv
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
        teacher_model: str = "sentence-transformers/distiluse-base-multilingual-cased-v2",
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
        self.teacher.max_seq_length = 512
        self.student.max_seq_length = 512
        for param in self.teacher.parameters():
            param.requires_grad = False
    
        self.teacher.to(self.device)
        self.student.to(self.device)

        self.optimizer = torch.optim.AdamW(self.student.parameters(), lr=self.learning_rate)

        self.ot_solver: OTSolver = OTSolver(self.device)

    def train_loop(self, source_sentence: str, target_sentence: str):
        print(source_sentence)
        self.optimizer.zero_grad()    

        source_tokens = source_sentence.split(' ')
        target_tokens = target_sentence.split(' ')

        if "padded" in self.distribution:
            source_tokens, target_tokens = pad_sentences(source_tokens, target_tokens, self.teacher.tokenizer.pad_token)
            
        if self.distribution == "tf-idf":
            source_sentence_list = []
            target_sentence_list = []
            with open('test_bitext.csv', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    source_sentence_list.append(row['source'])
                    target_sentence_list.append(row['target'])
            source_dist = tf_idf_dist(source_tokens, source_sentence, source_sentence_list, self.device)
            target_dist = tf_idf_dist(target_tokens, target_sentence, target_sentence_list, self.device)
        elif "uniform" in self.distribution:
            source_dist = uniform_dist(source_tokens, self.device)
            target_dist = uniform_dist(target_tokens, self.device)

        source_dist = l1_normalize(source_dist)
        target_dist = l1_normalize(target_dist)  

        source_ids = self.teacher.tokenizer.convert_tokens_to_ids(source_tokens)
        source_ids = torch.tensor([source_ids], device=self.device)
        attention_mask = [1 if token != self.teacher.tokenizer.pad_token else 0 for token in source_tokens]
        attention_mask = torch.tensor([attention_mask], device=self.device)
        source_encoded = {
            'input_ids': source_ids,
            'attention_mask': attention_mask
        }

        target_ids = self.student.tokenizer.convert_tokens_to_ids(target_tokens)
        target_ids = torch.tensor([target_ids], device=self.device)
        attention_mask = [1 if token != self.student.tokenizer.pad_token else 0 for token in target_tokens]
        attention_mask = torch.tensor([attention_mask], device=self.device)
        target_encoded = {
            'input_ids': target_ids,
            'attention_mask': attention_mask
        }

        with torch.no_grad():
            source_embeddings: Tensor = self.teacher.forward(source_encoded)['token_embeddings'].squeeze(0)
        
        target_embeddings: Tensor = self.student.forward(target_encoded)['token_embeddings'].squeeze(0)

        cost: Tensor = compute_cosine_cost_matrix(source_embeddings, target_embeddings)
        plan, loss = self.ot_solver(source_dist, target_dist, cost)

        print(cost, plan)
        
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

        self.student.save(f"sentence_transformer_multilingual_" + self.distribution)
        check_point = {
            'student_sentence_transformer_save_path': f"sentence_transformer_multilingual_" + self.distribution

        }
        torch.save(check_point, f"sentence_transformer_multilingual_"  + self.distribution + '/model.pth')

        return f"sentence_transformer_multilingual_" + self.distribution