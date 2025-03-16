import csv
import heapq
from pandas import read_csv
import py_vncorenlp
import torch
from torch import nn, Tensor
from sentence_transformers import SentenceTransformer
from components.ot_solver import OTSolver
from components.dataset.document_dataset import *
from utils.utils import compute_cosine_cost_matrix, pad_sentences, tf_idf_dist, uniform_dist, l1_normalize, get_language_processor, roberta_dist
from dotenv import load_dotenv
from collections import defaultdict
import os

load_dotenv()

class KnowledgeDistillation:
    def __init__(
        self, 
        teacher_model_language: str,
        student_model_language: str,
        teacher_model: str = "distiluse-base-multilingual-cased-v2",
        student_model: str = "xlm-roberta-base",
        bitext_data: str = os.getenv("PROJECT_DIR") + "bitext.csv",
        save_dir: str = os.getenv("PROJECT_DIR"),
        distribution: str = "padded_uniform", 
        device: str = "cpu",
        batch_size: int = 32,
        epochs: int = 4,
        learning_rate: float = 1e-5,
        epsilon: float = 0.1,
    ) -> None:
        """
        Args:
            teacher_model_language (str): The language used in the monolingual training phase.
            student_model_language (str): The language to be learned by the student model.
            teacher_model (str): The base model for the teacher

            bitext_data (str): Path to the CSV file with parallel sentences.
            save_dir (str): The folder to save the trained model

            distribution (str): The distribution for tokens in the sentences

            device (str): Device (like "cuda", "cpu", "mps", "npu") that indicate where all models and computations run.

            batch_size (int): Determine how many sentence chunks should be encode at once in SentenceTransformer.

            epochs (int): Indicate total number of iterations of all the training data.
  
            learning_rate (float): The learning rate of the traning process.

            epsilon (float): Regularization parameter for optimal transport
        """
        self.teacher_model_language: str = teacher_model_language
        self.student_model_language: str = student_model_language
        
        self.teacher_language_processing = get_language_processor(teacher_model_language)
        self.student_language_processing = get_language_processor(student_model_language)

        self.bitext_data: str = bitext_data
        self.save_dir: str = save_dir

        self.distribution = distribution

        self.device: str = device
        self.batch_size: int = batch_size
        self.epochs: int = epochs
        self.learning_rate: float = learning_rate
        self.epsilon: float = epsilon

        self.teacher: SentenceTransformer = SentenceTransformer(teacher_model, device=self.device, token=os.getenv("HUGGINGFACE_TOKEN"))
        self.student: SentenceTransformer = SentenceTransformer(student_model, device=self.device, token=os.getenv("HUGGINGFACE_TOKEN"))
        self.teacher.max_seq_length = 512
        self.student.max_seq_length = 512
        for param in self.teacher.parameters():
            param.requires_grad = False
    
        self.teacher.to(self.device)
        self.student.to(self.device)

        self.optimizer = torch.optim.AdamW(self.student.parameters(), lr=self.learning_rate)

        self.ot_solver: OTSolver = OTSolver(self.device)

    def optical(self, source_sentence: str, target_sentence: str):
        source_tokens = self.teacher_language_processing.text_preprocessing(source_sentence)
        target_tokens = self.student_language_processing.text_preprocessing(target_sentence)

        print(target_tokens)

        new_source_tokens = [token for token in source_tokens if token not in self.teacher.tokenizer.get_vocab()]
        if new_source_tokens:
            self.teacher.tokenizer.add_tokens(new_source_tokens)
            self.teacher[0].auto_model.resize_token_embeddings(len(self.teacher.tokenizer))
            self.optimizer.state = defaultdict(dict)

        new_target_tokens = [token for token in target_tokens if token not in self.student.tokenizer.get_vocab()]
        if new_target_tokens:
            self.student.tokenizer.add_tokens(new_target_tokens)
            self.student[0].auto_model.resize_token_embeddings(len(self.student.tokenizer))
            self.optimizer.state = defaultdict(dict)

        self.optimizer.zero_grad()

        if "padded" in self.distribution:
            source_tokens, target_tokens = pad_sentences(source_tokens, target_tokens, self.teacher.tokenizer.pad_token, self.student.tokenizer.pad_token)

        if self.distribution == "tf-idf":
            source_sentence_list = []
            target_sentence_list = []
            with open(self.parallel_dir, newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    source_sentence_list.append(row['source'])
                    target_sentence_list.append(row['target'])
            source_dist = tf_idf_dist(source_tokens, source_sentence, source_sentence_list, self.device)
            target_dist = tf_idf_dist(target_tokens, target_sentence, target_sentence_list, self.device)
        elif self.distribution == "roberta":
            source_dist = roberta_dist(source_tokens, self.teacher_model_language, self.device)
            target_dist = roberta_dist(target_tokens, self.student_model_language, self.device)
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

        for i, token in enumerate(source_tokens):
            temp = plan[i].tolist()
            largest_values = heapq.nlargest(1, temp)
            mapped_indices = [temp.index(value) for value in largest_values]
            mapped_tokens = [target_tokens[j] for j in mapped_indices]
            print(token, mapped_tokens, largest_values)

        return plan, loss

    def train_loop(self, source_sentence: str, target_sentence: str):
        print(source_sentence)   
        # self.optimizer.zero_grad() 
        plan, loss = self.optical(source_sentence, target_sentence)
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
            df = read_csv(self.bitext_data)
            bitext_data = list(zip(df["source"], df["target"]))
            for source_sentence, target_sentence in bitext_data:
                self.train_loop(source_sentence, target_sentence)

        self.student.save(self.save_dir + f"sentence_transformer_multilingual_" + self.distribution)
        check_point = {
            'student_sentence_transformer_save_path': self.save_dir + f"sentence_transformer_multilingual_" + self.distribution

        }
        torch.save(check_point, self.save_dir + f"sentence_transformer_multilingual_"  + self.distribution + '/model.pth')

        return self.save_dir + f"sentence_transformer_multilingual_" + self.distribution