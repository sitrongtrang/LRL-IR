import torch
import numpy as np
import torch.nn as nn
from sentence_transformer import SentenceTransformer
from optimal_transport_solver import OTSolver
from dataset import *
from scipy.spatial.distance import cdist
from torch.optim import Adam
from torch.nn import functional as F


class KnowledgeDistillation:
    def __init__(self, bitext_data: ParallelDataset, teacher_model, student_model, optimizer=None, epsilon=1e-3):
        self.bitext_data = bitext_data
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.optimizer = optimizer if optimizer else Adam(student_model.parameters(), lr=1e-5)
        self.ot_solver = OTSolver(epsilon=epsilon)

    def compute_cost_matrix(self, teacher_embeddings, student_embeddings):
        """
        Compute the cost matrix between teacher and student token embeddings using cosine distance.
        
        Parameters:
        - teacher_emb: Embeddings from the teacher model (B, L_t, H)
        - student_emb: Embeddings from the student model (B, L_s, H)
        
        Returns:
        - cost_matrix: Cosine distance cost matrix (L_t, L_s)
        """
        teacher_embeddings = teacher_embeddings.squeeze(0).detach().cpu().numpy()  # (L_t, H)
        student_embeddings = student_embeddings.squeeze(0).detach().cpu().numpy()  # (L_s, H)

        # Cosine distance between each token in the teacher and student embeddings
        cost_matrix = cdist(teacher_embeddings, student_embeddings, metric='cosine')
        return cost_matrix

    def compute_ot_loss(self, teacher_sentence, student_sentence):
        """
        Compute the optimal transport loss for knowledge distillation.

        Parameters:
        - teacher_sentence: Input sentence for teacher model.
        - student_sentence: Input sentence for student model.
        
        Returns:
        - loss: Optimal transport loss for training the student.
        """
        # Get embeddings for the teacher and student
        teacher_embeddings = self.teacher_model.get_embeddings(teacher_sentence)
        student_embeddings = self.student_model.get_embeddings(student_sentence)

        # Create uniform distributions for each token's probability (sum to 1)
        teacher_length = teacher_embeddings.shape[1]
        student_length = student_embeddings.shape[1]
        
        mu = np.ones(teacher_length) / teacher_length  # Uniform distribution
        nu = np.ones(student_length) / student_length  # Uniform distribution

        # Compute the cost matrix based on cosine distance
        cost_matrix = self.compute_cost_matrix(teacher_embeddings, student_embeddings)

        # Solve the optimal transport problem
        transport_plan, transport_cost = self.ot_solver.solve(mu, nu, cost_matrix, "ipot")

        # The transport cost is the loss we want to minimize
        return transport_cost

    def train_step(self, teacher_sentence, student_sentence):
        """
        Perform a single training step for knowledge distillation.
        
        Parameters:
        - teacher_sentence: Input sentence for teacher model.
        - student_sentence: Input sentence for student model.
        
        Returns:
        - loss: The distillation loss for the current step.
        """
        self.optimizer.zero_grad()
        
        # Compute the optimal transport loss
        loss = self.compute_ot_loss(teacher_sentence, student_sentence)
        
        # Backpropagate the loss to update the student model
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self):
        data = self.bitext_data._load_pairs()
        for pair in data:
            teacher_sentence, student_sentence = pair
            self.train_step(teacher_sentence, student_sentence)