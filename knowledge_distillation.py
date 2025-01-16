import torch
from torch import nn, Tensor
from sentence_transformers import SentenceTransformer
from components.ot_solver import OTSolver
from components.dataset import *

class OTLoss(nn.Module):
    """
    This class represents the Optimal Transport transportation cost and is used to calcualate the loss value.
    """
    def __init__(self):
        super(OTLoss, self).__init__()

    def forward(self, source: Tensor, target: Tensor, plan: Tensor) -> Tensor:
        """
        source (Tensor): Tensor of token embeddings of the source sentence.
        target (Tensor): Tensor of token embeddings of the target sentence.
        plan (Tensor): The transportation plan.
        """
        loss: Tensor = torch.sum(plan * torch.cdist(source, target))
        return loss

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
        self.ot_loss: OTLoss = OTLoss().to(device)

    def train(self) -> None:
        """
        Train the student model using knowledge distillation.
        """
        for epoch in range(self.epochs):
            for source_sentence, target_sentence in self.bitext_data:     
                self.optimizer.zero_grad()      
                with torch.no_grad():
                    teacher_embeddings: Tensor = self.teacher(
                        source_sentence,
                        convert_to_tensor=True,
                    )
                
                student_embeddings_source: Tensor = self.student(
                    source_sentence,
                    convert_to_tensor=True,
                )

                student_embeddings_target: Tensor = self.student(
                    target_sentence,
                    convert_to_tensor=True,
                )
                
                # Compute optimal transport plans
                cost_source = torch.cdist(teacher_embeddings, student_embeddings_source)
                transport_plan_source: Tensor = self.ot_solver.solve(teacher_embeddings, student_embeddings_source, cost_source)

                cost_target = torch.cdist(teacher_embeddings, student_embeddings_target)
                transport_plan_target: Tensor = self.ot_solver.solve(teacher_embeddings, student_embeddings_target, cost_target)
                
                loss = self.ot_loss(teacher_embeddings, student_embeddings_source, transport_plan_source) \
                    + self.ot_loss(teacher_embeddings, student_embeddings_target, transport_plan_target)
                
                loss.backward()
                self.optimizer.step()