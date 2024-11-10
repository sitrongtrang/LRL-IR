import torch
from torch import nn
from sentence_transformers import SentenceTransformer, InputExample, util
from torch.utils.data import DataLoader

class CustomSentenceTransformer(nn.Module):
    """
    This class is NOT finish. Please don't use.
    """
    def __init__(self, pretrained_model_name_or_path: str, hidden_dim: int = 128):
        super(CustomSentenceTransformer, self).__init__()

        self.sentence_transformer: SentenceTransformer = SentenceTransformer(pretrained_model_name_or_path)
        
        self.hidden_layer = nn.Linear(self.sentence_transformer.get_sentence_embedding_dimension(), hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
    
    def forward(self, query: str, sentence_stream: list[str]):
        total_similarity = 0.0
        
        for sentence_pair in sentence_stream:
            embedding1 = self.sentence_transformer.encode(sentence_pair[0], convert_to_tensor=True)
            embedding2 = self.sentence_transformer.encode(sentence_pair[1], convert_to_tensor=True)
            
            similarity_score = util.pytorch_cos_sim(embedding1, embedding2)
            
            total_similarity += similarity_score.squeeze()  # Remove extra dimensions if needed

        # Pass the summed similarity through hidden layers
        hidden_output = self.relu(self.hidden_layer(total_similarity))
        output = self.output_layer(hidden_output)
        
        return output

