import torch
import torch.nn as nn
import numpy

class QRANN(nn.Module):
    """
    Args:
        query(matrix): the query's embedding
        sentence(matrix): the sentence's embedding
    """

    def __init__(self, query, sentence):
        super(QRANN, self).__init__()
        
