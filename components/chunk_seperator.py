from functools import reduce
import random
import torch

from dataset import DocumentDataset


class ChunkSeperator:
    def __init__(
            self,
            document_dataset: DocumentDataset,
            chunk_length_limit: int = 128,
    ) -> None:
        """
        Args:
            document_dataset (str): An object of DocumentDataset class, represent a document dataset\
            that we want to get documents and seperate them into chunks
            chunk_length_limit (int): The limit length of each chunk                
        """
        self.document_dataset: DocumentDataset = document_dataset
        self.chunk_length_limit: int = max(1, chunk_length_limit)

    def get_chunks_of_document(self, file_path: str) -> list[str]:
        """
        Get the chunk list of the document.

        Args:
            file_path (str): Path to the document we want to and seperate them into chunks

        Returns:
            list[str]: Chunk list of the document. Each chunk consist one or more sentences that total number of tokens\
            in each chunk is less than the `chunk_length_limit`.
        """
        title_segmented, content_segmented = self.document_dataset.find_document_by_file_path(
            file_path)
        document_segmented: list[str] = title_segmented + content_segmented
        document_chunk_lst: list[str] = []
        current_len: int = 0
        current_sentence_lst: list[str] = []
        for segmented_sentence in document_segmented:
            token_list_of_sentence: list[str] = self.document_dataset.tokenizer(segmented_sentence)
            if len(token_list_of_sentence) + current_len >= self.chunk_length_limit:
                document_chunk_lst.append(self.document_dataset.chunk_combiner(current_sentence_lst))
                current_len = len(token_list_of_sentence)
                current_sentence_lst = []
                current_sentence_lst.append(segmented_sentence)
            else:
                current_sentence_lst.append(segmented_sentence)
                current_len += len(token_list_of_sentence)
        if len(current_sentence_lst) > 0:
            document_chunk_lst.append(self.document_dataset.chunk_combiner(current_sentence_lst))
        return document_chunk_lst

