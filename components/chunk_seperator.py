from functools import reduce
import random
import torch

from dataset import DocumentDataset


class ChunkSeperator:
    def __init__(
            self,
            document_dataset: DocumentDataset,
            chunk_length_limit: int = 128,
            random_length: bool = False
    ) -> None:
        """
        Args:
            document_dataset (str): An object of DocumentDataset class, represent a document dataset\
            that we want to get documents and seperate them into chunks
            chunk_length_limit (int): The limit length of each chunk
            random_length (bool): Indicate if we want the chunks to have variable length.\
                - If `True`: Each chunk will have the random length between `(ceil(chunk_length_limit / 2), chunk_length_limit)`.\
                This randomness means if we call method `get_chunks_of_document` several times, we will get different chunk list\
                for each times due to the variable length of each chunk. So more patterns for the chunk list can be generated\
                from a single document, which may help us to extract more characteristics of different chunk combinations.\
                With each chunk, sentence transformer can capture the relation of query and that chunk, so the more\
                chunk combinations, the more semantic relationship that sentence transformer may be able to capture.\
                - If `False`: Each chunk will have fixed length of `chunk_length_limit` element. Currently, we use `False` because\
                we not sure if this variable really work as we expected. Experiment will be conducted in the near future to test this out.
                
        """
        self.document_dataset: DocumentDataset = document_dataset
        self.chunk_length_limit = max(1, chunk_length_limit)
        self.random_length = random_length

    def _generate_sequence_of_ids_lst(self, title_segmented: list[str], content_segmented: list[str]) -> list[int]:
        """
        Generate the sequence of ids of the whole document from their segmented title and content.

        Args:
            title_segmented (list[str]): Segmented title of the document

            content_segmented (list[str]): Segmented content of the document

        Returns:
            list[int]: Sequence of ids of the whole document
        """
        tokenize_title: list[str] = reduce(
            lambda prev, curr: prev + self.document_dataset.tokenizer(curr), title_segmented, [])
        tokenize_content: list[str] = reduce(
            lambda prev, curr: prev + self.document_dataset.tokenizer(curr), content_segmented, [])
        title_sequence_of_ids: list[int] = self.document_dataset.encoder(
            tokenize_title)
        content_sequence_of_ids: list[int] = self.document_dataset.encoder(
            tokenize_content)
        return title_sequence_of_ids + content_sequence_of_ids

    def get_chunks_of_document(self, file_path: str) -> list[list[int]]:
        """
        Get the chunk list of the document.

        Args:
            file_path (str): Path to the document we want to and seperate them into chunks

        Returns:
            list[list[int]]: Chunk list of the document. Each chunk is a sequence of ids\
            represent a sequence of words in the language.
        """
        title_segmented, content_segmented = self.document_dataset.find_document_by_file_path(
            file_path)
        document_sequence_of_ids: list[int] = self._generate_sequence_of_ids_lst(
            title_segmented, content_segmented)
        document_chunk_lst: list[list[int]] = []
        begin: int = 0
        while begin < len(document_sequence_of_ids):
            end: int = random.randint(self.chunk_length_limit // 2 + self.chunk_length_limit %
                                 2, self.chunk_length_limit) if self.random_length else self.chunk_length_limit
            new_chunk = document_sequence_of_ids[begin: begin + end]
            document_chunk_lst.append(new_chunk)
            begin += end
        return document_chunk_lst
