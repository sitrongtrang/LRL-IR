from functools import reduce
from numpy import argsort, float64
from rank_bm25 import BM25Plus

from dataset import DocumentDataset


class LexicalMatching:
    def __init__(
            self,
            document_dataset: DocumentDataset,
            training: bool = True,
            number_to_choose: int = 30
    ) -> None:
        """
        Args:
            document_dataset (DocumentDataset): An object of DocumentDataset class, represent a document dataset that query wants to retrive.
            training (bool): Indicate if we are in training phase or not.\
                If `True`, which mean we are in traning phase, there will be `Positive and Negative Pairs` function run after this\
                to check labels and generate +/- samples. So when we call `get_documents_ranking` we just need to return all documents\
                with their lexical matching score, order by matching point with descending order.\
                Otherwise, which mean we are in production mode, there will NOT be any function to check labels or generate samples.\
                So we have to limit the number of samples for next steps by ourself. When we call `get_documents_ranking`,\
                only `number_to_choose` documents with their lexical matching score are returned, order by matching point with descending order.
            number_to_choose (int): The number of samples we want to choose for next steps. Only work when `training` is set to `False`.
        """
        self.document_dataset: DocumentDataset = document_dataset
        self.training = training
        self.number_to_choose = number_to_choose
        self.bm25plus, self.file_path_lst = self._load_bm25plus()
    
    def _load_bm25plus(self) -> tuple[BM25Plus, list[str]]:
        """
        Load BM25Plus instance initialized with documents from the document dataset, and list of file path of each documents.

        Returns:
            tuple[BM25Plus, list[str]]: BM25Plus instance initialized with documents from the document dataset, and list of file path of each documents
        """
        tokenize_title_lst, tokenize_content_lst, file_path_lst = self._tokenize_document_and_file_path()
        tokenized_document_corpus: list[list[str]] = []
        for i in range(len(tokenize_title_lst)):
            tokenize_title = tokenize_title_lst[i]
            tokenize_content = tokenize_content_lst[i]
            tokenize_document = tokenize_title + tokenize_content
            tokenized_document_corpus.append(tokenize_document)
        bm25plus = BM25Plus(tokenized_document_corpus)
        return bm25plus, file_path_lst

    def _tokenize_document_and_file_path(self) -> tuple[list[list[str]], list[list[str]], list[str]]:
        """
        Tokenize the title, content and take the file path of each document in the dataset

        Returns:
            tuple[list[list[str]], list[list[str]], list[str]]: Tuple contains list of tokenized titles,\
            list of tokenized contents, and list of file paths for each document.\
            The index is correspond to each other, which mean tokenize_title_lst[i]\
            is the title of the document whose content is tokenize_content_lst[i].\
            Similar to file path as well.
        """
        tokenize_title_lst: list[list[str]] = []
        tokenize_content_lst: list[list[str]] = []
        file_path_lst: list[str] = []
        for document in self.document_dataset:
            title, content, topic, file_path = document
            tokenize_title: list[str] = reduce(
                lambda prev, curr: prev + self.document_dataset.tokenizer(curr), title, [])
            tokenize_content: list[str] = reduce(
                lambda prev, curr: prev + self.document_dataset.tokenizer(curr), content, [])
            tokenize_title_lst.append(tokenize_title)
            tokenize_content_lst.append(tokenize_content)
            file_path_lst.append(file_path)
        return tokenize_title_lst, tokenize_content_lst, file_path_lst

    def get_documents_ranking(self, tokenized_query: list[str]) -> list[tuple[str, float]]:
        """
        Get the list of lexical matching score between query and each document in dataset.

        Returns:
            list[tuple[str, float]]: List contains pairs of document's file path and its matching score to the query.\
            This list is sorted in descending order of matching score.\
            Depending on the mode is production or training, there will be a limit to the number of pairs returned or not.
        """
        matching_scores = self.bm25plus.get_scores(tokenized_query)
        top_relevant_index_set = argsort(matching_scores)[::-1]
        file_path_relevant_score_pairs: list[tuple[str, float]] = []
        for index in top_relevant_index_set:
            relevant_score_raw: float64 = matching_scores[index]
            relevant_score: float = relevant_score_raw.item()
            file_path: str = self.file_path_lst[index]
            pair: tuple[str, float] = (file_path, relevant_score)
            file_path_relevant_score_pairs.append(pair)
        if self.training:
            return file_path_relevant_score_pairs
        else:
            return file_path_relevant_score_pairs[:self.number_to_choose]
