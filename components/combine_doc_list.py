class CombineDocList:
    def __init__(
            self,
            doc_list_original_query: list[tuple[str, float]],
            doc_list_extended_query: list[tuple[str, float]]
    ) -> None:
        """
        Args:
            doc_list_original_query (list[tuple[str, float]]): List of tuple contains file path to a document\
            and its lexical matching score to the original query.
            doc_list_extended_query (list[tuple[str, float]]): List of tuple contains file path to a document\
            and its lexical matching score to the extended query.
        """
        self.doc_list_original_query: list[tuple[str,
                                                 float]] = doc_list_original_query
        self.doc_list_extended_query: list[tuple[str,
                                                 float]] = doc_list_extended_query

    def get_combine_doc_list(self) -> list[tuple[str, float]]:
        """
        Returns:
            list[tuple[str, float]]: Combined list of tuple contains file path to a document\
            and its lexical matching score.
        """
        combined_doc_dict: dict[str, float] = {}
        for file_path, relevant_score in self.doc_list_original_query:
            combined_doc_dict[file_path] = relevant_score
        for file_path, relevant_score in self.doc_list_extended_query:
            if file_path not in combined_doc_dict or combined_doc_dict[file_path] < relevant_score:
                combined_doc_dict[file_path] = relevant_score
        combined_doc_list: list[tuple[str, float]] = [
            (file_path, relevant_score) for file_path, relevant_score in combined_doc_dict.items()]
        return combined_doc_list