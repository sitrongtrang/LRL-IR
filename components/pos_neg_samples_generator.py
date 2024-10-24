class PositiveNegativeSamplesGenerator:
    def __init__(
            self,
            document_file_path_answers: list[str],
            file_path_relevant_score_pairs: list[tuple[str, float]],
            negative_samples_limit: int = 35
    ) -> None:
        """
        Args:
            document_file_path_answers (list[str]): List of file paths of correct documents should be retrived for a query.\
            file_path_relevant_score_pairs (list[tuple[str, float]]): List of lexical matching score between query and\
            each document in dataset.
            negative_samples_limit (int): Number of negative samples you want to add to the sample set when returned.
        """
        self.document_file_path_answers: list[str] = document_file_path_answers
        self.file_path_relevant_score_pairs: list[tuple[str,
                                                        float]] = file_path_relevant_score_pairs
        self.negative_samples_limit: int = max(0, negative_samples_limit)

    def generate_samples(self) -> list[tuple[str, float]]:
        """
        Generate the sample set with positive and negative samples.

        Returns:
            list[tuple[str, float]]: List of sample docs, each sample is a tuple contains file path to a document\
            and its lexical matching score.
        """
        samples: list[tuple[str, float]] = []
        negative_samples_count: int = 0
        for file_path, relevant_score in self.file_path_relevant_score_pairs:
            if file_path in self.document_file_path_answers:
                samples.append((file_path, relevant_score))
            elif negative_samples_count < self.negative_samples_limit:
                samples.append((file_path, relevant_score))
                negative_samples_count += 1
        return samples
