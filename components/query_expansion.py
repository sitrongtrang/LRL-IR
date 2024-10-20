from numpy import float64, argsort
from rank_bm25 import BM25Plus
from typing import TypeAlias, Literal
from dataset import DocumentDataset
from functools import reduce
from math import log, inf

SourceForExpansion: TypeAlias = Literal[
    'COLLECTION_SET_TITLE',
    'COLLECTION_SET_CONTENT',
    'RELEVANT_SET_TITLE',
    'RELEVANT_SET_CONTENT'
]

# Too high will affect the speed, especially if the document is long
# The paper uses this number in experiment
LIMIT_K_DOCS_FOR_RELEVANT_SET = 10

# The paper also uses this number in experiment
NUMBER_OF_EXPANSION_TERM = 30

THRESHOLD_FOR_EM_ALGO = 0.0000001


class QueryExpansion:
    def __init__(self, document_dataset: DocumentDataset) -> None:
        """
        Args:
            document_dataset: An object of DocumentDataset class, represent a document dataset use to expand the query
        """
        self.document_dataset: DocumentDataset = document_dataset
        self.sources: dict[SourceForExpansion, list[list[str]]] = {
            'COLLECTION_SET_TITLE': [],
            'COLLECTION_SET_CONTENT': [],
            'RELEVANT_SET_TITLE': [],
            'RELEVANT_SET_CONTENT': []
        }
        self.sources['COLLECTION_SET_TITLE'], self.sources['COLLECTION_SET_CONTENT'] = self._tokenize_document()
        self.collection_set: set[str] = self._get_collection_set()

        # At first, this dict is empty, so we need to use .get() method with default value when retrieving prob,
        # so that if key does not exist yet, default value will be return.
        # Default value (currently 0.5) in this case will also be the initilize prob for all pairs in the model.
        self.prob_expansion_term_represents_source: dict[tuple[str, SourceForExpansion], float] = {
        }

        self.prob_of_selecting_source: dict[SourceForExpansion, float] = {
            # Initialize all 4 probs equally. Maximization step will update these probs later.
            'COLLECTION_SET_TITLE': 0.25,
            'COLLECTION_SET_CONTENT': 0.25,
            'RELEVANT_SET_TITLE': 0.25,
            'RELEVANT_SET_CONTENT': 0.25
        }

        # Like above, this dict is also empty and required to be treated similarly
        self.prob_term_belongs_to_source: dict[tuple[str, SourceForExpansion], float] = {
        }

    def _tokenize_document(self) -> tuple[list[list[str]], list[list[str]]]:
        """
        Tokenize the title and content of each document in the dataset

        Returns:
            tuple[list[list[str]], list[list[str]]]: Tuple contains list of tokenized title and list of tokenized content for each document.
            The index is correspond to each other, which mean tokenize_title_lst[i] is the title of the document whose content is tokenize_content_lst[i]
        """
        tokenize_title_lst: list[list[str]] = []
        tokenize_content_lst: list[list[str]] = []
        for document in self.document_dataset:
            title, content, topic = document
            tokenize_title: list[str] = reduce(
                lambda prev, curr: prev + self.document_dataset.tokenizer(curr), title, [])
            tokenize_content: list[str] = reduce(
                lambda prev, curr: prev + self.document_dataset.tokenizer(curr), content, [])
            tokenize_title_lst.append(tokenize_title)
            tokenize_content_lst.append(tokenize_content)
        return tokenize_title_lst, tokenize_content_lst

    def _get_collection_set(self) -> set[str]:
        """
        Get all terms from the collection set. This will get term set from two collection set (title and content),
        then combine to from the final term set.

        Returns:
            set[str]: The set contain all terms from collection set.
        """
        collection_set: set[str] = self._get_term_set_of_source(
            "COLLECTION_SET_TITLE")
        collection_set.update(
            self._get_term_set_of_source("COLLECTION_SET_CONTENT"))
        return collection_set

    def _get_term_set_of_source(self, source: SourceForExpansion) -> set[str]:
        """
        Get all terms from the specify source.

        Args:
            source (SourceForExpansion): Name of the source that needs to get terms

        Returns:
            set[str]: The set contain all terms extract from all document in the specify source.
        """
        term_set: set[str] = set()
        for sequence in self.sources[source]:
            for term in sequence:
                term_set.add(term)
        return term_set

    def get_expansion_term(self, tokenized_query: list[str]) -> list[str]:
        """
        Get the list of tokenized expansion terms for the input query.
        This is also the only public method for this class, the only method you need.

        Args:
            tokenized_query (list[str]): Query which has been tokenized with the same tokenizer used for the documents

        Returns:
            list[str]: The list of tokenized expansion terms
        """
        self._retrive(tokenized_query)
        return self._expand()

    def _retrive(self, tokenized_query: list[str]) -> None:
        """
        Get the list of the top highest lexical similarity documents for a query and add these documents to relevant set

        Args:
            tokenized_query (list[str]): Query which has been tokenized with the same tokenizer used for the documents
        """
        tokenized_document_corpus: list[list[str]] = []
        for i in range(len(self.sources['COLLECTION_SET_TITLE'])):
            tokenize_title: list[str] = self.sources['COLLECTION_SET_TITLE'][i]
            tokenize_content: list[str] = self.sources['COLLECTION_SET_CONTENT'][i]
            tokenize_document: list[str] = tokenize_title + tokenize_content
            tokenized_document_corpus.append(tokenize_document)
        bm25plus = BM25Plus(tokenized_document_corpus)
        matching_scores = bm25plus.get_scores(tokenized_query)
        top_relevant_index_set = argsort(matching_scores)[
            ::-1][:LIMIT_K_DOCS_FOR_RELEVANT_SET]

        for index in top_relevant_index_set:
            self.sources['RELEVANT_SET_TITLE'].append(
                self.sources['COLLECTION_SET_TITLE'][index])
            self.sources['RELEVANT_SET_CONTENT'].append(
                self.sources['COLLECTION_SET_CONTENT'][index])

    def _expand(self) -> list[str]:
        """
        Get the list of tokenized expansion terms by performing EM algorithm for each observation sequence:
        relevant set (title) and relevant set (content), then combining their expansion term list to form final list.

        Returns:
            list[str]: Final list of tokenized expansion terms
        """
        previous_likelihood_title: float = -inf
        previous_likelihood_content: float = -inf
        observation_sequence_title: set[str] = self._get_term_set_of_source(
            "RELEVANT_SET_TITLE")
        observation_sequence_content: set[str] = self._get_term_set_of_source(
            "RELEVANT_SET_CONTENT")

        while self._log_likelihood(observation_sequence_title) <= (previous_likelihood_title + THRESHOLD_FOR_EM_ALGO):
            previous_likelihood_title = self._log_likelihood(
                observation_sequence_title)
            self._estimation_step(observation_sequence_title)
            self._maximization_step(observation_sequence_title)

        title_expansion_prob_dict: dict[tuple[str, SourceForExpansion], float] = {
            k: v for k, v in self.prob_expansion_term_represents_source.items() if k[1] == "RELEVANT_SET_TITLE"}
        sorted_title_expansion_prob: list[tuple[tuple[str, SourceForExpansion], float]] = sorted(
            title_expansion_prob_dict.items(), key=lambda item: item[1], reverse=True)
        expansion_term_with_prob_from_title_relevant_set: list[tuple[str, float]] = [
            (term[0][0], term[1]) for term in sorted_title_expansion_prob[:NUMBER_OF_EXPANSION_TERM]]

        while self._log_likelihood(observation_sequence_content) <= (previous_likelihood_content + THRESHOLD_FOR_EM_ALGO):
            previous_likelihood_content = self._log_likelihood(
                observation_sequence_content)
            self._estimation_step(observation_sequence_content)
            self._maximization_step(observation_sequence_content)

        content_expansion_prob_dict: dict[tuple[str, SourceForExpansion], float] = {
            k: v for k, v in self.prob_expansion_term_represents_source.items() if k[1] == "RELEVANT_SET_CONTENT"}
        sorted_content_expansion_prob: list[tuple[tuple[str, SourceForExpansion], float]] = sorted(
            content_expansion_prob_dict.items(), key=lambda item: item[1], reverse=True)
        expansion_term_with_prob_from_content_relevant_set: list[tuple[str, float]] = [
            (term[0][0], term[1]) for term in sorted_content_expansion_prob[:NUMBER_OF_EXPANSION_TERM]]

        sorted_expansion_term_final_prob = sorted(
            expansion_term_with_prob_from_title_relevant_set +
            expansion_term_with_prob_from_content_relevant_set,
            key=lambda item: item[1],
            reverse=True)
        expansion_term_final: list[str] = [
            term[0] for term in sorted_expansion_term_final_prob[:NUMBER_OF_EXPANSION_TERM]]
        return expansion_term_final

    def _log_likelihood(self, observation_sequence: set[str]) -> float:
        """
        Calculate the log likelihood value for the observation sequence.

        Args:
            observation_sequence (set[str]): The term set of the source that is considered to be observation sequence. 
            For ease of understanding, this term set is all the words from relevant set (title) or relevant set (content),
            depends on which relevant set you are calculating.

        Returns:
            float: The log likelihood value calculated
        """
        likelihood: float = 0.0
        for term in observation_sequence:
            accumulate_likelihood_of_source: float = 0.0
            for source in self.sources.keys():
                term_source_pair: tuple[str,
                                        SourceForExpansion] = (term, source)
                prob_term_belongs_to_source: float = self.prob_term_belongs_to_source.get(
                    term_source_pair, 0.5)
                prob_of_selecting_source: float = self.prob_of_selecting_source[source]
                accumulate_likelihood_of_collection_set: float = 0.0
                for expansion_term in self.collection_set:
                    indicator = 1 if term == expansion_term else 0
                    expansion_term_source_pair: tuple[str, SourceForExpansion] = (
                        expansion_term, source)
                    prob_expansion_term_represents_source: float = self.prob_expansion_term_represents_source.get(
                        expansion_term_source_pair, 0.5)
                    accumulate_likelihood_of_collection_set += indicator * \
                        log(prob_expansion_term_represents_source)
                accumulate_likelihood_of_source += prob_term_belongs_to_source * \
                    (prob_of_selecting_source +
                     accumulate_likelihood_of_collection_set)
            likelihood += accumulate_likelihood_of_source
        return likelihood

    def _maximization_step(self, observation_sequence: set[str]) -> None:
        """
        Maximization step in EM algorithm. This will calculate new probability of selecting a source and
        probability an expansion term represents a source, and then save them.

        Args:
            observation_sequence (set[str]): The term set of the source that is considered to be observation sequence. 
            For ease of understanding, this term set is all the words from relevant set (title) or relevant set (content),
            depends on which relevant set you are calculating.
        """
        updated_prob_of_selecting_source: dict[SourceForExpansion, float] = {}
        updated_prob_expansion_term_represents_source: dict[tuple[str, SourceForExpansion], float] = {
        }
        for source_to_maximize in self.prob_of_selecting_source.keys():
            updated_prob_of_selecting_source[source_to_maximize] = self._maximize_prob_of_selecting_source(
                source_to_maximize, observation_sequence)
            for expansion_term_to_maximize in self.collection_set:
                key_pair = (expansion_term_to_maximize, source_to_maximize)
                updated_prob_expansion_term_represents_source[key_pair] = self._maximize_prob_expansion_term_represents_source(
                    source_to_maximize, expansion_term_to_maximize, observation_sequence)
        self.prob_of_selecting_source = updated_prob_of_selecting_source
        self.prob_expansion_term_represents_source = updated_prob_expansion_term_represents_source

    def _maximize_prob_of_selecting_source(
            self,
            source_to_maximize: SourceForExpansion,
            observation_sequence: set[str]
    ) -> float:
        """
        Calculate the new probability of selecting a source.

        Args:
            source_to_maximize (SourceForExpansion): Name of the source you want to calculate new probability

            observation_sequence (set[str]): The term set of the source that is considered to be observation sequence. 
            For ease of understanding, this term set is all the words from relevant set (title) or relevant set (content),
            depends on which relevant set you are calculating.

        Returns:
            float: The new probability calculated.
        """
        numerator: float = 0.0
        denominator: float = 0.0

        for term in observation_sequence:
            term_source_to_maximize_pair = (term, source_to_maximize)
            numerator += self.prob_term_belongs_to_source.get(
                term_source_to_maximize_pair, 0.5)

            relation_prob_term_all_source: float = 0.0
            for source in self.sources.keys():
                term_source_pair: tuple[str,
                                        SourceForExpansion] = (term, source)
                relation_prob_term_all_source += self.prob_term_belongs_to_source.get(
                    term_source_pair, 0.5)
            denominator += relation_prob_term_all_source

        return numerator / denominator

    def _maximize_prob_expansion_term_represents_source(
            self,
            source_to_maximize: SourceForExpansion,
            expansion_term_to_maximize: str,
            observation_sequence: set[str]
    ) -> float:
        """
        Calculate the new probability an expansion term represents a source.

        Args:
            source_to_maximize (SourceForExpansion): Name of the source you want to calculate new probability

            expansion_term_to_maximize (str): The term you want to calculate new probability

            observation_sequence (set[str]): The term set of the source that is considered to be observation sequence. 
            For ease of understanding, this term set is all the words from relevant set (title) or relevant set (content),
            depends on which relevant set you are calculating.

        Returns:
            float: The new probability calculated.
        """
        numerator: float = 0.0
        denominator: float = 0.0

        for term in observation_sequence:
            indicator = 1 if term == expansion_term_to_maximize else 0
            expansion_term_to_maximize_source_to_maximize_pair = (
                expansion_term_to_maximize, source_to_maximize)
            numerator += indicator * \
                self.prob_term_belongs_to_source.get(
                    expansion_term_to_maximize_source_to_maximize_pair, 0.5)

            relation_prob_term_whole_collection_set: float = 0.0
            for expansion_term in self.collection_set:
                expansion_term_indicator = 1 if term == expansion_term else 0
                expansion_term_source_to_maximize_pair: tuple[str, SourceForExpansion] = (
                    expansion_term, source_to_maximize)
                relation_prob_term_whole_collection_set += expansion_term_indicator * \
                    self.prob_term_belongs_to_source.get(
                        expansion_term_source_to_maximize_pair, 0.5)
            denominator += relation_prob_term_whole_collection_set

        return numerator / denominator

    def _estimation_step(self, observation_sequence: set[str]) -> None:
        """
        Expectation step in EM algorithm. This will calculate new probability that a term belongs to a source, 
        and then save it

        Args:
            observation_sequence (set[str]): The term set of the source that is considered to be observation sequence. 
            For ease of understanding, this term set is all the words from relevant set (title) or relevant set (content),
            depends on which relevant set you are calculating.
        """
        updated_prob_term_belongs_to_source: dict[tuple[str, SourceForExpansion], float] = {
        }
        for term_to_estimate in observation_sequence:
            for source_to_estimate in self.prob_of_selecting_source.keys():
                key_pair: tuple[str, SourceForExpansion] = (
                    term_to_estimate, source_to_estimate)
                updated_prob_term_belongs_to_source[key_pair] = self._estimate_prob_term_belongs_to_source(
                    term_to_estimate, source_to_estimate)
        self.prob_term_belongs_to_source = updated_prob_term_belongs_to_source

    def _estimate_prob_term_belongs_to_source(
            self,
            term_to_estimate: str,
            source_to_estimate: SourceForExpansion
    ) -> float:
        """
        Calculate new probability that a term belongs to a source.

        Args:
            term_to_estimate (str): The term you want to calculate new probability

            source_to_estimate (SourceForExpansion): Name of the source you want to calculate new probability

        Returns:
            float: The new probability calculated.
        """
        numerator: float = 0.0
        denominator: float = 0.0

        accumulate_prob_for_numerator: float = 1.0
        for expansion_term in self.collection_set:
            indicator = 1 if term_to_estimate == expansion_term else 0
            expansion_term_source_to_estimate_pair: tuple[str, SourceForExpansion] = (
                expansion_term, source_to_estimate)
            accumulate_prob_for_numerator *= (self.prob_expansion_term_represents_source.get(
                expansion_term_source_to_estimate_pair, 0.5)) ** indicator
        numerator = self.prob_of_selecting_source[source_to_estimate] * \
            accumulate_prob_for_numerator

        for source in self.prob_of_selecting_source.keys():
            accumulate_prob_for_denominator: float = 1.0
            for expansion_term in self.collection_set:
                indicator = 1 if term_to_estimate == expansion_term else 0
                expansion_term_source_pair: tuple[str, SourceForExpansion] = (
                    expansion_term, source)
                accumulate_prob_for_denominator *= (self.prob_expansion_term_represents_source.get(
                    expansion_term_source_pair, 0.5)) ** indicator
            denominator += self.prob_of_selecting_source[source] * \
                accumulate_prob_for_denominator

        return numerator / denominator
