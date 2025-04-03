import concurrent.futures
import numpy as np
from numpy import float64, argsort
from rank_bm25 import BM25Plus
from typing import TypeAlias, Literal, Dict, List, Set, Tuple
from functools import lru_cache
from math import log, inf
import threading

SourceForExpansion: TypeAlias = Literal[
    'COLLECTION_SET_TITLE',
    'COLLECTION_SET_CONTENT',
    'RELEVANT_SET_TITLE',
    'RELEVANT_SET_CONTENT'
]

class OptimizedQueryExpansion:
    LIMIT_K_DOCS_FOR_RELEVANT_SET = 10
    NUMBER_OF_EXPANSION_TERM = 30
    THRESHOLD_FOR_EM_ALGO = 0.000001
    
    def __init__(self, document_dataset) -> None:
        """Initialize with the same parameters as the original class"""
        self.document_dataset = document_dataset
        self.sources = {
            'COLLECTION_SET_TITLE': [],
            'COLLECTION_SET_CONTENT': [],
            'RELEVANT_SET_TITLE': [],
            'RELEVANT_SET_CONTENT': []
        }
        self.sources['COLLECTION_SET_TITLE'], self.sources['COLLECTION_SET_CONTENT'] = self._get_tokenize_document()
        
        # Perform collection set calculation once and store as a list for faster access
        self.collection_set = self._get_collection_set()
        self.collection_set_list = list(self.collection_set)
        self.collection_set_size = len(self.collection_set)
        
        # Create term to index mapping for faster lookups
        self.term_to_index = {term: idx for idx, term in enumerate(self.collection_set_list)}
        
        # Default probability value
        self.default_prob = 1.0 / float(self.collection_set_size)
        
        self.bm25plus = self._load_bm25plus()
        
        # Use dictionaries with better access patterns
        self.prob_expansion_term_represents_source = {}
        self.prob_of_selecting_source = {
            'COLLECTION_SET_TITLE': 0.25,
            'COLLECTION_SET_CONTENT': 0.25,
            'RELEVANT_SET_TITLE': 0.25,
            'RELEVANT_SET_CONTENT': 0.25
        }
        self.prob_term_belongs_to_source = {}
        
        # Cache for frequently accessed calculations
        self._term_source_cache = {}
        self._lock = threading.Lock()

    def _get_tokenize_document(self):
        """Keep original implementation"""
        tokenize_title_lst = []
        tokenize_content_lst = []
        for _, _, tokenize_title, tokenize_content, _, _, _ in self.document_dataset:
            tokenize_title_lst.append(tokenize_title)
            tokenize_content_lst.append(tokenize_content)
        return tokenize_title_lst, tokenize_content_lst

    def _get_collection_set(self) -> set:
        """Optimized to do only necessary operations"""
        collection_set = set()
        # Directly collect all terms from both sources
        for doc in self.sources['COLLECTION_SET_TITLE']:
            collection_set.update(doc)
        for doc in self.sources['COLLECTION_SET_CONTENT']:
            collection_set.update(doc)
        return collection_set

    def _load_bm25plus(self) -> BM25Plus:
        """Keep original implementation"""
        tokenized_document_corpus = []
        for i in range(len(self.sources['COLLECTION_SET_TITLE'])):
            tokenize_document = self.sources['COLLECTION_SET_TITLE'][i] + self.sources['COLLECTION_SET_CONTENT'][i]
            tokenized_document_corpus.append(tokenize_document)
        return BM25Plus(tokenized_document_corpus)

    def get_expansion_term(self, tokenized_query: list[str]) -> list[str]:
        """Main public method - same interface as original"""
        self._retrive(tokenized_query)
        return self._expand(tokenized_query)

    def _retrive(self, tokenized_query: list[str]) -> None:
        """Keep original implementation"""
        self._clear_previous_result()

        matching_scores = self.bm25plus.get_scores(tokenized_query)
        top_relevant_index_set = argsort(matching_scores)[::-1][:self.LIMIT_K_DOCS_FOR_RELEVANT_SET]

        for index in top_relevant_index_set:
            self.sources['RELEVANT_SET_TITLE'].append(
                self.sources['COLLECTION_SET_TITLE'][index])
            self.sources['RELEVANT_SET_CONTENT'].append(
                self.sources['COLLECTION_SET_CONTENT'][index])

    def _clear_previous_result(self) -> None:
        """Keep original implementation"""
        self.sources['RELEVANT_SET_TITLE'].clear()
        self.sources['RELEVANT_SET_CONTENT'].clear()
        self.prob_of_selecting_source = {
            'COLLECTION_SET_TITLE': 0.25,
            'COLLECTION_SET_CONTENT': 0.25,
            'RELEVANT_SET_TITLE': 0.25,
            'RELEVANT_SET_CONTENT': 0.25
        }
        self.prob_expansion_term_represents_source.clear()
        self.prob_term_belongs_to_source.clear()
        self._term_source_cache.clear()

    def _get_term_set_of_source(self, source: SourceForExpansion) -> set:
        """Optimized term set collection"""
        # Use caching for repeated calls
        if source in self._term_source_cache:
            return self._term_source_cache[source]
            
        term_set = set()
        for sequence in self.sources[source]:
            term_set.update(sequence)
            
        self._term_source_cache[source] = term_set
        return term_set

    def _log_likelihood(self, observation_sequence: set) -> float:
        """Optimized log likelihood calculation"""
        # Pre-calculate probabilities to avoid redundant lookups
        source_probs = {source: self.prob_of_selecting_source[source] for source in self.sources.keys()}
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for term in observation_sequence:
                futures.append(executor.submit(self._compute_term_likelihood, term, source_probs))
            
            likelihoods = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        return sum(likelihoods)
    
    def _compute_term_likelihood(self, term, source_probs):
        """Helper function to compute likelihood for a single term"""
        accumulate_likelihood = 0.0
        
        # Only check if term is in collection set once
        if term not in self.collection_set:
            return 0.0
            
        for source, prob_source in source_probs.items():
            term_source_pair = (term, source)
            prob_term_belongs = self.prob_term_belongs_to_source.get(term_source_pair, 0.25)
            
            # Only compute if probability is significant
            if prob_term_belongs > 0:
                # Only need to check for the current term, not the entire collection set
                expansion_term_source_pair = (term, source)
                prob_expansion = self.prob_expansion_term_represents_source.get(
                    expansion_term_source_pair, self.default_prob)
                
                if prob_expansion > 0:
                    log_prob = log(prob_expansion)
                    source_likelihood = prob_term_belongs * (prob_source + log_prob)
                    accumulate_likelihood += source_likelihood
        
        return accumulate_likelihood

    def _maximization_step(self, observation_sequence: set) -> None:
        """Optimized maximization step"""
        # Calculate source probabilities in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            source_futures = {
                executor.submit(self._maximize_prob_of_selecting_source, source, observation_sequence): source
                for source in self.prob_of_selecting_source.keys()
            }
            
            updated_prob_of_selecting_source = {}
            for future in concurrent.futures.as_completed(source_futures):
                source = source_futures[future]
                updated_prob_of_selecting_source[source] = future.result()
        
        # Update source probabilities
        self.prob_of_selecting_source = updated_prob_of_selecting_source
        
        # Precompute values needed for term-source probability calculation
        term_set = list(observation_sequence)
        sources = list(self.prob_of_selecting_source.keys())
        
        # Process in batches to reduce overhead
        batch_size = 100  # Adjust based on your system
        all_pairs = []
        
        for source in sources:
            for term in self.collection_set:
                all_pairs.append((term, source))
        
        # Process batches in parallel
        updated_prob_expansion_term_represents_source = {}
        
        for i in range(0, len(all_pairs), batch_size):
            batch = all_pairs[i:i+batch_size]
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                batch_results = executor.map(
                    lambda pair: (
                        pair,
                        self._maximize_prob_expansion_term_represents_source(pair[1], pair[0], observation_sequence)
                    ),
                    batch
                )
                
                for (term, source), prob in batch_results:
                    if prob > 0:  # Only store non-zero probabilities
                        updated_prob_expansion_term_represents_source[(term, source)] = prob
            
            if i % 1000 == 0 and i > 0:
                print(f"Processed {i}/{len(all_pairs)} term-source pairs")
        
        # Update term-source probabilities
        self.prob_expansion_term_represents_source = updated_prob_expansion_term_represents_source

    def _maximize_prob_of_selecting_source(self, source_to_maximize, observation_sequence):
        """Optimized source probability calculation"""
        numerator = 0.0
        denominator = 0.0
        
        # Group calculations for efficiency 
        for term in observation_sequence:
            term_source_pair = (term, source_to_maximize)
            prob = self.prob_term_belongs_to_source.get(term_source_pair, 0.25)
            numerator += prob
            
            # Calculate denominator efficiently
            term_total = 0.0
            for source in self.sources.keys():
                term_source = (term, source)
                term_total += self.prob_term_belongs_to_source.get(term_source, 0.25)
            
            denominator += term_total
        
        return numerator / max(denominator, 1e-10)  # Avoid division by zero

    def _maximize_prob_expansion_term_represents_source(self, source_to_maximize, expansion_term_to_maximize, observation_sequence):
        """Optimized expansion term probability calculation"""
        # Quick check - if term isn't in observation sequence, can be more efficient
        if expansion_term_to_maximize not in observation_sequence:
            # Check if we need this term at all
            # If not in top N terms by current probability, skip it
            current_prob = self.prob_expansion_term_represents_source.get(
                (expansion_term_to_maximize, source_to_maximize), 0.0)
            if current_prob == 0.0:
                return 0.0
        
        numerator = 0.0
        denominator = 0.0
        
        # Create a counter for the expansion term in the observation sequence
        term_count = 1 if expansion_term_to_maximize in observation_sequence else 0
        
        if term_count > 0:
            # This is much faster than iterating through each term
            term_pair = (expansion_term_to_maximize, source_to_maximize)
            prob = self.prob_term_belongs_to_source.get(term_pair, 0.5)
            numerator = term_count * prob
            
            # For denominator, sum over all terms that are in observation_sequence
            relation_prob = 0.0
            for term in observation_sequence:
                term_pair = (term, source_to_maximize)
                relation_prob += self.prob_term_belongs_to_source.get(term_pair, 0.5)
            
            denominator = relation_prob
        
        # Prevent division by zero
        return numerator / max(denominator, 1e-10) if denominator > 0 else 0.0

    def _estimation_step(self, observation_sequence: set) -> None:
        """Optimized expectation step"""
        updated_prob_term_belongs_to_source = {}
        
        # Process in parallel with optimized batching
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            
            # Create all pairs for processing
            pairs = [(term, source) 
                    for term in observation_sequence 
                    for source in self.prob_of_selecting_source.keys()]
            
            # Process pairs in chunks to reduce overhead
            chunk_size = min(100, max(1, len(pairs) // executor._max_workers))
            for i in range(0, len(pairs), chunk_size):
                chunk = pairs[i:i+chunk_size]
                futures.append(executor.submit(self._process_term_source_batch, chunk))
            
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                batch_results = future.result()
                updated_prob_term_belongs_to_source.update(batch_results)
        
        self.prob_term_belongs_to_source = updated_prob_term_belongs_to_source

    def _process_term_source_batch(self, pairs):
        """Process a batch of term-source pairs"""
        result = {}
        for term, source in pairs:
            prob = self._estimate_prob_term_belongs_to_source(term, source)
            if prob > 0:  # Only store non-zero probabilities
                result[(term, source)] = prob
        return result

    def _estimate_prob_term_belongs_to_source(self, term_to_estimate, source_to_estimate):
        """Optimized probability estimation"""
        # Short-circuit if term is not in collection set
        if term_to_estimate not in self.collection_set:
            return 0.0
            
        numerator = 0.0
        denominator = 0.0
        
        # Directly use probability for the term if it exists
        expansion_term_source_pair = (term_to_estimate, source_to_estimate)
        prob = self.prob_expansion_term_represents_source.get(
            expansion_term_source_pair, self.default_prob)
        
        # Only calculate if probability is significant
        if prob > 0:
            numerator = self.prob_of_selecting_source[source_to_estimate] * prob
            
            # Calculate denominator more efficiently
            for source in self.sources.keys():
                expansion_term_source = (term_to_estimate, source)
                source_prob = self.prob_expansion_term_represents_source.get(
                    expansion_term_source, self.default_prob)
                if source_prob > 0:
                    denominator += self.prob_of_selecting_source[source] * source_prob
        
        # Prevent division by zero
        return numerator / max(denominator, 1e-10) if denominator > 0 else 0.0

    def _perform_em_algorithm(self, observation_sequence: set, source: SourceForExpansion, tokenized_query: list[str]) -> list:
        """Optimize EM algorithm implementation"""
        previous_likelihood = -inf
        current_likelihood = 0
        iteration = 0
        max_iterations = 30  # Add a cap on iterations to prevent infinite loops
        
        # Initialize with reasonable default values where possible
        self._precompute_default_probabilities(observation_sequence, source)
        
        while True:
            iteration += 1
            current_likelihood = self._log_likelihood(observation_sequence)
            
            # Convergence check
            if (current_likelihood <= previous_likelihood + self.THRESHOLD_FOR_EM_ALGO or 
                iteration >= max_iterations):
                break
                
            previous_likelihood = current_likelihood
            
            print(f"Starting iteration {iteration} for source {source}")
            self._estimation_step(observation_sequence)
            if source == "RELEVANT_SET_CONTENT": 
                print("Finish estimation step")
            
            self._maximization_step(observation_sequence)
            if source == "RELEVANT_SET_CONTENT": 
                print("Finish maximization step")
                print(f"Current likelihood: {current_likelihood}")

        # Extract top terms efficiently
        return self._extract_top_terms(source, tokenized_query)

    def _precompute_default_probabilities(self, observation_sequence, source):
        """Precompute some default probabilities to speed up first iterations"""
        # For terms in observation sequence, set slightly higher initial probabilities
        for term in observation_sequence:
            if term in self.collection_set:
                self.prob_expansion_term_represents_source[(term, source)] = self.default_prob * 2

    def _extract_top_terms(self, source, tokenized_query):
        """Extract top terms more efficiently"""
        # Only consider terms not in the query
        query_set = set(tokenized_query)
        
        # Filter and sort in one pass
        expansion_items = [
            (term, prob) for (term, src), prob in self.prob_expansion_term_represents_source.items() 
            if src == source and term not in query_set
        ]
        
        # Sort once
        expansion_items.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N items
        return [(term, prob) for term, prob in expansion_items[:self.NUMBER_OF_EXPANSION_TERM]]

    def _expand(self, tokenized_query: list[str]) -> list[str]:
        """Keep the original expand implementation but optimize internal calls"""
        observation_sequence_title = self._get_term_set_of_source("RELEVANT_SET_TITLE")
        observation_sequence_content = self._get_term_set_of_source("RELEVANT_SET_CONTENT")

        print("Starting EM algorithm for title set")
        expansion_term_with_prob_from_title_relevant_set = self._perform_em_algorithm(
            observation_sequence_title, "RELEVANT_SET_TITLE", tokenized_query)
        print("DONE with title set")
        
        print("Starting EM algorithm for content set")
        expansion_term_with_prob_from_content_relevant_set = self._perform_em_algorithm(
            observation_sequence_content, "RELEVANT_SET_CONTENT", tokenized_query)
        print("DONE with content set")
        
        # Efficiently combine results
        term_prob_dict = {}
        for term, prob in expansion_term_with_prob_from_title_relevant_set:
            term_prob_dict[term] = prob

        for term, prob in expansion_term_with_prob_from_content_relevant_set:
            if term not in term_prob_dict or prob > term_prob_dict[term]:
                term_prob_dict[term] = prob

        # Sort once and return
        sorted_terms = sorted(term_prob_dict.items(), key=lambda item: item[1], reverse=True)
        expansion_term_final = [term for term, _ in sorted_terms[:self.NUMBER_OF_EXPANSION_TERM]]
        return expansion_term_final