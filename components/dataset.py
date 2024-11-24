from functools import reduce
import os
from torch.utils.data import Dataset
import csv
from typing import Literal, Callable, TypeAlias
import py_vncorenlp
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast, BatchEncoding
import json


SupportLanguage: TypeAlias = Literal[
    'vie',
    'khmer'
]


class LanguageProcessing:
    """
    Class for language processing. This class help the model can be scale to every languages. 
    You just need to provided appropriate function as decribe in the constructor for the language.
    Vietnamese and Khmer are supported by default, so no function need to be provided.
    """

    def __init__(
            self,
            language: str = 'vie',
            pre_trained_tokenizer_model: PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
            word_sentence_segment: Callable[[str], list[str]] | None = None,
            tokenizer: Callable[[str], list[str]] | None = None,
            encoder: Callable[[str | list[str]], list[int]] | None = None,
            chunk_combiner: Callable[[list[str]], str] = None
    ) -> None:
        """
        Args:
            language (str): Language that needs to be processed. Default is Vietnamese.

            pre_trained_tokenizer_model (PreTrainedTokenizer | PreTrainedTokenizerFast | None):\
            An instance of `PreTrainedTokenizer`/`PreTrainedTokenizerFast` class, include all common methods for\
            processing and encoding string input. If not provided, use default (Vietnamese and Khmer support only).\
            Vietnamese and Khmer are supported by default, so no need to pass this parameter.

            word_sentence_segment (Callable[[str], list[str]] | None): Word and sentence segmentation function for the language.\
            Each element in the returned list should be a word-segmented sentence.\
            If not provided, use default (Vietnamese and Khmer support only).\
            Vietnamese and Khmer are supported by default, so no need to pass this parameter.

            tokenizer (Callable[[str], list[str]] | None): Function to converts a string into a sequence of tokens.\
            If not provided, the method `pre_trained_tokenizer_model.tokenize` will be used instead\
            (This method has been implemented in PreTrainedTokenizer/PreTrainedTokenizerFast class definition).\
            Vietnamese and Khmer are supported by default, so no need to pass this parameter.

            encoder (Callable[[str | list[str]], list[int]] | None): Function to converts\
            a word-segmented string or list of tokenized string to a sequence of ids (integer) for the language.\
            If not provided, the method `pre_trained_tokenizer_model.encode` will be used instead\
            (This method has been implemented in PreTrainedTokenizer/PreTrainedTokenizerFast class definition).\
            Vietnamese and Khmer are supported by default, so no need to pass this parameter.

            chunk_combiner (Callable[[list[str]], str]): Function to combine a list of sentences (list of words or tokens are OK too,\
            depends on your inplementation of the function) to a single chunk (or can be called "partial paragraph").\
            This function is required due to the fact that there are some languages\
            which are non-segmented script (like Khmer), so when consecutive words or sentences are written,\
            the maybe a unique grammar rule that needs to be custom-defined. For Vietnamese and some language like English though,\
            we usually just need to join the list of sentences with a space character.\
            If not provided, use default (Vietnamese and Khmer support only).\
            Vietnamese and Khmer are supported by default, so no need to pass this parameter.
        """
        self.language: str = language

        self.pre_trained_tokenizer_model: PreTrainedTokenizer | PreTrainedTokenizerFast = self._load_pre_trained_tokenizer_model(
        ) if pre_trained_tokenizer_model is None else pre_trained_tokenizer_model

        self.word_sentence_segment: Callable[[str], list[str]] = self._load_word_sentence_segment(
        ) if word_sentence_segment is None else word_sentence_segment

        self.tokenizer: Callable[[
            str], list[str]] = self.pre_trained_tokenizer_model.tokenize if tokenizer is None else tokenizer

        self.encoder: Callable[[str | list[str]], list[int]
                               ] = self.pre_trained_tokenizer_model.encode if encoder is None else encoder

        self.chunk_combiner: Callable[[list[str]], str] = self._load_chunk_combiner(
        ) if chunk_combiner is None else chunk_combiner

    def _load_pre_trained_tokenizer_model(self):
        if self.language == "vie":
            tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
            return tokenizer
        elif self.language == "khmer":
            raise NotImplementedError("Khmer has not been implemented yet")
        else:
            raise NotImplementedError(
                f"The language {self.language} is not supported by default. You have to pass your own PreTrainedTokenizer object!")

    def _load_word_sentence_segment(self) -> Callable[[str], list[str]]:
        """
        Load appropriate word segmentation function for the language

        Returns:
            Callable[[str], list[str]]: A function receives a string and return a list of word-segmented string. 
            Each element in the list represents a word-segmented sentence.
        """
        if self.language == "vie":
            if os.path.isdir("/vncorenlp/models") == False or os.path.exists('/vncorenlp/VnCoreNLP-1.2.jar') == False:
                py_vncorenlp.download_model("/vncorenlp")
            return py_vncorenlp.VnCoreNLP(
                annotators=["wseg"], save_dir="/vncorenlp").word_segment
        elif self.language == "khmer":
            raise NotImplementedError("Khmer has not been implemented yet")
        else:
            raise NotImplementedError(
                f"The language {self.language} is not supported by default. You have to implement word segmentation by yourself!")

    def _load_chunk_combiner(self) -> Callable[[list[str]], str]:
        """
        Load appropriate function that combines a list of sentences to a single chunk for the language.

        Returns:
            Callable[[list[str]], str]: A function receives a list of string (represents a list of sentences)
            and returns a single string (represents a single combined chunk).
        """
        def vietnamese_chunk_combiner(sentence_lst: list[str]) -> str:
            return " ".join(sentence_lst)
        if self.language == "vie":
            return vietnamese_chunk_combiner
        elif self.language == "khmer":
            raise NotImplementedError("Khmer has not been implemented yet")
        else:
            raise NotImplementedError(
                f"The language {self.language} is not supported by default. You have to implement word segmentation by yourself!")


class DocumentDataset(Dataset, LanguageProcessing):
    def __init__(
            self,
            document_dir: str,
            processed_doc_store_dir: str,
            language: SupportLanguage = 'vie',
            language_processing: LanguageProcessing = LanguageProcessing('vie')
    ):
        """
        Args:
            document_dir (str): ABSOLUTE path to the directory containing documents with title, topic, and content xml-tag.

            processed_doc_store_dir (str): ABSOLUTE path to the directory where you want to store preprocessed-documents.

            language (SupportLanguage): Language of the documents. Default is Vietnamese (Khmer has not been implemented yet).

            language_processing (LanguageProcessing): Language processing object for the language of the documents.
        """
        LanguageProcessing.__init__(self, language, language_processing.word_sentence_segment,
                                    language_processing.tokenizer, language_processing.encoder)
        Dataset.__init__(self)
        self.document_dir: str = document_dir
        self.processed_doc_store_dir: str = processed_doc_store_dir
        self.document_count: int = self._load_documents()

    def _load_documents(self) -> int:
        """
        Load the raw documents from the specified directory and parse their titles, contents and topics.
        Then, segment and tokenize the text part of theses documents.
        Finally, save all these data in json format for later use.

        Returns:
            int: The number of documents
        """
        for idx, file_name in enumerate(os.listdir(self.document_dir)):
            if file_name.endswith('.txt'):  # Process only text files
                file_path: str = os.path.join(self.document_dir, file_name)
                title, topic, content: str = self._parse_document(file_path)
                title_segmented: list[str] = self.word_sentence_segment(title)
                content_segmented: list[str] = self.word_sentence_segment(
                    content)
                title_tokenized: list[str] = reduce(
                    lambda prev, curr: prev + self.tokenizer(curr), title_segmented, [])
                content_tokenized: list[str] = reduce(
                    lambda prev, curr: prev + self.tokenizer(curr), content_segmented, [])
                document = {
                    "title_segmented": title_segmented,
                    "content_segmented": content_segmented,
                    "title_tokenized": title_tokenized,
                    "content_tokenized": content_tokenized,
                    "topic": topic,
                    "file_path": file_path
                }
                json_file_path = os.path.join(
                    self.processed_doc_store_dir, f"{idx}.json")
                with open(json_file_path, 'w') as json_file:
                    json.dump(document, json_file)
        return len(os.listdir(self.processed_doc_store_dir))

    def _parse_document(self, file_path: str) -> tuple[str, str, str]:
        """
        Parse the document to extract title, topic, and content.

        Args:
            file_path (str): Path to the document file.

        Returns:
            tuple: (title, content, topic) where title and topic may be empty string if not found.
        """
        title: str = ''
        topic: str = ''
        content: str = ''

        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                lines: list[str] = file.readlines()
                title = self._extract_tag_content(
                    lines, 'title', default='')
                topic = self._extract_tag_content(
                    lines, 'topic', default='')
                content = self._extract_tag_content(
                    lines, 'content', default='')
        except FileNotFoundError as e:
            print(f"Error: {e} - Could not find file: {file_path}")

        return title, topic, content

    def _extract_tag_content(self, lines: list[str], tag: str, default: str = None) -> str:
        """
        Extract content between a pair of XML-like tags.

        Args:
            lines (list of str): list of lines from the document.
            tag (str): The tag to extract (e.g., 'title', 'content', 'topic').
            default: The default value to return if the tag is not found.

        Returns:
            str: The extracted content, or the default value if the tag is not found.
        """
        opening_tag = f'<{tag}>'
        closing_tag = f'</{tag}>'
        inside_tag = False
        content_lines = []

        for line in lines:
            line = line.strip()
            if opening_tag in line:
                inside_tag = True
                # Remove the opening tag if it's on the same line
                line = line.replace(opening_tag, '').strip()
            if inside_tag:
                if closing_tag in line:
                    # Remove the closing tag and stop collecting
                    line = line.replace(closing_tag, '').strip()
                    content_lines.append(line)
                    break
                else:
                    content_lines.append(line)

        # Return the joined content if found, otherwise return the default value
        return '\n'.join(content_lines).strip() if content_lines else default

    def find_document_by_file_path(self, path_to_find: str) -> tuple[list[str], list[str]]:
        """
        Return turn the segmented title and content of a document that have its file path match the input.

        Args:
            path_to_find (str): The path of the document need to be found and retrieved title and content.

        Returns:
            tuple[list[str], list[str]]: A tuple contains segmented title and content of the found document.
        """
        # Iterate over each JSON file in the output directory
        for json_file_name in os.listdir(self.processed_doc_store_dir):
            json_file_path = os.path.join(
                self.processed_doc_store_dir, json_file_name)
            with open(json_file_path, 'r') as json_file:
                document = json.load(json_file)

                # Check if the file_path matches the search path
                if document["file_path"] == path_to_find:
                    return (document["title_segmented"], document["content_segmented"])

        raise FileNotFoundError(
            f"No document can be found at the path: {path_to_find}")

    def __len__(self):
        return len(self.document_count)

    def __getitem__(self, idx: int) -> tuple[list[str], list[str], list[str], list[str], str, str]:
        json_file_path = os.path.join(
            self.processed_doc_store_dir, f"{idx}.json")
        with open(json_file_path, 'r') as json_file:
            document = json.load(json_file)

        return (
            document["title_segmented"],
            document["content_segmented"],
            document["title_tokenized"],
            document["content_tokenized"],
            document["topic"],
            document["file_path"]
        )


class QueryDocDataset(Dataset, LanguageProcessing):
    def __init__(
            self,
            qd_dir: str,
            language: SupportLanguage = 'vie',
            language_processing: LanguageProcessing = LanguageProcessing('vie')
    ):
        """
        Args:
            qd_dir (str): Path to the folder containing CSV files with queries and corresponding answer document file paths (ABSOLUTE path).

            language (SupportLanguage): Language of the QA set. Default is Vietnamese (Khmer has not been implemented yet).

            language_processing (LanguageProcessing): Language processing object for the language of the QA set.
        """
        LanguageProcessing.__init__(self, language, language_processing.word_sentence_segment,
                                    language_processing.tokenizer, language_processing.encoder)
        Dataset.__init__(self)
        self.qd_dir: str = qd_dir
        self.qd_pairs: list[tuple[str, list[str]]
                            ] = self._load_qd_pairs()

    def _load_qd_pairs(self) -> list[tuple[list[str], list[str]]]:
        """
        Load queries and their corresponding answer documents from CSV files in the specified folder,
        then word-segment the queries.

        Attention: The corresponding answer documents file paths for each query is just a list of file path
        to where the documents is stored. So if you just use this path to retrive the document,
        this document will be raw (pure text, no word segmentation or any technique applied). 
        You should do it yourself (this class inherits from LanguageProcessing, so it already has requied methods), 
        or use this path to find the document from `DocumentDataset` (method `find_document_by_file_path`).

        Returns:
            list[tuple[str, list[str]]]: A list of tuples containing (query_segmented, document_file_path_list).
        """
        qd_pairs: list[tuple[str, list[str]]] = []
        for file_name in os.listdir(self.qd_dir):
            if file_name.endswith('.csv'):  # Process only CSV files
                file_path: str = os.path.join(self.qd_dir, file_name)
                with open(file_path, 'r', encoding='utf-8') as csv_file:
                    reader = csv.reader(csv_file)
                    next(reader)  # Skip header if there is one
                    for row in reader:
                        if len(row) >= 1:  # Ensure there is at least one column (the query)
                            # The first column is the query
                            query = row[0].strip()
                            query_segmented: str = self.chunk_combiner(
                                self.word_sentence_segment(query))
                            # Collect all subsequent columns as document filenames
                            document_file_path_list = [f.strip()
                                                       for f in row[1:] if f.strip()]
                            qd_pairs.append(
                                (query_segmented, document_file_path_list))
        return qd_pairs

    def __len__(self):
        return len(self.qd_pairs)

    def __getitem__(self, idx: int) -> tuple[str, list[str]]:
        query_segmented, document_file_path_list = self.qd_pairs[idx]
        return query_segmented, document_file_path_list


class ParallelDataset(Dataset):
    def __init__(
        self,
        parallel_dir: str,
        teacher_language_processing: LanguageProcessing,
        student_language_processing: LanguageProcessing
    ):
        """
        Args:
            parallel_dir (str): Path to the folder containing CSV files with parallel sentences.

            teacher_language_processing (LanguageProcessing): Language processing object\
            for the teacher language.

            student_language_processing (LanguageProcessing): Language processing object\
            for the student language.
        """
        self.parallel_dir: str = parallel_dir
        self.teacher_language_processing: LanguageProcessing = teacher_language_processing,
        self.student_language_processing: LanguageProcessing = student_language_processing,
        self.pairs = self._load_pairs()

    def _load_pairs(self) -> list[tuple[list[str], list[str]]]:
        """
        Load parallel sentence pairs from CSV files in the specified folder.
        Then, word-segment the two sentences in each pair.

        Returns:
            list: A list of tuples containing pairs of word-segmented sentences.
        """
        pairs: list[tuple[list[str], list[str]]] = []
        for file_name in os.listdir(self.parallel_dir):
            if file_name.endswith('.csv'):  # Process only CSV files
                file_path = os.path.join(self.parallel_dir, file_name)
                with open(file_path, 'r', encoding='utf-8') as file:
                    reader = csv.reader(file)
                    for row in reader:
                        if len(row) == 2:  # Ensure there are exactly two sentences
                            # Append the pair of sentences
                            teacher_language_sentence: str = row[0].strip()
                            student_language_sentence: str = row[1].strip()
                            segmented_teacher_language_sentence: list[str] = self.teacher_language_processing.word_sentence_segment(
                                teacher_language_sentence)
                            segmented_student_language_sentence: list[str] = self.student_language_processing.word_sentence_segment(
                                student_language_sentence)
                            pairs.append(
                                (segmented_teacher_language_sentence, segmented_student_language_sentence))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[list[str], list[str]]:
        segmented_teacher_language_sentence, segmented_student_language_sentence = self.pairs[
            idx]
        return segmented_teacher_language_sentence, segmented_student_language_sentence


class MLMFineTuneDataset(Dataset):
    """
    Class for representing the MLM-style samples dataset. This class uses preprocess documents
    getting from the DocumentDataset instance to generate correspond MLM-style samples.
    """

    def __init__(self, document_dataset: DocumentDataset) -> None:
        """
        Args:
            document_dataset (DocumentDataset): An object of DocumentDataset class, represent a document dataset use to generate MLM samples
        """
        self.document_dataset = document_dataset
        self.samples: list[dict] = self._create_samples()

    def _create_samples(self) -> list[dict]:
        """
        Generate the list of MLM-style samples from each document in the dataset of document provided in constructor.

        Returns:
            list[dict]: List of dictionary, each of dictionary is a MLM-style sample with two keys: 
            `input_ids` represent the sequence of ids of a chunk of token, and 
            `attention_mask` represent the mask of each corresponding token (1 is normal token, 0 is padding).
        """
        samples: list[dict] = []
        for _, _, _, tokenize_content, _, _ in self.document_dataset:
            model_max_length: int = self.document_dataset.pre_trained_tokenizer_model.model_max_length
            stride: int = min(model_max_length // 10, 50)
            example_lst: BatchEncoding = self.document_dataset.pre_trained_tokenizer_model(
                tokenize_content,
                is_split_into_words=True,
                return_tensors='pt',
                max_length=model_max_length,
                truncation=True,
                stride=stride,
                padding="max_length")
            for i in range(len(example_lst["input_ids"])):
                example = {
                    "input_ids": example_lst["input_ids"][i],
                    "attention_mask": example_lst["attention_mask"][i]
                }
                samples.append(example)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]
