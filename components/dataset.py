import os
from torch.utils.data import Dataset
import csv
from typing import Literal, Callable, TypeAlias
import py_vncorenlp
from transformers import AutoTokenizer
from abc import ABC


SupportLanguage: TypeAlias = Literal[
    'vie',
    'khmer'
]


class LanguageProcessingInterface(ABC):
    """
    Base class for language processing. This class can be inherited, so that the model
    can be scale to every languages. You just need to provided appropriate function
    as decribe in the constructor for the language. Vietnamese and Khmer are supported
    by default, so no function need to be provided.
    """

    def __init__(
            self,
            language: str = 'vie',
            word_segment: Callable[[str], list[str]] | None = None,
            tokenizer: Callable[[str], list[str]] | None = None,
            encoder: Callable[[str | list[str]], list[int]] | None = None
    ) -> None:
        """
        Args:
            language (str): Language that needs to be processed. Default is Vietnamese.

            word_segment (Callable[[str], list[str]] | None): Word segmentation function for the language.\
            If not provided, use default (Vietnamese and Khmer support only).\
            Vietnamese and Khmer are supported by default, so no need to pass this parameter.

            tokenizer (Callable[[str], list[str]] | None): Tokenizer for the language.\
            If not provided, use default (Vietnamese and Khmer support only).\
            Vietnamese and Khmer are supported by default, so no need to pass this parameter.

            encoder (Callable[[str | list[str]], list[int]] | None): Function to converts\
            a word-segmented string or list of tokenized string\
            to a sequence of ids (integer) for the language.\
            If not provided, use default (Vietnamese and Khmer support only).\
            Vietnamese and Khmer are supported by default, so no need to pass this parameter.
        """
        self.language: str = language

        self.word_segment: Callable[[str], list[str]] = self._load_word_segment(
        ) if word_segment is None else word_segment

        self.tokenizer: Callable[[str], list[str]] = self._load_tokenizer(
        ) if tokenizer is None else tokenizer

        self.encoder: Callable[[str | list[str]], list[int]
                               ] = self._load_encoder() if encoder is None else encoder

    def _load_word_segment(self) -> Callable[[str], list[str]]:
        """
        Load appropriate word segmentation function for the language

        Returns:
            Callable[[str], list[str]]: A function receives a string and return a list of word-segmented string. 
            Each element in the list represents a word-segmented sentence.
        """
        if self.language == "vie":
            return py_vncorenlp.VnCoreNLP(
                annotators=["wseg"], save_dir="/vncorenlp").word_segment
        elif self.language == "khmer":
            raise NotImplementedError("Khmer has not been implemented yet")
        else:
            raise NotImplementedError(
                f"The language {self.language} is not supported by default. You have to implement word segmentation by yourself!")

    def _load_tokenizer(self) -> Callable[[str], list[str]]:
        """
        Load appropriate tokenizer for the language

        Returns:
            Callable[[str], list[str]]: A function receives a word-segmented string and return a list of tokenized string.
        """
        if self.language == "vie":
            tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
            return tokenizer.tokenize
        elif self.language == "khmer":
            raise NotImplementedError("Khmer has not been implemented yet")
        else:
            raise NotImplementedError(
                f"The language {self.language} is not supported by default. You have to implement tokenizer by yourself!")

    def _load_encoder(self) -> Callable[[str | list[str]], list[int]]:
        """
        Load appropriate function to converts a word-segmented string or list of tokenized string
        to a sequence of ids (integer) for the language

        Returns:
            Callable[[str | list[str]], list[int]]: A function receives a word-segmented string or list of tokenized string,
            return sequence of token IDs
        """
        if self.language == "vie":
            tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
            return tokenizer.encode
        elif self.language == "khmer":
            raise NotImplementedError("Khmer has not been implemented yet")
        else:
            raise NotImplementedError(
                f"The language {self.language} is not supported by default. You have to implement encoder by yourself!")


class DocumentDataset(Dataset, LanguageProcessingInterface):
    def __init__(
            self,
            document_dir: str,
            language: SupportLanguage = 'vie',
            word_segment: Callable[[str], list[str]] | None = None,
            tokenizer: Callable[[str], list[str]] | None = None,
            encoder: Callable[[str | list[str]], list[int]] | None = None
    ):
        """
        Args:
            document_dir (str): ABSOLUTE path to the directory containing documents with title, topic, and content xml-tag.

            language (SupportLanguage): Language of the documents. Default is Vietnamese (Khmer has not been implemented yet).

            word_segment (Callable[[str], list[str]] | None): Word segmentation function for the language.\
            If not provided, use default (Vietnamese and Khmer support only).\
            Vietnamese and Khmer are supported by default, so no need to pass this parameter.

            tokenizer (Callable[[str], list[str]] | None): Tokenizer for the language.\
            If not provided, use default (Vietnamese and Khmer support only).\
            Vietnamese and Khmer are supported by default, so no need to pass this parameter.

            encoder (Callable[[str | list[str]], list[int]] | None): Function to converts\
            a word-segmented string or list of tokenized string\
            to a sequence of ids (integer) for the language.\
            If not provided, use default (Vietnamese and Khmer support only).\
            Vietnamese and Khmer are supported by default, so no need to pass this parameter.
        """
        LanguageProcessingInterface.__init__(
            self, language, word_segment, tokenizer, encoder)
        Dataset.__init__(self)
        self.document_dir: str = document_dir
        self.documents: list[tuple[list[str],
                                   list[str], str, str]] = self._load_documents()

    def _load_documents(self) -> list[tuple[list[str], list[str], str]]:
        """
        Load documents from the specified directory, parse their contents, and segment by words.

        Returns:
            list[tuple[list[str], list[str], str, str]]: A list of tuples containing (title, content, topic, file_path).
        """
        documents: list[tuple[list[str], list[str], str]] = []
        for file_name in os.listdir(self.document_dir):
            if file_name.endswith('.txt'):  # Process only text files
                file_path: str = os.path.join(self.document_dir, file_name)
                title, topic, content: str = self._parse_document(file_path)
                title_segmented: list[str] = self.word_segment(title)
                content_segmented: list[str] = self.word_segment(content)
                documents.append(
                    (title_segmented, content_segmented, topic, file_path))
        return documents

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

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx: int) -> tuple[list[str], list[str], str, str]:
        title_segmented, content_segmented, topic, file_path = self.documents[idx]
        return title_segmented, content_segmented, topic, file_path


class QADataset(Dataset, LanguageProcessingInterface):
    def __init__(
            self,
            qa_dir: str,
            language: SupportLanguage = 'vie',
            word_segment: Callable[[str], list[str]] | None = None,
            tokenizer: Callable[[str], list[str]] | None = None,
            encoder: Callable[[str | list[str]], list[int]] | None = None
    ):
        """
        Args:
            qa_dir (str): Path to the folder containing CSV files with questions and corresponding answer documents.

            language (SupportLanguage): Language of the QA set. Default is Vietnamese (Khmer has not been implemented yet).

            word_segment (Callable[[str], list[str]] | None): Word segmentation function for the language.\
            If not provided, use default (Vietnamese and Khmer support only).\
            Vietnamese and Khmer are supported by default, so no need to pass this parameter.

            tokenizer (Callable[[str], list[str]] | None): Tokenizer for the language.\
            If not provided, use default (Vietnamese and Khmer support only).\
            Vietnamese and Khmer are supported by default, so no need to pass this parameter.

            encoder (Callable[[str | list[str]], list[int]] | None): Function to converts\
            a word-segmented string or list of tokenized string\
            to a sequence of ids (integer) for the language.\
            If not provided, use default (Vietnamese and Khmer support only).\
            Vietnamese and Khmer are supported by default, so no need to pass this parameter.
        """
        LanguageProcessingInterface.__init__(
            self, language, word_segment, tokenizer, encoder)
        Dataset.__init__(self)
        self.qa_dir: str = qa_dir
        self.qa_pairs: list[tuple[list[str], list[str]]
                            ] = self._load_qa_pairs()

    def _load_qa_pairs(self) -> list[tuple[list[str], list[str]]]:
        """
        Load questions and their corresponding answer documents from CSV files in the specified folder,
        then word-segment the questions.

        Attention: The corresponding answer documents for each question is just a list of file path
        to where the documents is stored. So if you just use this path to retrive the document,
        this document will be raw (pure text, no word segmentation or any technique applied). 
        You should do it yourself (this class inherits from LanguageProcessingInterface, so it already has requied methods), 
        or use this path to find the document from DocumentDataset.

        Returns:
            list[tuple[list[str], list[str]]]: A list of tuples containing (question_segmented, document_filenames).
        """
        qa_pairs: list[tuple[list[str], list[str]]] = []
        for file_name in os.listdir(self.qa_dir):
            if file_name.endswith('.csv'):  # Process only CSV files
                file_path: str = os.path.join(self.qa_dir, file_name)
                with open(file_path, 'r', encoding='utf-8') as csv_file:
                    reader = csv.reader(csv_file)
                    next(reader)  # Skip header if there is one
                    for row in reader:
                        if len(row) >= 1:  # Ensure there is at least one column (the question)
                            # The first column is the question
                            question = row[0].strip()
                            question_segmented: list[str] = self.word_segment(
                                question)
                            # Collect all subsequent columns as document filenames
                            document_filenames = [f.strip()
                                                  for f in row[1:] if f.strip()]
                            qa_pairs.append(
                                (question_segmented, document_filenames))
        return qa_pairs

    def __len__(self):
        return len(self.qa_pairs)

    def __getitem__(self, idx: int) -> tuple[list[str], list[str]]:
        question_segmented, document_filenames = self.qa_pairs[idx]
        return question_segmented, document_filenames


class ParallelDataset(Dataset):
    def __init__(
        self,
        parallel_dir: str,
        teacher_language_processing: LanguageProcessingInterface,
        student_language_processing: LanguageProcessingInterface
    ):
        """
        Args:
            parallel_dir (str): Path to the folder containing CSV files with parallel sentences.

            teacher_language_processing (LanguageProcessingInterface): Language processing object\
            for the teacher language.

            student_language_processing (LanguageProcessingInterface): Language processing object\
            for the student language.
        """
        self.parallel_dir: str = parallel_dir
        self.teacher_language_processing: LanguageProcessingInterface = teacher_language_processing,
        self.student_language_processing: LanguageProcessingInterface = student_language_processing,
        self.pairs = self._load_pairs()

    def _load_pairs(self) -> list[tuple[list[str], list[str]]]:
        """
        Load parallel sentence pairs from CSV files in the specified folder.
        Then, word-segment the two sentences in each pair.

        Returns:
            list: A list of tuples containing pairs of sentences.
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
                            segmented_teacher_language_sentence: list[str] = self.teacher_language_processing.word_segment(
                                teacher_language_sentence)
                            segmented_student_language_sentence: list[str] = self.student_language_processing.word_segment(
                                student_language_sentence)
                            pairs.append(
                                (segmented_teacher_language_sentence, segmented_student_language_sentence))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[list[str], list[str]]:
        sentence1, sentence2 = self.pairs[idx]
        return sentence1, sentence2
