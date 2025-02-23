import sys
import os

# Add the project root directory to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

import torch
from models import MonolingualRetrivalTrainer, MonoLingualRetrival, KnowledgeDistillation
from components import LanguageProcessing


def monolingual_train(
    document_dir,
    processed_doc_store_dir,
    qd_dir,
    pretrained_model_name_or_path,
    do_mlm_fine_tune='True',
    language='vie',
    chunk_length_limit='128',
    device='cpu',
    batch_size='32',
    margin='1.0',
    learning_rate='1e-5',
    epochs='4'
):
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("Your device does not have GPU!")

    trainer = MonolingualRetrivalTrainer(
        document_dir=document_dir,
        processed_doc_store_dir=processed_doc_store_dir,
        qd_dir=qd_dir,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        do_mlm_fine_tune=do_mlm_fine_tune in ('True', 'true', '1'),
        language=language,
        language_processing=LanguageProcessing(language),
        chunk_length_limit=int(chunk_length_limit),
        device=device,
        batch_size=int(batch_size),
        margin=float(margin),
        learning_rate=float(learning_rate),
        epochs=int(epochs)
    )

    sentence_transformer_save_path, custom_sentence_transformer_save_path = trainer.train()
    print(
        f"Fine-tuned SentenceTransformer model has been saved at: {sentence_transformer_save_path}. Use this model in Knowledge Distillation.")
    print(
        f"Trained CustomSentenceTransformer model has been saved at: {custom_sentence_transformer_save_path}. Load this in production model.")


def monolingual_retrive(
    document_dir,
    processed_doc_store_dir,
    custom_sentence_transformer_pretrained_or_save_path,
    language='vie',
    original_query_doc_count='30',
    extended_query_doc_count='30',
    chunk_length_limit='128',
    relevant_threshold='0.1',
    relevant_default_lowerbound='0.25',
    device='cpu',
    batch_size='32'
):
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("Your device does not have GPU!")

    model = MonoLingualRetrival(
        document_dir=document_dir,
        processed_doc_store_dir=processed_doc_store_dir,
        custom_sentence_transformer_pretrained_or_save_path=custom_sentence_transformer_pretrained_or_save_path,
        language=language,
        language_processing=LanguageProcessing(language),
        original_query_doc_count=int(original_query_doc_count),
        extended_query_doc_count=int(extended_query_doc_count),
        chunk_length_limit=int(chunk_length_limit),
        relevant_threshold=float(relevant_threshold),
        relevant_default_lowerbound=float(relevant_default_lowerbound),
        device=device,
        batch_size=int(batch_size)
    )

    while True:
        query = input('Please input your query: ')
        if not query:
            break
        query.strip()
        if not query:
            break
        print(f"Your query: {query}")
        print("Related documents:")
        result = model(query)
        print(result)

def knowledge_distillation(
    # parallel_dir,
    # teacher_language_processing,
    # student_language_processing,
    student_model_language,
    teacher_model_language,
    teacher_model="sentence-transformers/distiluse-base-multilingual-cased-v2",
    distribution="padded_uniform",
    device='cpu',
    batch_size='32',
    epsilon='1.0',
    learning_rate='1e-5',
    epochs='4'
):
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("Your device does not have GPU!")
    
    trainer = KnowledgeDistillation(
        # parallel_dir==parallel_dir,
        # teacher_language_processing=teacher_language_processing,
        # student_language_processing=student_language_processing,
        student_model_language=student_model_language,
        teacher_model_language=teacher_model_language,
        teacher_model = teacher_model,
        distribution = distribution,
        device=device,
        batch_size=int(batch_size),
        epsilon=float(epsilon),
        learning_rate=float(learning_rate),
        epochs=int(epochs)
    )

    multilingual_sentence_transformer_save_path = trainer.train()
    print(
        f"Distilled student Sentence Transformer model has been saved at: {multilingual_sentence_transformer_save_path}. Load this in production model.")


command_map = {
    'monolingual_train': monolingual_train,
    'knowledge_distillation': knowledge_distillation,
    'monolingual_retrive': monolingual_retrive
}


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <command> [arguments...]")
        print("Available commands:", ", ".join(command_map.keys()))
        sys.exit(1)

    command = sys.argv[1]
    args = sys.argv[2:]

    if command not in command_map:
        print(f"Unknown command: {command}")
        print("Available commands:", ", ".join(command_map.keys()))
        sys.exit(1)

    func = command_map[command]

    try:
        func(*args)
    except TypeError as e:
        print(f"Error: {e}")
        print(
            f"Usage: {command} requires {func.__code__.co_argcount} arguments.")


if __name__ == "__main__":
    main()
