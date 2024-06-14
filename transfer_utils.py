from transformers import (AutoModel, AutoModelForTokenClassification, AutoModelForCausalLM, AutoTokenizer)
from tokenizer_transfer.corpuses import load_buster_corpus
from logging import getLogger

logger = getLogger(__name__)
logger.setLevel('INFO')

def instantiate_model(path: str, model_class: str, model_kwargs: dict):
    if model_class == 'AutoModelForTokenClassification':
        model = AutoModelForTokenClassification.from_pretrained(path, **model_kwargs)
    elif model_class == 'AutoModelForCausalLM':
        model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
    else:
        model = AutoModel.from_pretrained(path, **model_kwargs)
    return model


def train_tokenizer(tokenizer_name, corpus_name: str, vocab_size: int,
                    output: str = 'tokenizer_trained', seed=42) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, add_prefix_space=True)
    vocab_size = int(tokenizer.vocab_size * vocab_size)
    if corpus_name == 'buster':
        logger.info("Loading BUSTER corpus")
        corpus = load_buster_corpus(seed)
    else:
        corpus = ['']

    target_tokenizer = tokenizer.train_new_from_iterator(corpus, vocab_size)
    target_tokenizer.save_pretrained(output)
    return target_tokenizer
