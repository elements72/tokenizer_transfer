import logging
import os
import subprocess
import sys
from logging import getLogger

import yaml
from dataclasses import dataclass
from transformers import AutoModelForTokenClassification
from transformers import HfArgumentParser, AutoTokenizer, AutoModel
from tokenizer_transfer.transfer_utils import instantiate_model, train_tokenizer
from fvt.fvt import FastVocabularyTransfer

logger = getLogger(__name__)
logger.setLevel('INFO')


@dataclass
class TransferArguments:
    tokenizer_name: str = "xlm-roberta-base"  # The tokenizer to be transferred
    model_class: str = "AutoModelForTokenClassification"
    target_model: str = "xlm-roberta-base"  # Target model to transfer the tokenizer
    output: str = target_model
    hypernet: str = "https://huggingface.co/benjamin/zett-hypernetwork-xlm-roberta-base"  # Hypernetwork to generate the new embeddings only zett
    lang_code: str = None
    make_whitespace_consistent: bool = True
    save_pt: bool = True
    method: str = "zett"
    vocab_size: float = 1.0
    corpus: str = 'buster'
    revision: str = None
    push_to_hub: bool = False
    fine_tuned: bool = True
    config_file: str = None
    base_model: str = "xlm-roberta-base"  # Base model to transfer the tokenizer, used only when we need to perform post-training transfer


def fvt_transfer(in_tokenizer: AutoTokenizer, gen_tokenizer: AutoTokenizer, gen_model: AutoModel):
    fvt = FastVocabularyTransfer()
    model = fvt.transfer(
        in_tokenizer=in_tokenizer,
        gen_tokenizer=gen_tokenizer,
        gen_model=gen_model)
    return model


def get_transfer_name(corpus_name: str, target_model: str, vocab_size: float, method, fine_tuned: bool) -> str:
    # If target model is a path, get the model name
    if '/' in target_model:
        target_model = target_model.split('/')[-1]
    # Delete version from name 'vx.x'
    target_model = target_model.split('v')[0]
    # Convert into percentage
    vocab_size = int(vocab_size * 100)
    return f'{corpus_name}-{"post" if fine_tuned else "pre"}FT-{method}-{target_model}-{vocab_size}'


def zett_transfer_script(target_tokenizer: str, target_model: str,
                         hypernet: str = 'https://huggingface.co/benjamin/zett-hypernetwork-xlm-roberta-base',
                         output='buster-corpus-zett',
                         model_class='AutoModelForTokenClassification',
                         zett_args=None,
                         model_kwargs=None
                         ) -> AutoModel:
    """
    Perform zero-shot zett transfer, the model and tokenizer are saved in the output directory.
    :param target_tokenizer:
    :param target_model:
    :param hypernet:
    :param output:
    :param model_class:
    :param model_kwargs:
    :return:
    """
    if model_kwargs is None:
        model_kwargs = {}
    if zett_args is None:
        zett_args = {}

    hypernet_name = hypernet.split('/')[-1]

    # Check if hypernetwork is already cloned
    if hypernet_name not in os.listdir():
        logger.info(f"Cloning hypernetwork {hypernet}")
        command = [
            "git", "clone", f"{hypernet}"]

        result = subprocess.run(command, capture_output=True, text=True)

        print(result.stdout)
        print(result.stderr)
    else:
        logger.info(f"Hypernetwork {hypernet_name} already cloned")

    path = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(path, 'zett/scripts/transfer.py')

    # Define the command and its arguments
    command = [
        "python3", script_path,
        f"--target_model={target_model}",
        f"--tokenizer_name={target_tokenizer}",
        f"--output={output}",
        f"--model_class={model_class}",
        f"--checkpoint_path={hypernet_name}",
        "--save_pt"
    ]
    # Add the zett arguments
    for key, value in zett_args.items():
        if value is not None:
            command.append(f"--{key}={value}")

    # Run the command
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    model = instantiate_model(output, model_class, model_kwargs)
    return model


def vocabulary_transfer(args: TransferArguments, model_kwargs=None):
    if model_kwargs is None:
        model_kwargs = {}
    transfer_name = get_transfer_name(args.corpus, args.output, args.vocab_size, args.method, args.fine_tuned)
    # Train the new tokenizer
    target_tokenizer = train_tokenizer(args.tokenizer_name, args.corpus, args.vocab_size)

    if args.method == 'zett':
        logger.info("Performing ZETT transfer")
        # If we are performing post FT we need to transfer the original model before
        zett_target = args.base_model if args.fine_tuned else args.target_model

        zett_args = {
            'lang_code': args.lang_code,
            'make_whitespace_consistent': args.make_whitespace_consistent,
            'save_pt': args.save_pt,
            'revision': args.revision
        }

        model = zett_transfer_script(target_tokenizer='tokenizer_trained',
                                     target_model=zett_target,
                                     output=transfer_name,
                                     model_class=args.model_class,
                                     hypernet=args.hypernet,
                                     zett_args=zett_args,
                                     model_kwargs=model_kwargs)
    else:
        logger.info("Performing FVT transfer")
        # Get the original tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, add_prefix_space=True)
        model = AutoModelForTokenClassification.from_pretrained(args.target_model, **model_kwargs)
        model = fvt_transfer(target_tokenizer, tokenizer, model)
        # Save the model and tokenizer
        model.save_pretrained(transfer_name)
        target_tokenizer.save_pretrained(transfer_name)
    if args.push_to_hub:
        logger.info("Pushing to Hugging Face Hub")
        model.push_to_hub(transfer_name)
        target_tokenizer.push_to_hub(transfer_name)
    logger.info("Transfer complete")
    return model, target_tokenizer


if __name__ == "__main__":
    (args,) = HfArgumentParser([TransferArguments]).parse_args_into_dataclasses()
    if args.config_file is not None:
        # Check if the config file exists and is a yaml file
        if not os.path.isfile(args.config_file):
            logger.error(f"Config file {args.config_file} not found")
            sys.exit(1)
        # Load the config file
        with open(args.config_file, 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
            args = TransferArguments(**config)

    vocabulary_transfer(args)
