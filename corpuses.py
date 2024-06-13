import datasets
import numpy as np


def load_buster_corpus(seed=42, val_size=0.2) -> list[list[str]]:
    """
    Return train, val and test data for the BUSTER dataset
    :param seed:
    :return:
    """

    # train_data = datasets.load_dataset('expertai/BUSTER', split='FOLD_1')
    # val_data = datasets.load_dataset('expertai/BUSTER', split='FOLD_2')
    data = datasets.load_dataset('expertai/BUSTER')
    # Remove 'SILVER' split
    data = datasets.concatenate_datasets([data['FOLD_1'], data['FOLD_2'], data['FOLD_3'], data['FOLD_4']])
    test_data = datasets.load_dataset('expertai/BUSTER', split='FOLD_5')

    # Split the dataset into training and validation and test
    generator = np.random.default_rng(seed)
    # Perform train val test split 0.8, 0.12
    data_dict = data.train_test_split(test_size=val_size, generator=generator)
    train_data = data_dict['train']
    return train_data['text']
