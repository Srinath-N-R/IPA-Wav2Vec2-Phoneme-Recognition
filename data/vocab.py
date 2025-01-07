import json
from pathlib import Path
from typing import List, Dict

def convert_phon61_to_phon39(sentence: str, phone_map: Dict[str, str]) -> str:
    tokens = [phone_map.get(x, "[UNK]") for x in sentence.split()]
    return " ".join(tokens)

def create_vocab(train_phonetics: List[str], valid_phonetics: List[str], test_phonetics: List[str]) -> Dict[str, int]:
    vocab_train = set(train_phonetics) | {' '}
    vocab_valid = set(valid_phonetics) | {' '}
    vocab_test = set(test_phonetics) | {' '}

    vocab_list = sorted(vocab_train | vocab_valid | vocab_test)
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}

    # Make the space more intuitive to understand
    vocab_dict["|"] = vocab_dict.pop(" ")
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict) + 1

    return vocab_dict

def save_vocab(vocab_dict: Dict[str, int], filepath: str):
    filepath = Path(__file__).resolve().parent.parent / filepath
    with open(filepath, 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)
