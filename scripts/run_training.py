from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
from utils import load_config
from data import return_data, get_durations, create_vocab, save_vocab
from training import train_model
from pathlib import Path
import psutil

import os

os.environ["HF_DATASETS_CACHE"] = "/workspace"
os.environ["TMPDIR"] = "/workspace/tmp"

from datasets import Dataset, Audio, config


os.environ["HF_DATASETS_CACHE"] = "/workspace/tmp"
config.HF_DATASETS_CACHE = Path(os.environ["HF_DATASETS_CACHE"])

print(f"Cache directory: {config.HF_DATASETS_CACHE}")

def main():
    # Load configuration
    config = load_config()

    # Load datasets
    train_data = return_data(config['dataset']['train_folder'], config['dataset']['max_files'])
    valid_data = return_data(config['dataset']['val_folder'], config['dataset']['max_files'])
    test_data = return_data(config['dataset']['test_folder'], config['dataset']['max_files'])

    # # Print durations
    # print(f"Duration of Train: {get_durations(train_data) // 60} mns")
    # print(f"Duration of Valid: {get_durations(valid_data) // 60} mns")
    # print(f"Duration of Test : {get_durations(test_data) // 60} mns")


    def lazy_data_generator(data_dict):
        for _, data in data_dict.items():
            yield {
                "audio_file": data["audio_file"],
                "word_file": data["word_file"],
                "phonetic_file": data["phonetic_file"],
            }

    # Create datasets
    # train_dataset = Dataset.from_dict(train_dict)
    # valid_dataset = Dataset.from_dict(valid_dict)
    # test_dataset = Dataset.from_dict(test_dict)

    train_dataset = Dataset.from_generator(lambda: lazy_data_generator(train_data))
    valid_dataset = Dataset.from_generator(lambda: lazy_data_generator(valid_data))
    test_dataset = Dataset.from_generator(lambda: lazy_data_generator(test_data))

    train_dataset.save_to_disk("/workspace/tmp/train_dataset_cache")
    valid_dataset.save_to_disk("/workspace/tmp/valid_dataset_cache")
    test_dataset.save_to_disk("/workspace/tmp/test_dataset_cache")

    # Read text files
    def read_text_file(filepath):
        with open(filepath, 'r') as f:
            return f.read()

    # def prepare_text_data(item):
    #     item['text'] = read_text_file(str(item['word_file']))
    #     item['phonetic'] = read_text_file(str(item['phonetic_file']))
    #     return item

    def prepare_text_data(batch):
        # Process each file in the batch
        batch['text'] = [read_text_file(str(word_file)) for word_file in batch['word_file']]
        batch['phonetic'] = [read_text_file(str(phonetic_file)) for phonetic_file in batch['phonetic_file']]
        return batch

    train_dataset = train_dataset.map(prepare_text_data, batch_size=1000, batched=True, num_proc=12).remove_columns(["word_file", "phonetic_file"])
    valid_dataset = valid_dataset.map(prepare_text_data, batch_size=1000, batched=True, num_proc=12).remove_columns(["word_file", "phonetic_file"])
    test_dataset  = test_dataset.map(prepare_text_data, batch_size=1000, batched=True, num_proc=12).remove_columns(["word_file", "phonetic_file"])

    # # Phoneme normalization
    # phone_map = config['vocab']['phoneme_map']

    # def normalize_phones(item):
    #     item['phonetic'] = convert_phon61_to_phon39(item['phonetic'], phone_map)
    #     return item

    # train_dataset = train_dataset.map(normalize_phones)
    # valid_dataset = valid_dataset.map(normalize_phones)
    # test_dataset = test_dataset.map(normalize_phones)

    # Collect phonemes
    train_phonetics = [phone for x in train_dataset for phone in x['phonetic'].split(" ")]
    valid_phonetics = [phone for x in valid_dataset for phone in x['phonetic'].split(" ")]
    test_phonetics = [phone for x in test_dataset for phone in x['phonetic'].split(" ")]

    print("num of train phones:\t", len(set(train_phonetics)))
    print("num of valid phones:\t", len(set(valid_phonetics)))
    print("num of test phones:\t", len(set(test_phonetics)))

    # Create vocabulary
    vocab_dict = create_vocab(train_phonetics, valid_phonetics, test_phonetics)
    save_vocab(vocab_dict, 'vocab.json')
    print("Vocabulary saved to vocab.json")

    # Plot phoneme ratios
    # Assuming count_frequency and ratio calculations are handled inside preprocess.py or helpers.py
    # Here, we'll skip plotting for brevity

    # Prepare datasets for model
    def cast_and_rename(dataset):
        return dataset.cast_column("audio_file", Audio(sampling_rate=config['audio']['sampling_rate'])) \
                      .rename_column('audio_file', 'audio')

    train_dataset = cast_and_rename(train_dataset)
    valid_dataset = cast_and_rename(valid_dataset)
    test_dataset  = cast_and_rename(test_dataset)

    vocab_path = str(Path(__file__).resolve().parent.parent / "vocab.json")

    # Tokenizer and Processor
    tokenizer = Wav2Vec2CTCTokenizer(vocab_path, unk_token=config['vocab']['unk_token'],
                                     pad_token=config['vocab']['pad_token'],
                                     word_delimiter_token=config['vocab']['delimiter_token'])

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=config['audio']['sampling_rate'],
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True
    )
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    # Update vocab size in config
    config['model']['vocab_size'] = len(tokenizer)

    # Prepare the dataset for training
    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        batch["input_length"] = len(batch["input_values"])

        with processor.as_target_processor():
            batch["labels"] = processor(batch["phonetic"]).input_ids
        return batch

    def prepare_dataset_batches(batch):
        # Process audio data for a batch
        audio_arrays = [audio["array"] for audio in batch["audio"]]
        sampling_rates = [audio["sampling_rate"] for audio in batch["audio"]]
        batch["input_values"] = [
            processor(array, sampling_rate=sampling_rate).input_values[0]
            for array, sampling_rate in zip(audio_arrays, sampling_rates)
        ]
        batch["input_length"] = [len(input_value) for input_value in batch["input_values"]]

        # Process phonetic labels for a batch
        with processor.as_target_processor():
            batch["labels"] = [processor(phonetic).input_ids for phonetic in batch["phonetic"]]
        return batch


    valid_dataset = valid_dataset.map(prepare_dataset_batches, remove_columns=["audio", "text", "phonetic"], batch_size=1000, batched=True, num_proc=12)
    test_dataset  = test_dataset.map(prepare_dataset_batches, remove_columns=["audio", "text", "phonetic"], batch_size=1000, batched=True, num_proc=12)
    train_dataset = train_dataset.map(prepare_dataset_batches, remove_columns=["audio", "text", "phonetic"], batch_size=1000, batched=True, num_proc=4)


    # Initialize and start training
    train_model(config, processor, train_dataset, valid_dataset)

if __name__ == "__main__":
    main()
