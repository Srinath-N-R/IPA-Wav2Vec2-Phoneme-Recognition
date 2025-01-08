# Wav2Vec2-Based Speech Recognition

## Overview

This repository contains an end-to-end pipeline for training and evaluating a speech recognition model using [Wav2Vec2](https://huggingface.co/docs/transformers/en/model_doc/wav2vec2). It leverages various tools, including the Hugging Face Transformers library, to preprocess data, build vocabulary, and train a model capable of recognizing phonemes from audio input.

The pipeline is modular, enabling customization for different datasets and configurations. It includes preprocessing, phoneme vocabulary generation, training, evaluation, and inference.

This project specifically focuses on IPA-based phoneme recognition.

---

## Features

- **Preprocessing:** Efficient audio and text processing for training datasets.
- **Vocabulary Generation:** Automatic phoneme vocabulary creation based on training data.
- **Model Training:** Fine-tuning Wav2Vec2 with custom phoneme vocabularies.
- **Evaluation Metrics:** Word Error Rate (WER) and Character Error Rate (CER) computation.
- **Dataset Caching:** Intermediate datasets cached for reuse.
- **Visualization:** Phoneme distribution plotting.
- **Scalability:** Supports multi-GPU training and large datasets.

---

## Project Structure

- **`collator.py`:** Custom data collator for handling CTC padding during training.
- **`metrics.py`:** Functions to compute WER and CER for model evaluation.
- **`train.py`:** Main script for configuring and training the model.
- **`config.py`:** Handles loading YAML configurations for the pipeline.
- **`helpers.py`:** Utility functions for phoneme processing and plotting.
- **`wav2vec2_ctc.py`:** Defines model initialization and configuration.
- **`run_training.py`:** Entry point for the training pipeline.
- **`loader.py`:** Dataset loader with support for multi-threaded file processing.
- **`preprocess.py`:** Functions for vocabulary creation and phoneme mapping.
- **`vocab.py`:** Utilities to save and load vocabularies.

---

## Setup

### Installation

To install the project, run:
```bash
pip install -e .
```

### Prerequisites

- Python 3.8+
- CUDA-enabled GPU (optional but recommended)

---

## Usage

### Configuration

Update the `configs/config.yaml` file to specify dataset paths, model parameters, and training hyperparameters.

### Training

To start training the model:
```bash
python scripts/run_training.py
```

### Evaluation

Evaluate the trained model using:
```bash
python scripts/eval.py
```

---

## Key Components

### Model Training

The model is fine-tuned with CTC loss, using the `Trainer` API from Hugging Face. It incorporates gradient accumulation, mixed-precision training, and early stopping.

### Data Collation

Custom collator ensures proper padding for input features and target labels during batch processing (`collator.py`).

### Metrics

WER and CER are computed using the `evaluate` library, ensuring detailed insights into model performance (`metrics.py`).

---

## Customization

- **Vocabulary:** Edit `vocab.py` for custom phoneme mappings.
- **Data:** Update `config.yaml` to point to new dataset folders.
- **Model:** Modify `wav2vec2_ctc.py` for model-specific tweaks.

---

## Future Improvements

- Implement data augmentation for robust training.
- Add support for real-time inference.
- Extend the pipeline for multilingual datasets.

---

## License

This project is licensed under the MIT License.

---

## Contributions

Contributions, issues, and feature requests are welcome! Please create a pull request or open an issue for discussion.
