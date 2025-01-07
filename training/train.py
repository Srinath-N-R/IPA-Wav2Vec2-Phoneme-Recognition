from transformers import Trainer, TrainingArguments
from models.wav2vec2_ctc import get_model
from training.metrics import compute_metrics
from training.collator import DataCollatorCTCWithPadding
import torch
from transformers import EarlyStoppingCallback
from pathlib import Path

def train_model(config, processor, train_dataset, valid_dataset):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create model
    vocab_size = config['model']['vocab_size']
    model = get_model(
        pretrained_model=config['model']['pretrained_model'],
        vocab_size=vocab_size,
        processor=processor,
        device=device
    )

    ds_config = str(Path(__file__).resolve().parent.parent / "configs/dp_config.json")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config['training']['output_dir'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        eval_strategy=config['training']['evaluation_strategy'],
        gradient_checkpointing=config['training']['gradient_checkpointing'],
        fp16=config['training']['fp16'],
        num_train_epochs=config['training']['epochs'],
        save_steps=config['training']['save_steps'],
        eval_steps=config['training']['eval_steps'],
        logging_steps=config['training']['logging_steps'],
        learning_rate=float(config['training']['learning_rate']),
        warmup_steps=config['training']['warmup_steps'],
        save_total_limit=config['training']['save_total_limit'],
        load_best_model_at_end=config['training']['load_best_model_at_end'],
        deepspeed=ds_config
    )

    # Data collator
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=lambda pred: compute_metrics(pred, processor.tokenizer),
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=processor.feature_extractor,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=config['training']['patience'])
        ],
    )

    # Start training
    trainer.train()
