from transformers import Wav2Vec2ForCTC
import torch

def get_model(pretrained_model: str, vocab_size: int, processor, device: str = "cpu") -> Wav2Vec2ForCTC:
    model = Wav2Vec2ForCTC.from_pretrained(
        pretrained_model,
        attention_dropout=0.1,
        layerdrop=0.0,
        feat_proj_dropout=0.0,
        mask_time_prob=0.75, 
        mask_time_length=10,
        mask_feature_prob=0.25,
        mask_feature_length=64,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=vocab_size
    ).to(device)

    model.freeze_feature_encoder()
    return model
