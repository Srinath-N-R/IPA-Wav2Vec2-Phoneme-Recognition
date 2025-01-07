from evaluate import load

wer_metric = load("wer")
cer_metric = load("cer")

def compute_metrics(pred, tokenizer):
    pred_logits = pred.predictions
    pred_ids = pred_logits.argmax(axis=-1)

    # Replace -100 with pad_token_id
    labels = pred.label_ids
    labels[labels == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids)
    label_str = tokenizer.batch_decode(labels, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer, "cer": cer}
