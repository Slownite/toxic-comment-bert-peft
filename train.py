from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from functools import partial
import evaluate
import numpy as np
from peft import LoraConfig, TaskType, get_peft_model

# Tokenize the input text
def tokenize(tokenizer, data):
    return tokenizer(data["text"], padding="max_length", truncation=True)

# Keep toxicity scores as float labels
def preprocess_labels(example):
    return {"labels": float(example["labels"])}

# Prepare and preprocess the dataset
def prepare_dataset():
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    tokenizing = partial(tokenize, tokenizer)
    ds = load_dataset("google/civil_comments")
    dataset = ds.rename_column("toxicity", "labels")
    dataset = dataset.map(preprocess_labels)
    dataset = dataset.map(tokenizing, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return dataset, tokenizer

# Metric function for evaluation
def compute_metrics(metric, eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.squeeze()
    return metric.compute(predictions=predictions, references=labels)

# Training pipeline
def train():
    # LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )

    # Load model and apply LoRA for regression
    model = AutoModelForSequenceClassification.from_pretrained(
        "google-bert/bert-base-uncased",
        num_labels=1,
        problem_type="regression"
    )
    model = get_peft_model(model, lora_config)

    # Prepare data and metrics
    dataset, tokenizer = prepare_dataset()
    metrics = evaluate.load("mse")
    p_compute_metrics = partial(compute_metrics, metrics)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="model",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"].shuffle(seed=42).select(range(1000)),
        eval_dataset=dataset["test"].shuffle(seed=42).select(range(1000)),
        tokenizer=tokenizer,
        compute_metrics=p_compute_metrics,
    )

    # Start training
    trainer.train()

# Entry point
def main():
    train()

if __name__ == '__main__':
    main()
