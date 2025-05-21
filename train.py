from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, concatenate_datasets
from functools import partial
import evaluate
import numpy as np
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
import torch

# Tokenize the input text
def tokenize(tokenizer, data):
    return tokenizer(data["text"], padding="max_length", truncation=True)

# Convert float toxicity scores into binary labels
def preprocess_labels(example):
    # print(int(example["labels"] > 0.5))
    return {"labels": int((example["labels"]) > 0.5)}

# Prepare and preprocess the dataset
def prepare_dataset():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizing = partial(tokenize, tokenizer)
    ds = load_dataset("civil_comments")
    dataset = ds.rename_column("toxicity", "labels")
    dataset = dataset.map(preprocess_labels)
    dataset = dataset.map(tokenizing, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"], dtype=torch.long)
    positive = dataset["train"].filter(lambda x: x["labels"] == 1)
    negative = dataset["train"].filter(lambda x: x["labels"] == 0).select(range(len(positive)))
    balanced_train = concatenate_datasets([positive, negative]).shuffle(seed=42)
    return balanced_train, dataset, tokenizer

# Metric function for binary classification
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

# Training pipeline
def train():
    # LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        
    )

    # Load model and apply LoRA for binary classification
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2,
        problem_type="single_label_classification"
    )
    model = get_peft_model(model, lora_config)

    # Prepare data and metrics
    balanced_train, dataset, tokenizer = prepare_dataset()

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="model",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1"
    )

    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=balanced_train,
        eval_dataset=dataset["test"].shuffle(seed=42).select(range(5000)),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Start training
    trainer.train()
    model.save_pretrained("inference/model/final")
    tokenizer.save_pretrained("inference/model/final")

# Entry point
def main():
    train()

if __name__ == '__main__':
    main()
