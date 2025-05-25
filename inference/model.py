from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import torch
import torch.nn.functional as F

def initialize_model():
    base = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model = PeftModel.from_pretrained(base, "inference/model/final/")
    tokenizer = AutoTokenizer.from_pretrained("inference/model/final/")
    return model.eval(), tokenizer

model, tokenizer = initialize_model()

def predict(text: str)->dict:
    vector = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        output = model(**vector)
        logits = output.logits.squeeze()
        probs = F.softmax(logits, dim=-1)
        prob = probs[1].item()  # class 1 = toxic
        toxic = prob > 0.5
    return {"toxic": toxic, "confidence": prob}

if __name__ == "__main__":
    logits = predict("You are a terrible person!")
    print(logits)
