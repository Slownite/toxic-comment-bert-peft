from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import torch
def initialize_model():
    base = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1)
    model = PeftModel.from_pretrained(base, "inference/model/final/")
    tokenizer = AutoTokenizer.from_pretrained("inference/model/final/")
    return model.eval(), tokenizer

model, tokenizer = initialize_model()

def predict(text: str)->dict:
    vector = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        output = model(**vector)
        logits = output.logits.squeeze()
        prob = torch.sigmoid(logits).item()
        toxic = prob > 0.5
    return {"toxic": toxic, "confidence": prob}

if __name__ == "__main__":
    logits = predict("you are beautiful")
    print(logits)
