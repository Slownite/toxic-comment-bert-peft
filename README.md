# 🛡️ Deploying a Hugging Face PEFT Model with Accelerate on AWS SageMaker

## 🚀 Overview

This project demonstrates how to fine-tune a BERT model using Parameter-Efficient Fine-Tuning (PEFT) techniques with Hugging Face's `transformers`, `accelerate`, and `peft` libraries, and deploy it as a scalable web service on AWS SageMaker.

The model is trained to perform **Toxic Comment Classification**, detecting harmful or offensive content in user-generated text.

---

## 🎯 Goals

- Fine-tune a BERT model for toxic comment classification  
- Use PEFT (LoRA) for parameter-efficient training  
- Train using Hugging Face Accelerate  
- Serve the model via a FastAPI app  
- Containerize the app using Docker  
- Deploy to AWS SageMaker (or EC2)  
- Monitor with AWS CloudWatch  
- Document workflows in GitHub + Colab + blog  

---

## 🧠 Model

- **Base Model:** [`bert-base-uncased`](https://huggingface.co/bert-base-uncased)  
- **Task:** Binary text classification (toxic / non-toxic)  
- **Fine-tuning Technique:** LoRA (Low-Rank Adaptation) via Hugging Face `peft`  
- **Frameworks:** `transformers`, `datasets`, `accelerate`, `peft`, `FastAPI`  

---

## 📊 Dataset

- **Source:** [Civil Comments](https://huggingface.co/datasets/civil_comments) via Hugging Face Datasets  
- **Size:** ~400K user comments  
- **Labels:** Binary (toxic or not)  
- **Preprocessing:** Tokenization with `BertTokenizer`, truncation/padding, label encoding  

---

## 🏗️ Project Structure

```
toxic-comment-bert-peft/
│
├── train.py                    # Fine-tuning script using Accelerate + PEFT
├── inference/
│   ├── app.py                  # FastAPI server exposing /predict
│   └── model.py                # Loads tokenizer and model, handles predictions
├── Dockerfile                  # Container config for serving the FastAPI app
├── requirements.txt            # Python dependencies
├── sagemaker/
│   ├── deploy.py               # Script to deploy to AWS SageMaker
│   └── config/                 # IAM roles, endpoint configs, etc.
├── monitoring/
│   └── cloudwatch_setup.py     # Set up logging and alarms with AWS CloudWatch
├── notebooks/
│   └── inference_demo.ipynb    # Colab notebook for live inference demo
├── README.md                   # Project documentation
└── .gitignore
```

---

## 🧪 Training

```bash
accelerate launch train.py
```

This command runs distributed training using Hugging Face Accelerate. PEFT with LoRA reduces memory consumption and training time while preserving model performance.

---

## 🌐 Serving the Model (Locally)

```bash
uvicorn inference.app:app --host 0.0.0.0 --port 8000 --reload
```

Once launched, send a POST request to `/predict` with JSON input:
```json
{
  "text": "You are a terrible person!"
}
```

Response:
```json
{
  "toxic": true,
  "confidence": 0.92
}
```

---

## 🐳 Docker Container

To build and run the model server in a container:

```bash
docker build -t toxic-api .
docker run -p 8000:8000 toxic-api
```

---

## ☁️ AWS SageMaker Deployment (Optional)

This project supports cloud deployment to **Amazon SageMaker** via the `sagemaker/` scripts. Steps include:

1. Upload the model to S3  
2. Create a SageMaker inference endpoint  
3. Deploy the Docker image  
4. Monitor usage with CloudWatch  

---

## 📈 Monitoring with AWS CloudWatch

- Automatically tracks:
  - Request latency
  - Error rate
  - Invocations per second
- Custom logs include input samples and predictions for traceability

---

## 📓 Colab / Hugging Face Spaces

A Colab notebook is provided to:

- Load the fine-tuned model from Hugging Face Hub or S3  
- Run live predictions on user text  
- Compare toxic vs non-toxic outputs  
- Visualize confidence scores and embeddings  

> Optionally, a Hugging Face Space demo can be created using Gradio for a web UI.

---

## 🧠 Extensions / Future Work

- Multi-label toxicity (threat, insult, obscene, etc.)
- Support multilingual classification with `bert-base-multilingual-cased`
- Integrate a frontend with Streamlit or Gradio
- Add unit testing and CI/CD (GitHub Actions)

---

## ✍️ Author

**Samuel Diop**  
- 🔗 [LinkedIn](https://www.linkedin.com/in/samuel-diop/)  
- 💻 [Portfolio](http://samueldiop.com)  
- 🧠 [GitHub](https://github.com/Slownite)

---

## 📄 License

MIT License. Feel free to reuse and contribute.
