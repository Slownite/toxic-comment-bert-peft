from fastapi import FastAPI
from pydantic import BaseModel
from model import predict

class RequestBody(BaseModel):
    text: str

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/inference")
async def inference(input: RequestBody):
    return predict(input.text)
