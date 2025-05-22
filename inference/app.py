from fastapi import FastAPI
from pydantic import BaseModel
from inference.model import predict

class RequestBody(BaseModel):
    text: list[str]

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/inference")
async def inference(input: RequestBody):
    return [predict(text) for text in input.text]
