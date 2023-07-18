from helpers.text_embeddings import TextEmbeddingsHelper
from helpers.latents import LatentsHelper
from helpers.cache import CacheHelper
from helpers.auto_encoder import AutoEncoderHelper

from helpers.pipeline import Pipe

import torch

from fastapi import FastAPI
from fastapi.exceptions import HTTPException
from fastapi import status, Depends, Request
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

app = FastAPI(openapi_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cpu")

text_embeddings_helper = TextEmbeddingsHelper(device)
latents_helper = LatentsHelper(device)
auto_encoder_helper = AutoEncoderHelper(device)

pipeline = Pipe(text_embeddings_helper, latents_helper, auto_encoder_helper)


class PromptModel(BaseModel):
    prompt: str


@app.get("/")
async def root():
    return {"message": f"Hello from team Invictus!"}


@app.post("/inference")
async def inference(prompt: PromptModel):
    imgs, semi_inp = pipeline.forward(prompt.prompt)

    print("Done")
