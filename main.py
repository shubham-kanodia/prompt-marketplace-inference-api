from helpers.text_embeddings import TextEmbeddingsHelper
from helpers.latents import LatentsHelper
from helpers.cache import CacheHelper
from helpers.auto_encoder import AutoEncoderHelper

from helpers.pipeline import Pipe

import torch
import pickle

from fastapi import FastAPI
from fastapi.exceptions import HTTPException
from fastapi import status, Depends, Request
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cpu"

text_embeddings_helper = TextEmbeddingsHelper(device)
latents_helper = LatentsHelper(device)
auto_encoder_helper = AutoEncoderHelper(device)

pipeline = Pipe(text_embeddings_helper, latents_helper, auto_encoder_helper)


class PromptModel(BaseModel):
    prompt: str


@app.get("/")
async def root():
    return {"message": f"Hello from team Invictus!"}


import base64


def tensor_to_base64(arr):
    bt = str(arr.tolist()).encode("utf-8")
    encoded_bytes = base64.b64encode(bt)
    encoded_string = encoded_bytes.decode('utf-8')

    return encoded_string

@app.post("/inference")
async def inference(prompt: PromptModel):
    # imgs, semi_inp = pipeline.forward(prompt.prompt)

    imgs = pickle.load(open("data/new-cache/imgs.pkl", "rb"))
    semi_inp = pickle.load(open("data/new-cache/semi_inp.pkl", "rb"))

    return {
        "imgs": imgs.tolist(),
        "semi_inp": semi_inp.tolist()
    }
