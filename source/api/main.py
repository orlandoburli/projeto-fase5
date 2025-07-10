import os

import pandas as pd
from fastapi import FastAPI

from prometheus_fastapi_instrumentator import Instrumentator

from database import init_db
from datasets import build_dataset
from train import train_dataset


app = FastAPI()

Instrumentator().instrument(app).expose(app)

pd.set_option('future.no_silent_downcasting', True)

POSTGRES_USERNAME = os.environ.get("POSTGRES_USERNAME")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD")
POSTGRES_HOST = os.environ.get("POSTGRES_HOST")
POSTGRES_PORT = os.environ.get("POSTGRES_PORT")
POSTGRES_DB = os.environ.get("POSTGRES_DB")

# PostgreSQL database connection string
DB_URL = f"postgresql://{POSTGRES_USERNAME}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

@app.get("/")
def hello():
    return {"status": "ok"}


@app.get("/train")
def train():
    engine = init_db()

    df = build_dataset()
    train_dataset(df, engine)
    return {"status": "ok", "details": "Dataset trained successfully!"}
