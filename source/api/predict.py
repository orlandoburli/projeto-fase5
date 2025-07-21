from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from database import load_model_from_db
from sqlalchemy import create_engine
import os

model: RandomForestClassifier = None
tfidf: TfidfVectorizer = None


class CandidateInfo(BaseModel):
    candidate_code: str
    name: str
    education: str
    english_level: str
    spanish_level: str
    job_title: str
    area: str
    certifications: str
    salary_expectation: float

class PositionInfo(BaseModel):
    job_id: str
    position_title: str
    client: str
    location: str
    seniority: str
    required_english: str
    required_spanish: str
    required_certifications: str
    job_description: str

class PredictionRequest(BaseModel):
    candidate: CandidateInfo
    position: PositionInfo


def predict_candidate(data: PredictionRequest, engine):
    global model, tfidf

    if model is None or tfidf is None:
        model = load_model_from_db("random_forest_model", engine)
        tfidf = load_model_from_db("tfidf_vectorizer", engine)
        if model is None or tfidf is None:
            raise ValueError("Model or TF-IDF vectorizer not found in database")

    candidate = data.candidate
    position = data.position

    # Pré-processamento dos dados
    english_map = {"Nenhum": 0, "Básico": 1, "Intermediário": 2, "Avançado": 3, "Fluente": 4}
    spanish_map = {"Nenhum": 0, "Básico": 1, "Intermediário": 2, "Avançado": 3, "Fluente": 4}

    def has_cert(cert: Optional[str]) -> int:
        return 0 if cert is None or cert.strip() == "" else 1

    # Unificar dados relevantes do candidato + vaga para gerar TF-IDF
    text_data = " ".join([
        str(candidate.education or ""),
        str(candidate.job_title or ""),
        str(candidate.area or ""),
        str(candidate.certifications or ""),
        str(position.position_title or ""),
        str(position.client or ""),
        str(position.job_description or ""),
        str(position.required_certifications or "")
    ])

    # Criar DataFrame
    row = {
        "english_level_code": english_map.get(candidate.english_level, 0),
        "spanish_level_code": spanish_map.get(candidate.spanish_level, 0),
        "has_certifications": has_cert(candidate.certifications),
        "cv": text_data.lower()
    }
    df = pd.DataFrame([row])

    # Palavras-chave fixas (mantidas para compatibilidade com modelo treinado)
    keywords = ["sap", "sql", "peoplesoft", "aws", "suporte", "rollout", "analista", "consultor"]
    for kw in keywords:
        df[f"kw_{kw}"] = df["cv"].str.contains(kw).fillna(False).astype(int)

    # TF-IDF
    tfidf_matrix = tfidf.transform(df["cv"].fillna(""))
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())
    tfidf_df = tfidf_df.add_prefix("tfidf_")

    df_model = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)

    # Features usadas no modelo
    feature_cols = (
        ["english_level_code", "spanish_level_code", "has_certifications"] +
        [f"kw_{kw}" for kw in keywords] +
        [f"tfidf_{col}" for col in tfidf.get_feature_names_out()]
    )

    model_features = getattr(model, "feature_names_in_", feature_cols)
    valid_features = [f for f in feature_cols if f in model_features]

    X_input = df_model[valid_features]

    # Predição
    match_probability = model.predict_proba(X_input)[:, 1][0]

    return {
        "match_probability": round(float(match_probability), 4),
        "candidate": candidate.name,
        "job_id": position.job_id,
        "position": position.position_title
    }
