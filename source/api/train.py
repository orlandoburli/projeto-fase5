import pandas as pd
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sqlalchemy.engine.base import Engine

from database import save_model_to_db


def train_dataset(df: DataFrame, engine: Engine):
    df["target"] = df["status"].apply(lambda x: 1 if "Contratado" in str(x) else 0)

    # Nível de inglês → numérico
    english_map = {"Nenhum": 0, "Básico": 1, "Intermediário": 2, "Avançado": 3, "Fluente": 4}
    df["english_level_code"] = df["english_level"].map(english_map).fillna(0).astype(int)

    # Certificações → binário
    df["has_certifications"] = df["certifications"].apply(lambda x: 0 if pd.isna(x) or x.strip() == "" else 1)

    # Palavras-chave no currículo
    keywords = ["sap", "sql", "peoplesoft", "aws", "suporte", "rollout", "analista", "consultor"]
    for kw in keywords:
        df[f"kw_{kw}"] = df["cv"].str.lower().str.contains(kw).fillna(False).astype(int)

    # TF-IDF do currículo
    tfidf = TfidfVectorizer(max_features=20)
    tfidf_matrix = tfidf.fit_transform(df["cv"].fillna(""))
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"tfidf_{w}" for w in tfidf.get_feature_names_out()])

    # Juntar no dataframe final
    df_model = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)

    # --- 4. Selecionar features ---
    feature_cols = (
        ["english_level_code", "has_certifications"] +
        [f"kw_{kw}" for kw in keywords] +
        list(tfidf_df.columns)
    )

    X = df_model[feature_cols]
    y = df_model["target"]

    # --- 5. Treinar modelo ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # --- 6. Avaliar ---
    y_pred = model.predict(X_test)
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))

    # --- 7. Salvar modelo e vetorizador ---
    save_model_to_db(model, "model", engine)
    save_model_to_db(tfidf, "tfidf_vectorizer", engine)

