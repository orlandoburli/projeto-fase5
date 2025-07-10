import datetime
import logging
import os
import tempfile

import joblib
from sqlalchemy import create_engine, Column, Integer, String, LargeBinary, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

POSTGRES_USERNAME = os.environ.get("POSTGRES_USERNAME")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD")
POSTGRES_HOST = os.environ.get("POSTGRES_HOST")
POSTGRES_PORT = os.environ.get("POSTGRES_PORT")
POSTGRES_DB = os.environ.get("POSTGRES_DB")

# PostgreSQL database connection string
DB_URL = f"postgresql://{POSTGRES_USERNAME}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

Base = declarative_base()


class MLModel(Base):
    __tablename__ = 'ml_models'
    
    id = Column(Integer, primary_key=True)
    model_name = Column(String)
    model_data = Column(LargeBinary)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


def init_db():
    # Set up logging
    logging.basicConfig()
    # Log all SQL statements
    logging.getLogger('sqlalchemy.engine').setLevel(logging.DEBUG)

    engine = create_engine(DB_URL)
    Base.metadata.create_all(engine)
    return engine


def save_model_to_db(model, model_name, engine):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
        joblib.dump(model, tmp.name)
        with open(tmp.name, 'rb') as f:
            model_data = f.read()
        os.unlink(tmp.name)

    Session = sessionmaker(bind=engine)
    session = Session()

    ml_model = MLModel(
        model_name=model_name,
        model_data=model_data,
    )

    try:
        session.add(ml_model)
        session.commit()
        print(f"Model for {model_name} saved to database successfully")
    except Exception as e:
        session.rollback()
        print(f"Error saving model to database: {e}")
    finally:
        session.close()



def load_model_from_db(model_name, engine):
    from tensorflow.keras.models import load_model

    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Retorna o ultimo modelo salvo para a empresa
        model_record = session.query(MLModel) \
            .filter(MLModel.model_name == model_name) \
            .order_by(MLModel.created_at.desc()) \
            .first()

        print(f"Model: ${model_record}")

        if model_record is None:
            return None

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
            tmp.write(model_record.model_data)
            tmp.flush()
            model = load_model(tmp.name)
            os.unlink(tmp.name)

        return model

    except Exception as e:
        print(f"Erro carregando modelo do banco de dados: {e}")
        return None
    finally:
        session.close()
