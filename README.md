# 🧠 Decision AI Recruitment Classifier

Este projeto é um pipeline completo de Machine Learning para classificação de candidatos em processos seletivos, com métricas expostas via Prometheus e deploy via Docker.

## 🚀 Como Executar

### Pré-requisitos

- Docker
- Docker Compose

### Subir o ambiente

```bash
docker-compose up --build
```

### Acessos

- API FastAPI: [http://localhost:8000/docs](http://localhost:8000/docs)
- Métricas Prometheus: [http://localhost:9090](http://localhost:9090)

---



## 🔍 Métricas

As seguintes métricas são expostas no endpoint `/metrics`:

- `predict_requests_total`: total de requisições ao endpoint de predição
- `predict_request_duration_seconds`: tempo de resposta das predições
- `model_accuracy_score`: acurácia do modelo atual
- `model_auc_score`: ROC AUC score do modelo atual

Você pode visualizar as métricas em tempo real no Prometheus.

---

## 📊 Treinamento do Modelo

O modelo é treinado e salvo usando a requisição ao endpoint `/train`. O treinamento é feito com base nos dados de candidatos e posições disponíveis no arquivo `data/candidates.csv`.:


- As features utilizadas: `experience_years`, `has_certifications`, `technical_score`, `cultural_fit_score`
- O modelo treinado é um `RandomForestClassifier`
- O modelo é salvo no banco de dados e exposto via API


Para treinar o modelo, execute a seguinte requisição:
```shell
curl 'http://localhost:8000/train'
```

---


## 📈 Exemplos de Requisição

### Endpoint de predição:


```shell
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     --data '
{
  "candidate": {
    "candidate_code": "123",
    "name": "João Silva",
    "education": "Bacharel em Sistemas de Informação",
    "english_level": "Avançado",
    "spanish_level": "Intermediário",
    "job_title": "Analista SAP",
    "area": "TI",
    "certifications": "SAP Certified",
    "salary_expectation": 8500.0
  },
  "position": {
    "job_id": "456",
    "position_title": "Consultor SAP Sênior",
    "client": "Acme Corp",
    "location": "São Paulo",
    "seniority": "Sênior",
    "required_english": "Avançado",
    "required_spanish": "Intermediário",
    "required_certifications": "SAP",
    "job_description": "Responsável pela implementação SAP ECC, com foco em rollout e suporte."
  }
}'
```

Resposta esperada:
```json
{
    "match_probability": 0.0946,
    "candidate": "João Silva",
    "job_id": "456",
    "position": "Consultor SAP Sênior"
}
```