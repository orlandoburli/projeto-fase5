

Train model
```shell
curl 'http://localhost:8000/train'
```

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