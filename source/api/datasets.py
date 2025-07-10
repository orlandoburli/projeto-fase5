import os

import pandas as pd
import json


def build_dataset():
    data_path = os.curdir + "/data"

    with open(f"{data_path}/prospects.json") as f:
        prospects_data = json.load(f)

    with open(f"{data_path}/applicants.json") as f:
        applicants_data = json.load(f)

    with open(f"{data_path}/positions.json") as f:
        positions_data = json.load(f)

    prospect_records = []
    for job_id, job_data in prospects_data.items():
        for p in job_data["prospects"]:
            record = {
                "job_id": job_id,
                "job_title": job_data.get("titulo", ""),
                "candidate_code": p.get("codigo"),
                "candidate_name": p.get("nome"),
                "status": p.get("situacao_candidado"),
                "application_date": p.get("data_candidatura"),
                "last_update": p.get("ultima_atualizacao"),
                "comment": p.get("comentario", ""),
                "recruiter": p.get("recrutador", "")
            }
            prospect_records.append(record)

    df_prospects = pd.DataFrame(prospect_records)

    applicant_records = []
    for code, app in applicants_data.items():
        record = {
            "candidate_code": code,
            "name": app.get("infos_basicas", {}).get("nome", ""),
            "cv": app.get("cv_pt", ""),
            "education": app.get("formacao_e_idiomas", {}).get("nivel_academico", ""),
            "english_level": app.get("formacao_e_idiomas", {}).get("nivel_ingles", ""),
            "spanish_level": app.get("formacao_e_idiomas", {}).get("nivel_espanhol", ""),
            "job_title": app.get("informacoes_profissionais", {}).get("titulo_profissional", ""),
            "area": app.get("informacoes_profissionais", {}).get("area_atuacao", ""),
            "certifications": app.get("informacoes_profissionais", {}).get("certificacoes", ""),
            "salary_expectation": app.get("informacoes_profissionais", {}).get("remuneracao", "")
        }
        applicant_records.append(record)

    df_applicants = pd.DataFrame(applicant_records)

    position_records = []
    for job_id, data in positions_data.items():
        info = data.get("informacoes_basicas", {})
        perfil = data.get("perfil_vaga", {})
        record = {
            "job_id": job_id,
            "position_title": info.get("titulo_vaga", ""),
            "client": info.get("cliente", ""),
            "location": perfil.get("cidade", ""),
            "seniority": perfil.get("nivel profissional", ""),
            "required_english": perfil.get("nivel_ingles", ""),
            "required_spanish": perfil.get("nivel_espanhol", ""),
            "required_certifications": perfil.get("competencia_tecnicas_e_comportamentais", ""),
            "job_description": perfil.get("principais_atividades", "")
        }
        position_records.append(record)

    df_positions = pd.DataFrame(position_records)

    df_merged = df_prospects.merge(df_applicants, on="candidate_code", how="left")
    df_merged = df_merged.merge(df_positions, on="job_id", how="left")

    return df_merged
