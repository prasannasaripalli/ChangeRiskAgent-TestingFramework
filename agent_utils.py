import os
import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def load_data():
    changes_df = pd.read_csv("data/test_changerequests.csv")
    incidents_df = pd.read_csv("data/test_incidents.csv")
    return changes_df, incidents_df


def prepare_texts(changes_df, incidents_df):
    texts = []

    for _, row in changes_df.iterrows():
        texts.append(
            f"CHANGE | service={row['service']} | type={row['change_type']} | "
            f"description={row['description']} | risk={row['risk_level']}"
        )

    for _, row in incidents_df.iterrows():
        texts.append(
            f"INCIDENT | summary={row['incident_summary']} | "
            f"root_cause={row['root_cause']} | resolution={row['resolution']}"
        )

    return texts


def build_vector_store(texts):
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vectors = embeddings.embed_documents(texts)
    vectors_np = np.array(vectors).astype("float32")

    index = faiss.IndexFlatL2(vectors_np.shape[1])
    index.add(vectors_np)
    return index, texts, embeddings


def retrieve_similar(change_request, index, texts, embeddings, top_k=3):
    query_vector = embeddings.embed_query(change_request)
    query_np = np.array([query_vector]).astype("float32")
    distances, indices = index.search(query_np, top_k)
    return [texts[i] for i in indices[0] if i < len(texts)]


def calculate_risk(change_request):
    text = change_request.lower()
    score = 0

    if "database" in text or "schema" in text or "migration" in text:
        score += 3
        
    if "payment" in text or "auth" in text or "login" in text:
        score += 2

    if "deploy" in text or "upgrade" in text or "version" in text:
        score += 1

    if score >= 6:
        level = "High"
    elif score >= 3:
        level = "Medium"
    else:
        level = "Low"

    return {"risk_score": score, "risk_level": level}


def generate_analysis(change_request, similar_records, risk_result):
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model=OPENAI_MODEL, temperature=0)

    prompt = f"""
You are an AI assistant for software change risk analysis.

User change request:
{change_request}

Similar past records:
{chr(10).join(similar_records)}

Risk level: {risk_result['risk_level']}
Risk score: {risk_result['risk_score']}

Provide:
1. impacted areas
2. possible failures
3. recommended tests
4. rollback checks

Keep the answer concise and practical.
"""

    response = llm.invoke(prompt)
    return response.content