from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agent_utils import (
    OPENAI_API_KEY,
    load_data,
    prepare_texts,
    build_vector_store,
    retrieve_similar,
    calculate_risk,
    generate_analysis,
)

app = FastAPI(title="Change Risk AI Agent API")

index = None
texts = None
embeddings = None


class ChangeRequest(BaseModel):
    change_request: str


@app.on_event("startup")
def load_agent():
    global index, texts, embeddings

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing in .env")

    changes_df, incidents_df = load_data()
    text_rows = prepare_texts(changes_df, incidents_df)
    index, texts, embeddings = build_vector_store(text_rows)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze-change")
def analyze_change(payload: ChangeRequest):
    if not payload.change_request.strip():
        raise HTTPException(status_code=400, detail="change_request is required")

    similar_records = retrieve_similar(
        payload.change_request,
        index,
        texts,
        embeddings,
        top_k=3,
    )

    risk_result = calculate_risk(payload.change_request)

    analysis = generate_analysis(
        payload.change_request,
        similar_records,
        risk_result,
    )

    return {
        "risk_score": risk_result["risk_score"],
        "risk_level": risk_result["risk_level"],
        "similar_records": similar_records,
        "analysis": analysis,
    }