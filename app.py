import streamlit as st
from agent_utils import (
    OPENAI_API_KEY,
    load_data,
    prepare_texts,
    build_vector_store,
    retrieve_similar,
    calculate_risk,
    generate_analysis,
)


@st.cache_resource
def setup():
    changes_df, incidents_df = load_data()
    texts = prepare_texts(changes_df, incidents_df)
    index, text_store, embeddings = build_vector_store(texts)
    return index, text_store, embeddings


st.set_page_config(page_title="AI Change Risk Agent", layout="wide")
st.title("AI Change Risk & Impact Analysis Agent")
st.caption("Analyze deployment changes, estimate risk, retrieve similar incidents, and suggest tests.")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is missing in .env")
    st.stop()

index, texts, embeddings = setup()

change_request = st.text_area(
    "Enter change request",
    value="",
    height=120,
)

if st.button("Analyze Change", type="primary"):
    if not change_request.strip():
        st.warning("Please enter a change request.")
        st.stop()

    with st.spinner("Analyzing..."):
        similar_records = retrieve_similar(change_request, index, texts, embeddings)
        risk_result = calculate_risk(change_request)
        analysis = generate_analysis(change_request, similar_records, risk_result)

    st.subheader("Risk Summary")
    st.write(f"**Risk Level:** {risk_result['risk_level']}")
    st.write(f"**Risk Score:** {risk_result['risk_score']}")
    st.write("**Similar Past Records:**")
    for item in similar_records:
        st.write(f"- {item}")

    st.subheader("AI Analysis")
    st.write(analysis)