import os

from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric,
    ContextualRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)
from deepeval.test_case import LLMTestCase
from deepeval.tracing import observe, update_current_span, update_current_trace

from agent_utils import (
    load_data,
    prepare_texts,
    build_vector_store,
    retrieve_similar,
    calculate_risk,
    generate_analysis,
)


def get_model():
    return os.getenv("DEEPEVAL_MODEL", "gpt-4o-mini")


def get_threshold():
    return float(os.getenv("DEEPEVAL_THRESHOLD", "0.70"))


def retriever_metrics():
    model = get_model()
    threshold = get_threshold()

    return [
        ContextualRelevancyMetric(
            threshold=threshold,
            model=model,
            include_reason=True,
        ),
        ContextualPrecisionMetric(
            threshold=threshold,
            model=model,
            include_reason=True,
        ),
        ContextualRecallMetric(
            threshold=threshold,
            model=model,
            include_reason=True,
        ),
    ]


def llm_metrics():
    model = get_model()
    threshold = get_threshold()

    return [
        AnswerRelevancyMetric(
            threshold=threshold,
            model=model,
            include_reason=True,
        ),
        FaithfulnessMetric(
            threshold=threshold,
            model=model,
            include_reason=True,
        ),
        HallucinationMetric(
            threshold=0.30,
            model=model,
            include_reason=True,
        ),
    ]


def format_output(risk_result, similar_records, analysis):
    return f"""
Risk Level: {risk_result["risk_level"]}
Risk Score: {risk_result["risk_score"]}

Similar Records:
{chr(10).join(similar_records)}

Analysis:
{analysis}
""".strip()


@observe(type="retriever", metrics=retriever_metrics())
def retrieve_records(change_request, index, text_store, embeddings, golden):
    similar_records = retrieve_similar(
        change_request,
        index,
        text_store,
        embeddings,
        top_k=3,
    )

    update_current_span(
        test_case=LLMTestCase(
            input=change_request,
            actual_output="\n".join(similar_records),
            expected_output=golden.expected_output,
            context=golden.context,
            retrieval_context=similar_records,
        )
    )

    return similar_records


@observe(type="tool")
def score_risk(change_request):
    risk_result = calculate_risk(change_request)

    update_current_span(
        test_case=LLMTestCase(
            input=change_request,
            actual_output=str(risk_result),
        )
    )

    return risk_result


@observe(type="llm", metrics=llm_metrics())
def create_analysis(change_request, similar_records, risk_result, golden):
    analysis = generate_analysis(
        change_request,
        similar_records,
        risk_result,
    )

    update_current_span(
        test_case=LLMTestCase(
            input=change_request,
            actual_output=analysis,
            expected_output=golden.expected_output,
            context=golden.context,
            retrieval_context=similar_records,
        )
    )

    return analysis


@observe(type="agent")
def run_observed_agent(golden):
    change_request = golden.input

    changes_df, incidents_df = load_data()
    texts = prepare_texts(changes_df, incidents_df)
    index, text_store, embeddings = build_vector_store(texts)

    similar_records = retrieve_records(
        change_request,
        index,
        text_store,
        embeddings,
        golden,
    )

    risk_result = score_risk(change_request)

    analysis = create_analysis(
        change_request,
        similar_records,
        risk_result,
        golden,
    )

    final_output = format_output(
        risk_result=risk_result,
        similar_records=similar_records,
        analysis=analysis,
    )

    update_current_trace(
        test_case=LLMTestCase(
            input=golden.input,
            actual_output=final_output,
            expected_output=golden.expected_output,
            context=golden.context,
            retrieval_context=similar_records,
        )
    )

    return {
        "risk_result": risk_result,
        "similar_records": similar_records,
        "analysis": analysis,
        "final_output": final_output,
    }