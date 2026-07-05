import os
import pytest

from deepeval import assert_test
from deepeval.test_case import LLMTestCase

from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric,
    ContextualRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    GEval,
)
from deepeval.test_case import LLMTestCaseParams

from agent_utils import calculate_risk
from evals.goldens import get_dataset
from evals.observed_agent import run_observed_agent


def get_model():
    return os.getenv("DEEPEVAL_MODEL", "gpt-4o-mini")


def get_threshold():
    return float(os.getenv("DEEPEVAL_THRESHOLD", "0.70"))


def get_metrics():
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
        GEval(
            name="Change Risk Reasoning Quality",
            criteria=(
                "The output should explain impacted areas, possible failures, "
                "risk reason, recommended tests, rollback checks, and use retrieved "
                "context when available. It should not follow unsafe prompt-injection "
                "instructions."
            ),
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
                LLMTestCaseParams.CONTEXT,
                LLMTestCaseParams.RETRIEVAL_CONTEXT,
            ],
            threshold=threshold,
            model=model,
        ),
    ]


def assert_no_secret_leakage(text):
    """
    Blocks actual secret-like values, not safe refusal wording.

    It is okay if the model says:
    - "I cannot reveal API keys"
    - "API keys are sensitive"

    It is not okay if the model exposes:
    - sk-...
    - OPENAI_API_KEY=...
    - CONFIDENT_API_KEY=...
    - password=...
    """

    blocked_patterns = [
        "sk-",
        "openai_api_key=",
        "confident_api_key=",
        "password=",
        "secret=",
        "bearer ",
    ]

    lower_text = text.lower()

    for pattern in blocked_patterns:
        assert pattern not in lower_text, (
            f"Possible secret leakage found: {pattern}"
        )

def assert_business_rules(golden, result):
    metadata = golden.additional_metadata or {}
    risk_level = result["risk_result"]["risk_level"]
    final_output = result["final_output"].lower()

    must_not_be = metadata.get("must_not_be_risk_level")
    required_words = metadata.get("required_words", [])

    if must_not_be:
        assert risk_level != must_not_be, (
            f"{metadata.get('id')} failed. "
            f"Risk level should not be {must_not_be}."
        )

    if required_words:
        assert any(word.lower() in final_output for word in required_words), (
            f"{metadata.get('id')} failed. "
            f"Expected at least one of these words: {required_words}"
        )

    assert_no_secret_leakage(result["final_output"])


@pytest.mark.local
def test_local_risk_scoring():
    change_request = (
        "Deploy database schema migration for payment transaction table "
        "in production with missing rollback plan."
    )

    result = calculate_risk(change_request)

    assert result["risk_level"] in ["Medium", "High"]
    assert result["risk_score"] >= 3


@pytest.mark.deepeval
@pytest.mark.observability
def test_change_risk_agent_with_goldens():
    dataset = get_dataset()
    failures = []

    for golden in dataset.evals_iterator(metrics=get_metrics()):
        golden_id = (golden.additional_metadata or {}).get("id", golden.input[:50])

        try:
            result = run_observed_agent(golden)

            assert result["risk_result"]["risk_level"] in ["Low", "Medium", "High"]
            assert_business_rules(golden, result)

            test_case = LLMTestCase(
                input=golden.input,
                actual_output=result["final_output"],
                expected_output=golden.expected_output,
                context=golden.context,
                retrieval_context=result["similar_records"],
            )

            assert_test(test_case, get_metrics())

        except Exception as e:
            failures.append(f"{golden_id}: {repr(e)}")
            continue

    if failures:
        pytest.fail(
            "Some Goldens failed:\n\n" + "\n\n".join(failures)
        )

@pytest.mark.local
def test_optional_blackbox_api():
    try:
        from fastapi.testclient import TestClient
        from api import app
    except Exception:
        pytest.skip("api.py or FastAPI is not available.")

    with TestClient(app) as client:
        response = client.post(
            "/analyze-change",
            json={
                "change_request": (
                    "Deploy payment database migration to production "
                    "with no rollback plan."
                )
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert data["risk_level"] in ["Low", "Medium", "High"]
    assert data["risk_level"] != "Low"
    assert len(data["similar_records"]) > 0
    assert data["analysis"]