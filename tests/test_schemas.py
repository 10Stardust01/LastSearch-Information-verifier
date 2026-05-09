import pytest

from app.schemas import SchemaValidationError, VerdictResponse


def test_decisive_verdict_requires_citation():
    with pytest.raises(SchemaValidationError):
        VerdictResponse.model_validate(
            {
                "verdict": "SUPPORTED",
                "confidence": 0.8,
                "reasoning": "The retrieved source confirms the claim.",
                "citations": [],
            }
        )


def test_unverified_can_have_no_citations():
    verdict = VerdictResponse.model_validate(
        {
            "verdict": "UNVERIFIED",
            "confidence": 0.0,
            "reasoning": "No verified source clearly addressed the claim.",
            "citations": [],
        }
    )

    assert verdict.verdict == "UNVERIFIED"
