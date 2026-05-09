import logging

from app.bedrock import BedrockVerdictGenerator, fallback_unverified
from app.config import get_settings
from app.retrieval import ElasticsearchRetriever
from app.schemas import VerdictResponse
from app.security import UnsafeClaimError, sanitize_claim

logger = logging.getLogger(__name__)


class FactCheckAgent:
    def __init__(self):
        logger.info("Initializing FactCheckAgent")
        self.settings = get_settings()
        self.retriever: ElasticsearchRetriever | None = None
        self.verdict_generator: BedrockVerdictGenerator | None = None

    def _get_retriever(self) -> ElasticsearchRetriever:
        if self.retriever is None:
            self.retriever = ElasticsearchRetriever(self.settings)
        return self.retriever

    def _get_verdict_generator(self) -> BedrockVerdictGenerator:
        if self.verdict_generator is None:
            self.verdict_generator = BedrockVerdictGenerator(self.settings)
        return self.verdict_generator

    def check(self, raw_claim: str) -> VerdictResponse:
        logger.info(f"Processing claim: {raw_claim[:100]}...")
        try:
            claim = sanitize_claim(raw_claim)
            logger.info("Claim sanitized successfully")
        except UnsafeClaimError as e:
            logger.warning(f"Claim rejected as unsafe: {e}")
            return fallback_unverified(
                "The submitted text contained instruction-like content, so it was not used."
            )

        evidence = self._get_retriever().search(claim)
        logger.info(f"Retrieved {len(evidence)} evidence passages")

        if not evidence:
            logger.info("No evidence found, returning unverified")
            return fallback_unverified("No verified source in the corpus clearly addressed the claim.")

        result = self._get_verdict_generator().generate(claim, evidence)
        logger.info(f"Generated verdict: {result.verdict}")
        return result
