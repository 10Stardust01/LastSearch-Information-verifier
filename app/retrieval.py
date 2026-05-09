from typing import Any

from app.config import Settings
from app.schemas import EvidencePassage


SOURCE_TIERS = ["official", "news"]


def build_hybrid_query(
    claim: str,
    *,
    elser_inference_id: str,
    jina_reranker_inference_id: str,
    size: int = 5,
) -> dict[str, Any]:
    """Build the guarded BM25 + ELSER + RRF + Jina reranker query."""
    source_filter = [{"terms": {"source_tier": SOURCE_TIERS}}]

    bm25_leg = {
        "standard": {
            "query": {
                "function_score": {
                    "query": {
                        "bool": {
                            "must": {
                                "multi_match": {
                                    "query": claim,
                                    "fields": ["title^3", "body"],
                                }
                            },
                            "filter": source_filter,
                        }
                    },
                    "functions": [
                        {
                            "gauss": {
                                "published_date": {
                                    "origin": "now",
                                    "scale": "7d",
                                    "offset": "1d",
                                    "decay": 0.5,
                                }
                            }
                        }
                    ],
                    "boost_mode": "multiply",
                }
            }
        }
    }

    elser_leg = {
        "standard": {
            "query": {
                "bool": {
                    "must": {
                        "sparse_vector": {
                            "field": "body_elser",
                            "inference_id": elser_inference_id,
                            "query": claim,
                        }
                    },
                    "filter": source_filter,
                }
            }
        }
    }

    return {
        "retriever": {
            "text_similarity_reranker": {
                "inference_id": jina_reranker_inference_id,
                "inference_text": claim,
                "field": "body",
                "rank_window_size": 20,
                "retriever": {
                    "rrf": {
                        "window_size": 20,
                        "rank_constant": 60,
                        "retrievers": [bm25_leg, elser_leg],
                    }
                },
            }
        },
        "_source": ["title", "body", "source_name", "source_url", "published_date"],
        "size": size,
    }


class ElasticsearchRetriever:
    def __init__(self, settings: Settings):
        if not settings.elasticsearch_api_key:
            raise ValueError("ELASTICSEARCH_API_KEY is required for live retrieval")

        try:
            from elasticsearch import Elasticsearch
        except ImportError as exc:
            raise RuntimeError("Install live dependencies with: python -m pip install -e .[live]") from exc

        self.index = settings.elasticsearch_index
        self.settings = settings
        self.client = Elasticsearch(
            settings.elasticsearch_url,
            api_key=settings.elasticsearch_api_key,
            request_timeout=30,
        )

    def search(self, claim: str, size: int = 5) -> list[EvidencePassage]:
        query = build_hybrid_query(
            claim,
            elser_inference_id=self.settings.elser_inference_id,
            jina_reranker_inference_id=self.settings.jina_reranker_inference_id,
            size=size,
        )
        response = self.client.search(index=self.index, body=query)
        passages: list[EvidencePassage] = []

        for hit in response.get("hits", {}).get("hits", []):
            source = hit["_source"]
            passages.append(EvidencePassage.model_validate(source))

        return passages
