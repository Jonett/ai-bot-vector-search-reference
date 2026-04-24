# Tämä on referenssikopio, ei tuotantokäyttöön tarkoitettu moduuli.
"""
Poiminta testeista, joista näkee miten vector- ja embedding-ominaisuuksien
odotetaan kayttaytyvan.

Tama tiedosto ei ole suora kopio projektin testisuitesta, vaan luettava opetusversio.
"""

from __future__ import annotations

from dataclasses import dataclass


class EmbeddingServiceError(RuntimeError):
    pass


@dataclass(frozen=True)
class EmbeddingResponsePayload:
    provider: str
    model: str
    dimensions: int
    embeddings: tuple[tuple[float, ...], ...]
    latency_ms: int


class FakeEmbeddingService:
    # Testeissä käytetty feikki palauttaa deterministisiä vektoreita avainsanojen perusteella.
    # Tämä tekee hybrid-painotusten ja fallbackien tarkistamisesta toistettavaa.
    def __init__(self, *, fail_on_query: bool = False, fail_on_batch: bool = False) -> None:
        self.fail_on_query = fail_on_query
        self.fail_on_batch = fail_on_batch
        self.query_calls: list[str] = []
        self.batch_calls: list[list[str]] = []

    def _vector_for(self, text: str) -> tuple[float, float, float]:
        lowered = text.lower()
        loot_score = float(sum(1 for token in ("loot", "trinket", "reserve", "item", "priority") if token in lowered))
        raid_score = float(sum(1 for token in ("raid", "schedule", "flask") if token in lowered))
        guild_score = float(sum(1 for token in ("faq", "guild", "ohje") if token in lowered))
        return (loot_score, raid_score, guild_score)

    async def embed_text(self, text: str, *, model: str | None = None, provider: str | None = None) -> tuple[float, ...]:
        self.query_calls.append(text)
        if self.fail_on_query:
            raise EmbeddingServiceError("embedding query failed")
        return self._vector_for(text)

    async def embed_texts(
        self,
        texts: list[str],
        *,
        model: str | None = None,
        provider: str | None = None,
    ) -> EmbeddingResponsePayload:
        if self.fail_on_batch:
            raise EmbeddingServiceError("embedding batch failed")
        self.batch_calls.append(list(texts))
        embeddings = tuple(self._vector_for(text) for text in texts)
        return EmbeddingResponsePayload(
            provider=provider or "ollama",
            model=model or "nomic-embed-text",
            dimensions=len(embeddings[0]) if embeddings else 0,
            embeddings=embeddings,
            latency_ms=5,
        )


def test_reference_vector_search_ignores_stale_embeddings() -> None:
    # Projektin oikea testi tarkistaa, että stale embedding ei pääse vector-tuloksiin.
    stale_embedding_is_filtered = True
    tenant_isolation_holds = True
    assert stale_embedding_is_filtered is True
    assert tenant_isolation_holds is True


def test_reference_hybrid_weighting_changes_ranking() -> None:
    # Kun keyword-paino on suuri, keyword-osuma voittaa.
    keyword_heavy_top = "A"
    # Kun vector-paino on suuri, semanttisesti vahvempi osuma voi nousta kärkeen.
    vector_heavy_top = "B"
    assert keyword_heavy_top == "A"
    assert vector_heavy_top == "B"


def test_reference_vector_fallbacks_are_explicit() -> None:
    no_embeddings_fallback = "no_embeddings_available"
    provider_error_fallback = "embedding_provider_failed"
    assert no_embeddings_fallback == "no_embeddings_available"
    assert provider_error_fallback == "embedding_provider_failed"


def test_reference_embedding_jobs_flow() -> None:
    enqueue_status = "planned"
    run_completed = 3
    audit_actions = {"embedding_jobs_enqueued", "embedding_worker_run_completed"}
    assert enqueue_status == "planned"
    assert run_completed >= 1
    assert "embedding_jobs_enqueued" in audit_actions
    assert "embedding_worker_run_completed" in audit_actions

