# Tämä on referenssikopio, ei tuotantokäyttöön tarkoitettu moduuli.
"""
Referenssi AI-bridgen embedding-polusta.

Jonka ajatus on tämä:

- ai-botti ei kutsu Ollamaa suoraan
- AI bridge vastaanottaa embedding-pyynnön
- bridge validoi ja normalisoi pyynnön
- bridge kutsuu Ollaman embedding-rajapintaa
- Ollaman vastaus normalisoidaan ai-botille sopivaan muotoon

Tämän kerroksen tarkoitus on pitää provider-kohtaiset yksityiskohdat
AI-bridgen sisällä. Näin ai-botin muu koodi voi käsitellä embedding-vastauksia
aina samalla tavalla.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol


class BridgeEmbeddingError(RuntimeError):
    """Nostetaan, jos embedding-pyyntö tai providerin vastaus on virheellinen."""


@dataclass(frozen=True)
class EmbeddingRequestReference:
    model: str
    inputs: list[str]


@dataclass(frozen=True)
class EmbeddingResponseReference:
    ok: bool
    model: str
    embeddings: list[list[float]]
    dimensions: int
    latency_ms: int


class OllamaEmbeddingClientReference(Protocol):
    """Vähimmäisrajapinta käyttäjälle, joka osaa pyytää embeddingit Ollamalta."""

    async def embed(self, *, model: str, inputs: list[str]) -> dict[str, Any]:
        ...


def normalize_embedding_request(payload: EmbeddingRequestReference) -> EmbeddingRequestReference:
    # Bridge ei lähetä tyhjiä tekstejä providerille.
    #
    # Virheenkäsittelyn helpottamiseksi ja estämään turhia embedding-kutsuja.
    model = payload.model.strip()
    inputs = [item.strip() for item in payload.inputs if item.strip()]

    if not model:
        raise BridgeEmbeddingError("Embedding model is missing.")

    if not inputs:
        raise BridgeEmbeddingError("Embedding inputs cannot be blank.")

    return EmbeddingRequestReference(
        model=model,
        inputs=inputs,
    )


def normalize_ollama_embeddings(
    raw_response: dict[str, Any],
    *,
    requested_model: str,
    expected_count: int,
    latency_ms: int,
) -> EmbeddingResponseReference:
    # Ollaman vastaus normalisoidaan yhteen muotoon, jota ai-botti osaa lukea.
    #
    # Tässä referenssissä oletetaan ensisijaisesti, että vastaus sisältää kentän:
    #
    #   embeddings: list[list[float]]
    #
    # Lisäksi hyväksytään yksittäisen embeddingin muoto:
    #
    #   embedding: list[float]
    #
    # Tekee referenssistä helpommin ymmärrettävän esim. silloin, jos providerin
    # vastausmuoto vaihtelee yhden ja usean tekstin välillä.
    raw_embeddings = raw_response.get("embeddings")

    if raw_embeddings is None and "embedding" in raw_response:
        raw_embeddings = [raw_response["embedding"]]

    if not isinstance(raw_embeddings, list) or not raw_embeddings:
        raise BridgeEmbeddingError("Ollama embeddings payload did not contain embeddings.")

    normalized: list[list[float]] = []

    for index, item in enumerate(raw_embeddings):
        if not isinstance(item, list) or not item:
            raise BridgeEmbeddingError(
                f"Ollama embeddings payload contained an invalid embedding row at index {index}."
            )

        try:
            normalized.append([float(value) for value in item])
        except (TypeError, ValueError) as exc:
            raise BridgeEmbeddingError(
                f"Ollama embeddings payload contained a non-numeric value at index {index}."
            ) from exc

    if len(normalized) != expected_count:
        raise BridgeEmbeddingError(
            f"Embedding count mismatch: expected {expected_count}, got {len(normalized)}."
        )

    dimensions = len(normalized[0])

    if dimensions <= 0:
        raise BridgeEmbeddingError("Embedding dimensions must be greater than zero.")

    if any(len(item) != dimensions for item in normalized):
        raise BridgeEmbeddingError("Ollama embeddings payload contained inconsistent dimensions.")

    response_model = str(raw_response.get("model") or requested_model)

    return EmbeddingResponseReference(
        ok=True,
        model=response_model,
        embeddings=normalized,
        dimensions=dimensions,
        latency_ms=latency_ms,
    )


async def embed_route_reference(
    payload: EmbeddingRequestReference,
    *,
    ollama_client: OllamaEmbeddingClientReference,
) -> EmbeddingResponseReference:
    # Vastaa ajatustasolla AI-bridgen `/v1/embeddings`-reittiä.
    #
    # Botin tuotantokoodissa tämä olisi esimerkiksi FastAPI-reitti, joka:
    #
    # 1. vastaanottaa JSON-payloadin ai-botilta
    # 2. validoi mallin ja syötteet
    # 3. kutsuu provideria eli tässä tapauksessa Ollamaa
    # 4. mittaa kutsun keston
    # 5. normalisoi vastauksen ai-botille sopivaan muotoon
    normalized_payload = normalize_embedding_request(payload)

    started_at = perf_counter()

    try:
        raw_response = await ollama_client.embed(
            model=normalized_payload.model,
            inputs=normalized_payload.inputs,
        )
    except Exception as exc:  # pragma: no cover - idea tasolla esimerkki koodiin
        raise BridgeEmbeddingError(f"Ollama embedding request failed: {exc}") from exc

    latency_ms = int((perf_counter() - started_at) * 1000)

    return normalize_ollama_embeddings(
        raw_response,
        requested_model=normalized_payload.model,
        expected_count=len(normalized_payload.inputs),
        latency_ms=latency_ms,
    )