# Tämä on referenssikopio, ei tuotantokäyttöön tarkoitettu moduuli.
"""
Lyhennetty ja kommentoitu referenssi ai-botin embedding job -putkesta.

Tämä tiedosto käsittää kolme ydinasiaa:

- milloin embedding-jobi syntyy
- miten missing-, stale- ja current-tilat tunnistetaan
- miten worker käsittelee claimaamansa jobin turvallisesti

Embedding-jobien tarkoitus on rakentaa tai päivittää tietopankin chunkkien
embeddingit hallitusti taustalla. Näin käyttäjän hakupolku ei joudu odottamaan,
että kaikki embeddingit rakennetaan samalla hetkellä.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal


JobProcessStatus = Literal["completed", "skipped", "cancelled", "retry", "failed"]


@dataclass(frozen=True)
class EmbeddingCandidateReference:
    tenant_id: int
    document_id: int
    chunk_id: int
    chunk_index: int
    content: str
    content_length: int
    content_hash: str
    has_current_embedding: bool
    has_active_job: bool
    has_stale_embedding: bool


@dataclass(frozen=True)
class EmbeddingJobReference:
    job_id: int
    tenant_id: int
    document_id: int
    chunk_id: int
    embedding_provider: str
    embedding_model: str
    content_hash: str
    attempts: int
    max_attempts: int


@dataclass(frozen=True)
class EnqueueResultReference:
    status: str
    jobs_enqueued: int
    jobs_skipped: int
    missing_embedding_count: int
    stale_embedding_count: int


def should_auto_enqueue(
    *,
    allow_vector_search: bool,
    allow_hybrid_retrieval: bool,
    retrieval_backend: str,
    auto_enqueue_setting: bool,
) -> bool:
    # Jobien automaattinen luonti on järkevää vain, jos tenalla on oikeus
    # käyttää vector-hakua.
    #
    # Jos käytössä on vector- / hybrid-backend, embeddingejä tarvitaan haun
    # toimintaan. Silloin jobien automaattinen luonti voidaan sallia, vaikka
    # erillinen auto_enqueue-asetus ei olisi päällä.
    if not allow_vector_search:
        return False

    if auto_enqueue_setting:
        return True

    if retrieval_backend == "vector":
        return True

    if retrieval_backend == "hybrid" and allow_hybrid_retrieval:
        return True

    return False


def classify_candidate_rows(
    candidates: list[EmbeddingCandidateReference],
    *,
    limit: int | None = None,
) -> EnqueueResultReference:
    # Candidate-lista kuvaa chunkkeja, joista embedding voisi mahdollisesti
    # puuttua tai olla vanhentunut.
    #         chunkilla...
    # missing =  ...ei ole ajantasaista embeddingiä
    # stale   =  ...on embedding, mutta se ei vastaa nykyistä content_hashia
    # current =  ...on ajantasainen embedding
    missing_embedding_count = sum(
        1 for row in candidates if not row.has_current_embedding
    )

    stale_embedding_count = sum(
        1 for row in candidates if row.has_stale_embedding
    )

    # Jobi luodaan vain, jos:
    #
    # - chunkilla ei ole ajantasaista embeddingiä
    # - chunkille ei ole jo aktiivista jobia jonossa
    #
    # Tarkoituksena olisi estää saman chunkin turha jonottamisen tarve useaan kertaan.
    # Tämä koodi on todella rumaa, lapset laittakaa silmät kiinni ja scrollatkaa eteenpäin.
    enqueue_rows = [
        row
        for row in candidates
        if not row.has_current_embedding and not row.has_active_job
    ]

    if limit is not None and limit >= 0:
        enqueue_rows = enqueue_rows[:limit]

    return EnqueueResultReference(
        status="planned",
        jobs_enqueued=len(enqueue_rows),
        jobs_skipped=len(candidates) - len(enqueue_rows),
        missing_embedding_count=missing_embedding_count,
        stale_embedding_count=stale_embedding_count,
    )


def process_claimed_job_reference(
    *,
    job: EmbeddingJobReference,
    chunk_snapshot: dict[str, object] | None,
    vector_feature_enabled: bool,
    force: bool,
    embed_text: Callable[[str], list[float]],
) -> tuple[JobProcessStatus, dict[str, object] | None]:
    # Worker ei voi eikä saa luotta siihen, että jobin luontihetken tiedot ovat edelleen oikein.
    #
    # Ennen ulkoista embedding-kutsua tarkistetaan vielä uudestaan:
    #
    # - onko chunk edelleen olemassa
    # - onko dokumentti edelleen aktiivinen
    # - onko chunkin sisältö muuttunut jobin luomisen jälkeen
    # - saako tena edelleen käyttää vector-hakua
    #
    # Tämä on tärkeää, koska jobi on voinut olla jonossa hetken aikaa.
    # Sinä aikana dokumentti on voitu poistaa, passivoida tai sen sisältöä on voitu muuttaa.
    if chunk_snapshot is None:
        return "cancelled", {
            "reason": "chunk_missing",
            "job_id": job.job_id,
            "chunk_id": job.chunk_id,
        }

    if chunk_snapshot.get("document_status") != "active":
        return "cancelled", {
            "reason": "document_not_active",
            "job_id": job.job_id,
            "document_id": job.document_id,
            "document_status": chunk_snapshot.get("document_status"),
        }

    if chunk_snapshot.get("content_hash") != job.content_hash:
        return "skipped", {
            "reason": "content_hash_changed",
            "job_id": job.job_id,
            "chunk_id": job.chunk_id,
            "job_content_hash": job.content_hash,
            "current_content_hash": chunk_snapshot.get("content_hash"),
        }

    if not force and not vector_feature_enabled:
        return "cancelled", {
            "reason": "vector_feature_disabled",
            "job_id": job.job_id,
            "tenant_id": job.tenant_id,
        }

    content = str(chunk_snapshot.get("content") or "").strip()

    if not content:
        return "skipped", {
            "reason": "empty_chunk_content",
            "job_id": job.job_id,
            "chunk_id": job.chunk_id,
        }

    try:
        embedding = embed_text(content)

    except Exception as exc:
        # Botin tuotantokoodissa epäonnisttunut jobi voidaan joko yrittää myöhemmin uudelleen
        # tai merkitä failed-tilaan lopullisesti.
        #
        # Referenssissä palautetaan nyt "retry", jos yrityksiä on vielä jäljellä.
        # Muuten palautetaan "failed".
        next_attempt = job.attempts + 1

        if next_attempt < job.max_attempts:
            return "retry", {
                "reason": "embedding_provider_failed",
                "job_id": job.job_id,
                "attempts": next_attempt,
                "max_attempts": job.max_attempts,
                "error": str(exc),
            }

        return "failed", {
            "reason": "embedding_provider_failed",
            "job_id": job.job_id,
            "attempts": next_attempt,
            "max_attempts": job.max_attempts,
            "error": str(exc),
        }

    if not embedding:
        return "failed", {
            "reason": "empty_embedding_returned",
            "job_id": job.job_id,
            "chunk_id": job.chunk_id,
        }

    # Tuotantokoodissa tähän kohtaan upsertoidaan knowledge_embeddings-rivi.
    #
    # "Upsert" tarkoittaa käytännössä:
    #
    # - jos embedding-riviä ei ole, se luodaan
    # - jos embedding-rivi on jo olemassa, se päivitetään
    #
    # content_hash tallennetaan mukaan, jotta myöhemmin voidaan tunnistaa,
    # vastaako embedding edelleen chunkin nykyistä sisältöä.
    saved_embedding = {
        "tenant_id": job.tenant_id,
        "chunk_id": job.chunk_id,
        "document_id": job.document_id,
        "provider": job.embedding_provider,
        "model": job.embedding_model,
        "embedding_dimensions": len(embedding),
        "content_hash": job.content_hash,
        "embedding_json": list(embedding),
    }

    return "completed", saved_embedding


def stale_embedding_explanation() -> str:
    return (
        "Embedding on stale, jos knowledge_chunks.content_hash ei enää vastaa "
        "knowledge_embeddings.content_hash-arvoa. Tämä tarkoittaa, että chunkin "
        "sisältö on muuttunut embed rakentamisen jälkeen. Tällöin vanhaa "
        "vektoria ei käytetä vector-haussa, vaan chunkille pitää rakentaa uusi embedding."
    )