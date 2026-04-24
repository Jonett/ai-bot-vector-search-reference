# Tämä on referenssikopio, ei tuotantokäyttöön tarkoitettu moduuli.
"""
Lyhennetty ja kommentoitu referenssi ai-botin vector- ja hybrid-haun ytimestä.

Tämän tiedoston tarkoitus on näyttää:
- miten hakubackend valitaan
- miten käyttäjän kysymyksestä muodostetaan query embedding
- miten tenant-kohtaiset embedding-kandidaatit ladataan
- miten cosine similarity lasketaan
- miten hybrid-score yhdistää keyword- ja vector-tulokset
- miten järjestelmä palaa tarvittaessa keyword-hakuun

Tiedosto ei sisällä koko botin tuotantokoodia. Mukana on vain haun kannalta
oleellinen logiikan runko, jotta kokonaisuutta on helpompi lukea ja selittää.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True)
class SearchResultReference:
    tenant_id: int
    document_id: int
    document_title: str
    document_status: str
    chunk_id: int
    chunk_index: int
    content: str
    content_length: int
    score: float
    score_breakdown: dict[str, Any]
    match_reason: str
    passed_threshold: bool


@dataclass(frozen=True)
class VectorCandidateReference:
    tenant_id: int
    document_id: int
    document_title: str
    document_status: str
    chunk_id: int
    chunk_index: int
    content: str
    content_length: int
    embedding: list[float]


@dataclass(frozen=True)
class SearchReportReference:
    configured_backend: str
    selected_backend: str
    fallback_reason: str | None
    search_mode: str
    results: tuple[SearchResultReference, ...]
    injected_results: tuple[SearchResultReference, ...]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    # Tämä on vector-haun ydinkaava.
    #
    # Kosinisimilariteetti mittaa kahden vektorin välistä suuntaa.
    # Tässä tapauksessa se tarkoittaa sitä, kuinka samankaltaiseen
    # merkityssuuntaan käyttäjän kysymys ja tietopankin chunkki osoittavat.
    #
    # Arvon tulkinta yksinkertaistettuna:
    # - 1.0 tarkoittaa erittäin vahvaa samankaltaisuutta
    # - 0.0 tarkoittaa, ettei selvää yhteistä suuntaa ole
    # - negatiivinen arvo tarkoittaa yleensä heikkoa tai vääränsuuntaista osumaa
    if not left or not right or len(left) != len(right):
        return 0.0

    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))

    if left_norm == 0 or right_norm == 0:
        return 0.0

    return numerator / (left_norm * right_norm)


def vector_score_scale() -> float:
    # Similarity-arvo on yleensä välillä -1.0 ja 1.0.
    #
    # Projektissa positiivinen similarity muunnetaan hakupisteiksi kertomalla se
    # tällä skaalalla. Kun skaala on 5.0, esimerkiksi similarity 0.82 muuttuu
    # pisteiksi 4.10.
    #
    # Tämä pitää vector-haun pisteet samassa mittakaavassa muun haun kanssa.
    return 5.0


def select_backend(
    *,
    configured_backend: str,
    allow_vector_search: bool,
    allow_hybrid_retrieval: bool,
    embedding_service_available: bool,
    embedded_chunk_count: int,
    keyword_results_exist: bool,
    vector_results_exist: bool,
    embedding_query_failed: bool = False,
) -> tuple[str, str | None]:
    # Vector- ja hybrid-haku ovat keyword-haun päälle rakennettu lisäkerros.
    #
    # Jos jokin vector-haun edellytys ei täyty, järjestelmä palaa keyword-hakuun.
    # Tämän ansiosta ai-botti pystyy edelleen vastaamaan käyttäjälle, vaikka
    # embedding-palvelu ei olisi käytössä, embeddingejä ei olisi vielä rakennettu
    # tai tenantilla ei olisi oikeutta vector-hakuun.
    selected_backend = configured_backend
    fallback_reason: str | None = None

    if configured_backend in {"vector", "hybrid"}:
        if not allow_vector_search:
            return "keyword", "vector_feature_disabled"

        if configured_backend == "hybrid" and not allow_hybrid_retrieval:
            return "keyword", "hybrid_feature_disabled"

        if not embedding_service_available:
            return "keyword", "embedding_service_unavailable"

        if embedded_chunk_count <= 0:
            return "keyword", "no_embeddings_available"

        if embedding_query_failed:
            return "keyword", "embedding_provider_failed"

    # Jos käytössä on pelkkä vector-haku, mutta se ei löydä osumia,
    # keyword-haku voi vielä antaa käyttökelpoisen tuloksen.
    if configured_backend == "vector" and not vector_results_exist and keyword_results_exist:
        return "keyword", "vector_no_matches"

    return selected_backend, fallback_reason


def vector_search(
    *,
    query_embedding: list[float],
    candidates: list[VectorCandidateReference],
    min_similarity: float,
    min_relevance_score: float,
    provider: str,
    model: str,
    limit: int,
) -> list[SearchResultReference]:
    results: list[SearchResultReference] = []

    for candidate in candidates:
        # Tenant-eristys tehdään ennen tätä vaihetta.
        #
        # candidates-listaan saa päätyä vain sen tenantin chunkkeja,
        # jonka tietopankista käyttäjän kysymykseen haetaan vastauksia.
        similarity = round(cosine_similarity(query_embedding, candidate.embedding), 6)

        # Negatiivisia tai nollaan jääviä similarity-arvoja ei kannata
        # nostaa mukaan hakutuloksiin.
        if similarity <= 0:
            continue

        score = round(similarity * vector_score_scale(), 3)
        passed = similarity >= min_similarity and score >= min_relevance_score

        results.append(
            SearchResultReference(
                tenant_id=candidate.tenant_id,
                document_id=candidate.document_id,
                document_title=candidate.document_title,
                document_status=candidate.document_status,
                chunk_id=candidate.chunk_id,
                chunk_index=candidate.chunk_index,
                content=candidate.content,
                content_length=candidate.content_length,
                score=score,
                score_breakdown={
                    "backend": "vector",
                    "vector_similarity": similarity,
                    "vector_score": score,
                    "vector_min_similarity": min_similarity,
                    "provider": provider,
                    "model": model,
                },
                match_reason="vector",
                passed_threshold=passed,
            )
        )

    # Vector-tuloksissa tärkein järjestysperuste on similarity.
    # Sen jälkeen käytetään pisteitä ja lopuksi vakauttavia järjestyskenttiä,
    # jotta tulosjärjestys pysyy ennustettavana.
    results.sort(
        key=lambda result: (
            -float(result.score_breakdown.get("vector_similarity", 0.0)),
            -result.score,
            result.document_title.lower(),
            result.chunk_index,
        )
    )

    return results[:limit]


def load_vector_candidates(rows: list[dict[str, Any]]) -> list[VectorCandidateReference]:
    candidates: list[VectorCandidateReference] = []

    for row in rows:
        # Stale-embeddingit jätetään pois vertaamalla chunkin nykyistä
        # content_hash-arvoa embedding-rivin content_hash-arvoon.
        #
        # Jos arvot eivät täsmää, chunkin sisältö on muuttunut sen jälkeen,
        # kun embedding rakennettiin. Silloin vanhaa embeddingiä ei saa käyttää haussa.
        if row["content_hash"] is None or row["embedding_content_hash"] != row["content_hash"]:
            continue

        embedding = row["embedding_json"]

        # Referenssissä oletetaan, että embedding_json on jo purettu listaksi.
        # Tuotantokoodissa tämä voi tulla tietokannasta JSON-muotoisena merkkijonona,
        # jolloin se puretaan ennen tätä vaihetta.
        if not isinstance(embedding, list) or not embedding:
            continue

        candidates.append(
            VectorCandidateReference(
                tenant_id=int(row["tenant_id"]),
                document_id=int(row["document_id"]),
                document_title=str(row["document_title"]),
                document_status=str(row["document_status"]),
                chunk_id=int(row["chunk_id"]),
                chunk_index=int(row["chunk_index"]),
                content=str(row["content"]),
                content_length=int(row["content_length"]),
                embedding=[float(value) for value in embedding],
            )
        )

    return candidates


def merge_hybrid_results(
    *,
    keyword_results: list[SearchResultReference],
    vector_results: list[SearchResultReference],
    keyword_weight: float,
    vector_weight: float,
    min_relevance_score: float,
    vector_min_similarity: float,
) -> list[SearchResultReference]:
    # Hybrid-haku yhdistää saman chunkin kaksi näkökulmaa:
    #
    # 1. keyword-score kertoo, kuinka hyvin chunkki osui avainsanahakuun
    # 2. vector-score kertoo, kuinka lähellä chunkki oli kysymystä merkityksen perusteella
    #
    # Jos sama chunkki löytyy molemmilla tavoilla, pisteet yhdistetään painotetusti.
    # Jos chunkki löytyy vain toisella tavalla, se voi silti päästä mukaan tuloksiin.
    merged: dict[int, dict[str, SearchResultReference | None]] = {}

    for result in keyword_results:
        merged[result.chunk_id] = {"keyword": result, "vector": None}

    for result in vector_results:
        entry = merged.setdefault(result.chunk_id, {"keyword": None, "vector": None})
        entry["vector"] = result

    final: list[SearchResultReference] = []

    for entry in merged.values():
        keyword_result = entry["keyword"]
        vector_result = entry["vector"]

        keyword_score = keyword_result.score if keyword_result is not None else None

        vector_similarity = (
            float(vector_result.score_breakdown.get("vector_similarity", 0.0))
            if vector_result is not None
            else None
        )

        vector_score = (
            round(vector_similarity * vector_score_scale(), 3)
            if vector_similarity is not None
            else None
        )

        if keyword_score is not None and vector_score is not None:
            denominator = keyword_weight + vector_weight

            # Painojen summan pitäisi normaalisti olla positiivinen.
            # Tämä suojaa referenssikoodia nollalla jakamiselta.
            if denominator <= 0:
                denominator = 1.0

            final_score = round(
                ((keyword_score * keyword_weight) + (vector_score * vector_weight)) / denominator,
                3,
            )
            base_result = keyword_result

        elif keyword_score is not None:
            final_score = keyword_score
            base_result = keyword_result

        elif vector_score is not None:
            final_score = vector_score
            base_result = vector_result

        else:
            continue

        assert base_result is not None

        passed = final_score >= min_relevance_score and (
            (keyword_result is not None and keyword_result.passed_threshold)
            or (vector_similarity is not None and vector_similarity >= vector_min_similarity)
        )

        breakdown = dict(base_result.score_breakdown)
        breakdown.update(
            {
                "backend": "hybrid",
                "keyword_score": keyword_score,
                "vector_score": vector_score,
                "vector_similarity": vector_similarity,
                "hybrid_keyword_weight": keyword_weight,
                "hybrid_vector_weight": vector_weight,
            }
        )

        match_parts: list[str] = []

        if keyword_result is not None:
            match_parts.append(f"keyword:{keyword_result.match_reason}")

        if vector_result is not None:
            match_parts.append("vector")

        final.append(
            replace(
                base_result,
                score=final_score,
                score_breakdown=breakdown,
                match_reason=", ".join(match_parts) or "hybrid",
                passed_threshold=passed,
            )
        )

    # Hybrid-tulokset järjestetään ensisijaisesti lopullisen score-arvon mukaan.
    # Sen jälkeen suositaan tuloksia, joilla on myös vector-osuma, ja lopuksi
    # käytetään vakauttavia järjestyskenttiä.
    final.sort(
        key=lambda result: (
            -result.score,
            -(1 if result.score_breakdown.get("vector_similarity") else 0),
            -len(result.score_breakdown.get("matched_terms", [])),
            result.content_length,
            result.document_title.lower(),
            result.chunk_index,
        )
    )

    return final


def build_knowledge_context_reference(
    results: list[SearchResultReference],
    max_chars: int,
) -> str | None:
    # Promptiin ei lisätä kaikkia hakutuloksia.
    #
    # Mukaan pääsevät vain ne chunkit, jotka ovat läpäisseet haun raja-arvot.
    # Tämä pitää promptin tiiviimpänä ja vähentää riskiä, että ai-botti saa
    # vastauksensa tueksi heikkoja tai asiaan liittymättömiä osumia.
    injected = [result for result in results if result.passed_threshold]

    if not injected:
        return None

    blocks: list[str] = []
    total = 0

    for result in injected:
        block = (
            f"Lähde: {result.document_title}\n"
            f"Chunk: {result.chunk_index}\n"
            f"Sisältö: {result.content.strip()}"
        )

        separator_length = 2 if blocks else 0

        if total + separator_length + len(block) > max_chars:
            break

        blocks.append(block)
        total += separator_length + len(block)

    return "\n\n".join(blocks) if blocks else None