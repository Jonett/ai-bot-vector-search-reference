# Tämä on referenssikopio, ei tuotantokäyttöön tarkoitettu moduuli.
"""
Referenssi doctor- ja smoke-tarkistuksista,
jotka liittyvät vector-hakuun, embedingeihin ja embeding-jobiputkeen.

Doctor-tarkistus kertoo järjestelmän tilasta esim. embedingien kattavuuden ja jobiputken tilanteen.

Smoken varmistaa, että tärkeimmät peruspolut toimivat.
Smoket ei ole yleensä raskaita eikä niillä ole tai tehdä ulkoisia kutsuja. esim.
embeding-provideria ei tarvitse kutsua pelkässä smoke-testissä.

Yksinkertaistettuna:

- doctor kertoo, onko järjestelmä terve
- smoke kertoo, rikkoutuiko jokin peruspolku kokonaan
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


DiagnosticStatus = Literal["OK", "WARN", "FAIL"]


@dataclass(frozen=True)
class DiagnosticCheckReference:
    name: str
    status: DiagnosticStatus
    summary: str
    details: tuple[str, ...] = ()


def check_knowledge_embeddings_reference(
    *,
    embedded_chunk_count: int,
    missing_embedding_count: int,
    stale_embedding_count: int,
) -> DiagnosticCheckReference:
    # Tämä tarkistus kertoo, onko tenantin tietopankin embeding-kattavuus kunnossa.
    #      kuinka monella...
    # embedded =  ...aktiivisella chunkilla on ajantasainen embedding
    # missing  = ...aktiiviselta chunkilta embedding puuttuu
    # stale    = ...chunkilla embedding on vanhentunut
    #
    # Missing- tai stale-tilanne ei välttämättä tarkoita, että järjestelmä on rikki.
    # Se voi tarkoittaa myös sitä, että embeding-jobit ovat vielä kesken.
    if missing_embedding_count > 0 or stale_embedding_count > 0:
        return DiagnosticCheckReference(
            name="knowledge.embeddings",
            status="WARN",
            summary="Embeding-kattavuus on osittainen tai osa embedingeistä on vanhentunut.",
            details=(
                f"embedded={embedded_chunk_count}",
                f"missing={missing_embedding_count}",
                f"stale={stale_embedding_count}",
            ),
        )

    return DiagnosticCheckReference(
        name="knowledge.embeddings",
        status="OK",
        summary="Embeding-kattavuus näyttää ehjältä tai embedingejä ei vielä tarvita.",
        details=(
            f"embedded={embedded_chunk_count}",
            f"missing={missing_embedding_count}",
            f"stale={stale_embedding_count}",
        ),
    )


def check_embedding_jobs_reference(
    *,
    pending_count: int,
    running_count: int,
    failed_count: int,
    stale_running_count: int,
) -> DiagnosticCheckReference:
    # Näyttääkö embeding-jobiputki normaalilta.
    #
    # pending       = jonossa odottavat jobit
    # running       = parhaillaan käsittelyssä olevat jobit
    # failed        = epäonnistuneet jobit
    # stale_running = liian pitkäksi aikaa running-tilaan jääneet jobit
    #
    # Failed-jobit tai stale-running-jobit nostetaan varoituksena,
    # koska ne vaativat yleensä ylläpitäjän huomiota.
    if failed_count > 0 or stale_running_count > 0:
        return DiagnosticCheckReference(
            name="embedding.jobs",
            status="WARN",
            summary="Embeding-jobiputkessa on epäonnistuneita tai jumittuneita hommia.",
            details=(
                f"pending={pending_count}",
                f"running={running_count}",
                f"failed={failed_count}",
                f"stale_running={stale_running_count}",
            ),
        )

    return DiagnosticCheckReference(
        name="embedding.jobs",
        status="OK",
        summary="Embeding-jobiputken tila näyttää normaalilta.",
        details=(
            f"pending={pending_count}",
            f"running={running_count}",
            f"failed={failed_count}",
            f"stale_running={stale_running_count}",
        ),
    )


def smoke_retrieval_reference() -> DiagnosticCheckReference:
    # Smoke-tarkistus varmastamaan hakupolun kevyet perusosat toimivat.
    #
    # Tuotantokoodissa tämä tarkistaa esim:
    #
    # - cosine_similarity-apurin
    # - keyword-haun peruspolun
    # - debug_search-polun
    # - fallback-logiikan perusrakenteen
    #
    # Smoke ei yleensä tee ulkoisia embedding-kutsuja, että testi pysyy nopeana ja
    # vakaana ajaa myös kehitys- tai CI-ympäristössä.
    return DiagnosticCheckReference(
        name="smoke.retrieval",
        status="OK",
        summary="Keyword- ja vector-haun kevyet apurit vastasivat smoke-tarkistuksessa ilman ulkoisia kutsuja.",
    )


def smoke_embedding_jobs_reference() -> DiagnosticCheckReference:
    # Smoke-tarkistus varmistaa, että embeding-jobiputken perusrakenne toimii.
    #
    # Tuotantokoodissa tämä voi tarkistaa esimerkiksi:
    #
    # - embedding-status-näkymät
    # - stale/missing-chunkkien laskennan
    # - enqueue-missing-komennon dry-run-tilassa
    # - jobien tilayhteenvedon muodostamisen
    #
    # Tämä smoke ei rakenna oikeita embedingejä eikä kutsu ulkoista provideria.
    return DiagnosticCheckReference(
        name="smoke.embedding_jobs",
        status="OK",
        summary="Embeding-jobien dry-run enqueue ja näkymät toimivat ilman ulkoisia kutsuja.",
    )


def render_diagnostic_report_reference(
    checks: tuple[DiagnosticCheckReference, ...],
) -> list[str]:
    # Apumetodi kpla näyttää, miltä doctor-raportin yksinkertainen
    # tekstimuotoinen tulostus näyttää.
    #
    # Tuotantokoodissa sama tieto voidaan tulostaa esimerkiksi:
    #
    # - terminaaliin
    # - JSON-muodossa
    # - lokiin
    # - ylläpitonäkymään
    lines: list[str] = []

    for check in checks:
        lines.append(f"[{check.status}] {check.name}: {check.summary}")

        for detail in check.details:
            lines.append(f"  - {detail}")

    return lines