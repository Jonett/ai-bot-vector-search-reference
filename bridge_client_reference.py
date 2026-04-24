# Tämä on referenssikopio, ei tuotantokäyttöön tarkoitettu moduuli.
"""
Lyhennetty ja kommentoitu referenssi ai-botin ja AI-bridgen välisestä käyttäjästä.

Tämä on se kerros, jonka kautta ai-botti pyytää AI-bridgeltä:

- chat-vastaukset endpointista `/v1/chat`
- embeddingit endpointista `/v1/embeddings`

Ai-botti ei siis kutsu esimerkiksi Ollamaa suoraan. Sen sijaan kaikki
AI-palveluihin liittyvät HTTP-kutsut kulkevat AI-bridgen kautta.

Tämän kerroksen tehtävä on pitää HTTP-kutsujen yksityiskohdat poissa
botin muusta koodista.
"""

from __future__ import annotations

import json
from typing import Any

import httpx


class BridgeClientError(RuntimeError):
    """Nostetaan, jos AI-bridgeen tehty HTTP-kutsu epäonnistuu."""


class BridgeClientReference:
    def __init__(self, base_url: str, timeout_seconds: float) -> None:
        if not base_url.strip():
            raise BridgeClientError("AI bridge base URL is missing.")

        if timeout_seconds <= 0:
            raise BridgeClientError("AI bridge timeout must be greater than zero.")

        # Timeout jaetaan eri vaiheisiin.
        #
        # connect = yhteyden avaaminen
        # read    = vastauksen lukeminen
        # write   = pyynnön lähettäminen
        # pool    = yhteysaltaasta vapaan yhteyden odottaminen
        #
        # Lyhyemmät connect/pool-timeoutit estävät, ettei ai-botti jää pitkäksi
        # aikaa odottamaan palvelua, johon ei saada yhteyttä.
        timeout = httpx.Timeout(
            connect=min(5.0, timeout_seconds),
            read=timeout_seconds,
            write=min(15.0, timeout_seconds),
            pool=min(5.0, timeout_seconds),
        )

        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
        )

    async def ask(self, payload: dict[str, Any]) -> dict[str, Any]:
        # Chat-kutsu välitetään AI-bridgen /v1/chat-endpointtiin.
        #
        # payload sisältää käytännössä sen datan, jonka perusteella bridge osaa
        # muodostaa varsinaisen provider-kutsun. Provider voi olla esimerkiksi
        # paikallinen malli, Ollama tai jokin myöhemmin lisätty palvelu.
        return await self._post_json("/v1/chat", payload)

    async def embed_texts(self, *, model: str, inputs: list[str]) -> dict[str, Any]:
        # Embedding-kutsu välitetään AI-bridgen /v1/embeddings-endpointtiin.
        #
        # Tässä vaiheessa ai-botti kertoo vain:
        # - mitä mallia halutaan käyttää
        # - mistä teksteistä embeddingit halutaan muodostaa
        #
        # Provider-kohtainen toteutus jää AI-bridgen vastuulle.
        if not model.strip():
            raise BridgeClientError("Embedding model is missing.")

        normalized_inputs = [item.strip() for item in inputs if item.strip()]

        if not normalized_inputs:
            raise BridgeClientError("Embedding inputs cannot be blank.")

        return await self._post_json(
            "/v1/embeddings",
            {
                "model": model.strip(),
                "inputs": normalized_inputs,
            },
        )

    async def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        # Sama JSON-lähetyslogiikka on käytössä sekä chat- että embedding-kutsuissa.
        #
        # ensure_ascii=False säilyttää ääkköset luettavassa muodossa.
        # Varsinainen lähetys tehdään UTF-8-muotoisena.
        try:
            response = await self._client.post(
                path,
                content=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json; charset=utf-8"},
            )
            response.encoding = "utf-8"
            response.raise_for_status()

        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            body_preview = exc.response.text[:500]
            raise BridgeClientError(
                f"AI bridge returned HTTP {status_code}: {body_preview}"
            ) from exc

        except httpx.HTTPError as exc:
            raise BridgeClientError(f"AI bridge request failed: {exc}") from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise BridgeClientError("AI bridge returned invalid JSON.") from exc

        if not isinstance(data, dict):
            raise BridgeClientError("AI bridge response must be a JSON object.")

        return dict(data)

    async def close(self) -> None:
        # AsyncClient pitää sulkea, kun sitä ei enää tarvita.
        # Tuotantokoodissa tämä hoituu viimeistää kontin tuhoutuessa tai sovelluksen shutdown-vaiheessa.
        await self._client.aclose()

    async def __aenter__(self) -> BridgeClientReference:
        # Mahdollistaa käytön muodossa:
        #
        # async with BridgeClientReference(...) as client:
        #     response = await client.ask(payload)
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        await self.close()