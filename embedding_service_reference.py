# Tämä on referenssikopio, ei tuotantokäyttöön tarkoitettu moduuli.
"""
Lyhennetty ja kommentoitu referenssi ai-botin embedding-adapterista.

Tämän kerroksen tehtävä on tarjota ai-botille yksinkertainen rajapinta
embeddingien muodostamiseen.

Ai-botti ei kutsu Ollamaa suoraan, vaan pyyntö kulkee näin:

    ai-botti -> AI bridge -> Ollama

Tälle on muutama käytännön syy:

- ai-botin runtime pysyy yksinkertaisempana
- ulkoiset HTTP-virheet voidaan käsitellä yhdessä paikassa
- chat- ja embedding-kutsut kulkevat saman sillan kautta
- provider-kohtaiset yksityiskohdat eivät leviä botin muuhun koodiin

Tiedostossa on mukana on vain embedding-palvelun oleellinen logiikka,
jotta kokonaisuutta olisi helpompi lukea ja seurata.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol


class EmbeddingServiceError(RuntimeError):
    """Nostetaan, jos embedding-provider ei ole saatavilla tai palauttaa virheellisen datan."""


@dataclass(frozen=True)
class EmbeddingResponsePayload:
    provider: str
    model: str
    dimensions: int
    embeddings: tuple[tuple[float, ...], ...]
    latency_ms: int


class BridgeEmbeddingResponse(Protocol):
    """AI bridgen palauttaman embedding-vastauksen vähimmäismuoto."""

    model: str
    dimensions: int
    embeddings: list[list[float]]
    latency_ms: int


class BridgeEmbeddingClient(Protocol):
    """Rajapinta käyttäjälle, joka osaa pyytää embeddingit AI bridgeltä."""

    async def embed_texts(self, *, model: str, inputs: list[str]) -> BridgeEmbeddingResponse:
        ...


class EmbeddingServiceReference:
    def __init__(
        self,
        *,
        bridge_client_factory: Callable[[], BridgeEmbeddingClient],
        default_provider: str,
        default_model: str,
        default_dimensions: int | None = None,
    ) -> None:
        self._bridge_client_factory = bridge_client_factory
        self._default_provider = default_provider
        self._default_model = default_model
        self._default_dimensions = default_dimensions

    async def embed_text(
        self,
        text: str,
        *,
        model: str | None = None,
        provider: str | None = None,
    ) -> tuple[float, ...]:
        # Yhden tekstin ohut embedding apumetodi monen tekstin metodin päällä.
        payload = await self.embed_texts([text], model=model, provider=provider)
        return payload.embeddings[0]

    async def embed_texts(
        self,
        texts: list[str],
        *,
        model: str | None = None,
        provider: str | None = None,
    ) -> EmbeddingResponsePayload:
        # Tyhjät merkkijonot poistetaan ennen providerille lähettämistä.
        # Embedding-mallille ei yleensä kannata lähettää tyhjää sisältöä.
        normalized = [text.strip() for text in texts if text.strip()]

        if not normalized:
            raise EmbeddingServiceError("Embedding inputs cannot be blank.")

        resolved_provider = (provider or self._default_provider).strip().lower()

        # Ensimmäinen räpellys tukee vain Olamaa providerina.
        # Rajapinta on silti jätetty provider-pohjaiseksi, jotta myöhemmin voidaan
        # lisätä esimerkiksi OpenAI, Azure OpenAI tai jokin muu embedding-provider.
        if resolved_provider != "ollama":
            raise EmbeddingServiceError(f"Unsupported embedding provider: {resolved_provider}")

        resolved_model = (model or self._default_model).strip()

        if not resolved_model:
            raise EmbeddingServiceError("Embedding model is missing.")

        client = self._bridge_client_factory()

        try:
            response = await client.embed_texts(model=resolved_model, inputs=normalized)
        except Exception as exc:  # pragma: no cover - referenssikoodissa riittää idea
            # Tuotantokoodissa tähän voidaan lisätä tarkempi virheluokittelu,
            # esimerkiksi timeout, provider unavailable tai invalid response.
            raise EmbeddingServiceError(str(exc)) from exc

        dimensions = int(response.dimensions)

        if dimensions <= 0:
            raise EmbeddingServiceError(f"Invalid embedding dimensions: {dimensions}")

        # Jos projektiin on määritelty odotettu embedding-ulottuvuus,
        # varmistetaan ettei provider vaihda mallia tai formaattia huomaamatta.
        if self._default_dimensions is not None and dimensions != self._default_dimensions:
            raise EmbeddingServiceError(
                f"Embedding dimensions mismatch: expected {self._default_dimensions}, got {dimensions}"
            )

        embeddings = tuple(tuple(float(value) for value in item) for item in response.embeddings)

        # Jokaiselle syötteelle pitää tulla yksi embedding.
        if len(embeddings) != len(normalized):
            raise EmbeddingServiceError(
                f"Embedding count mismatch: expected {len(normalized)}, got {len(embeddings)}"
            )

        # Jokaisen embeddingin pituus vastaa ilmoitettua dimensions-arvoa.
        for index, embedding in enumerate(embeddings):
            if len(embedding) != dimensions:
                raise EmbeddingServiceError(
                    f"Embedding dimension mismatch at index {index}: "
                    f"expected {dimensions}, got {len(embedding)}"
                )

        return EmbeddingResponsePayload(
            provider=resolved_provider,
            model=str(response.model),
            dimensions=dimensions,
            embeddings=embeddings,
            latency_ms=int(response.latency_ms),
        )

    def embed_texts_sync(
        self,
        texts: list[str],
        *,
        model: str | None = None,
        provider: str | None = None,
    ) -> EmbeddingResponsePayload:
        # Synkroninen wrapperi CLI-käyttäön.
        #
        # Tätä ei kannata kutsua sellaisesta ympäristöstä, jossa event loop on jo käynnissä,
        # esimerkiksi async-webhandlerin sisältä tai osuu kakka tuulettimeen... Niissä kannattaa käyttää suoraan
        # async-metodia embed_texts().
        return asyncio.run(self.embed_texts(texts, model=model, provider=provider))