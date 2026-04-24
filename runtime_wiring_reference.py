# Tämä on referenssikopio, ei tuotantokäyttöön tarkoitettu moduuli.
"""
Referenssi siitä, miten vector-hakuun liittyvät komponentit kytketään ai-botin runtimeen.

Runtime tarkoittaa sovelluksen ajonaikaista kokoonpanoa: sitä vaihetta,
jossa tietokantayhteys, palveluluokat, asetukset ja riippuvuudet rakennetaan
ennen kuin botti alkaa käsitellä käyttäjien pyyntöjä.

Tiedostoa ei ole tarkoitus ajaa sellaisenaan. Sen tarkoitus on näyttää
lukijalle rakennusjärjestys sekä komponenttien väliset riippuvuudet.

Yksinkertaistettu rakennusjärjestys on:
1. RuntimeStore
2. TenantService
3. EmbeddingService
4. KnowledgeService
5. EmbeddingJobService
6. knowledge_service.set_embedding_job_service(...)

Tärkein vaihe on viimeisenä, koska KnowledgeService voi tarvittaessa luoda
embedding-jobin automaattisesti, jos vector-hakuun tarvittava embedding puuttuu
tai se on vanhentunut.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeVectorWiringReference:
    store: str
    tenant_service: str
    embedding_service: str
    knowledge_service: str
    embedding_job_service: str
    note: str


def build_runtime_vector_wiring_reference() -> RuntimeVectorWiringReference:
    # Ajatus:
    #
    # 1. Store rakennetaan ensin.
    #    Store vastaa tietokantayhteydestä, skeeman alustuksesta ja pysyvästä datasta.
    #
    # 2. TenantService rakennus storen ja konfiguraatioiden päälle.
    #    TenantService ratkaisee esimerkiksi tenantin planin, oikeudet ja feature-flagit.
    #
    # 3. EmbeddingServicen rakennus AI-bridge-asetuksilla.
    #    Tämä palvelu pyytää embeddingit AI-bridgen kautta.
    #
    # 4. KnowledgeServicelle käyttöön store, tena-palvelu ja embed-palvelu.
    #    Vastaa tietopankin dokumenteista, chunkeista ja hakulogiikasta.
    #
    # 5. EmbeddingJobService saa käyttöönsä storen, tenant-palvelun ja embedding-palvelun.
    #    Se vastaa embedding-jobien jonottamisesta ja käsittelystä.
    #
    # 6. Lopuksi KnowledgeServicelle viite EmbeddingJobServiceen
    #    jolla KnowledgeService voi auto-enqueue-ta embedding-jobin, jos
    #    käyttäjän haku osuu tilanteeseen, jossa embedding puuttuu tai on stale.
    return RuntimeVectorWiringReference(
        store="RuntimeStore(database_config, initialize=True)",
        tenant_service=(
            "TenantService("
            "store, app_config, plans_config, bot_instance_service"
            ")"
        ),
        embedding_service=(
            "EmbeddingService("
            "base_url, timeout_seconds, default_provider, default_model"
            ")"
        ),
        knowledge_service=(
            "KnowledgeService("
            "store, tenant_service, app_config, embedding_service"
            ")"
        ),
        embedding_job_service=(
            "EmbeddingJobService("
            "store, tenant_service, app_config, embedding_service"
            ")"
        ),
        note=(
            "Lopuksi kutsutaan "
            "knowledge_service.set_embedding_job_service(embedding_job_service)"
        ),
    )