# Source map

Tämä koostan tiedon että mistä alkuperäisen projektin osista referenssipaketin tiedostot on koottu.

Referenssipaketin tiedostot eivät ole suoria tuotantokopioita. Niistä on poistettu muuta sovellusta koskevaa kuormaa, jotta vector-haun, embeddingien ja jobiputken peruslogiikkaa on helpompi lukea.

| Referenssitiedosto | Alkuperäinen lähde | Mitä mukaan otettiin | Huomio |
| --- | --- | --- | --- |
| `vector_retrieval_reference.py` | `bot/app/services/knowledge_service.py` | Hakubackendin valinta, vector-haku, hybrid-merge, cosine similarity ja prompt-kontekstin muodostus | Lyhennetty ja kommentoitu referenssiversio |
| `embedding_service_reference.py` | `bot/app/services/embedding_service.py` | Embedding-palvelun adapteri, provider-valinta, mallin valinta ja dimension-tarkistukset | Irrotettu omaksi luettavaksi kokonaisuudeksi |
| `bridge_client_reference.py` | `bot/app/services/bridge_client.py` | AI-bridgen HTTP-kutsut endpointteihin `/v1/chat` ja `/v1/embeddings` | Lähes suora referenssikopio, mutta kommentteja on selkeytetty |
| `ai_bridge_embedding_reference.py` | `ai_bridge/app/services/embedding_service.py`, `ai_bridge/app/api/routes.py` | Embeddings-endpointin pyyntöjen validointi, Ollama-kutsu ja vastauksen normalisointi | Yhdistetty referenssitiedosto AI-bridgen embedding-polusta |
| `embedding_jobs_reference.py` | `bot/app/services/embedding_job_service.py`, `bot/app/services/knowledge_service.py` | Jobien luontilogiikka, missing/stale/current-tunnistus ja workerin turvallinen suorituspolku | Lyhennetty referenssiversio jobiputken ydinkohdista |
| `runtime_wiring_reference.py` | `bot/app/services/runtime_environment.py` | Runtime-komponenttien rakennusjärjestys: store, tenant service, embedding service, knowledge service ja job service | Rakenteellinen kuvaus siitä, miten osat kytketään yhteen |
| `operator_cli_reference.py` | `bot/app/operator.py` | `debug-search`, `embeddings-build`, `embeddings-status`, `embeddings-clear` ja embedding-jobien komennot | Komennot on koottu luettavaksi CLI-referenssiksi |
| `diagnostics_reference.py` | `bot/app/services/diagnostics_service.py` | Doctor- ja smoke-tarkistukset vector-haulle, embeddingeille ja embedding-jobeille | Lyhennetty opetusversio diagnostiikan perusajatuksesta |
| `db_schema_reference.sql` | `bot/app/db/context_store.py` | `knowledge_embeddings`, `embedding_jobs`, niihin liittyvät indeksit ja olennaiset näkymät | SQL-poiminta, jota on kommentoitu lukemista varten |
| `config_reference.json` | `bot/config/app_config.json`, `bot/config/plans.json` | Retrieval- ja embedding-asetukset sekä plan-kohtaiset feature-flagit | Turvallinen poiminta ilman salaisuuksia |
| `tests_reference.py` | `tests/bot/test_vector_search_embeddings.py`, `tests/bot/test_production_hardening.py` | Fake embedding service, tärkeimmät testiskenaariot sekä doctor/smoke-odotukset | Lyhennetty opetusversio testeistä |
| `summary.md` | Tämä referenssipaketti | Yhteenveto siitä, mitä pakettiin otettiin mukaan ja mitä jätettiin pois | Paketin lukemista helpottava yleiskuva |

## Huomio nimistöstä

Alkuperäisessä ai-botti projektissa osa tunnisteista voi vielä viitata Discord-rakenteisiin, esimerkiksi `guild_id`.

Tässä referenssipaketissa sama ajatus on pyritty esittämään yleisemmällä tavalla tenant-käsitteen kautta. Sen vuoksi referenssitiedostoissa käytetään ensisijaisesti nimeä `tenant_id`.

Tämä tekee paketista helpommin ymmärrettävän myös silloin, kun lukija ei tunne alkuperäisen projektin Discord-taustaa.

## Mitä tästä paketista on jätetty pois?

Referenssipaketista on tarkoituksella jätetty pois asioita, jotka eivät ole vector-haun ymmärtämisen kannalta välttämättömiä.

Pois on jätetty esimerkiksi:

- sovelluksen muu komentorakenne
- autentikointi ja käyttöoikeuksien koko toteutus
- tuotantokonfiguraation salaisuudet
- Discord-spesifinen käsittely
- laajempi tietokantaskeema
- koko testisuite
- virheenkäsittelyn kaikki tuotantopolut

Tavoitteena on, että lukija voi keskittyä vain tähän kokonaisuuteen:

```text
käyttäjän kysymys
-> query embedding
-> tenantin ajantasaiset chunk-embeddingit
-> cosine similarity
-> vector- tai hybrid-score
-> parhaat osumat promptin kontekstiksi
```