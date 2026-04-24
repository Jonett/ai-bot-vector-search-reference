# Tämä on referenssikopio, ei tuotantokäyttöön tarkoitettu moduuli.
"""
Operator-CLI:n vector- ja embedding-komennot.

Operator-CLI on ylläpitäjän komentorivityökalua, jolla voidaan
tutkia, rakentaa, tyhjentää ja diagnosoida tietopankin embeddingejä sekä
vector-/hybrid-hakua.

Tämä tiedosto ei rakenna oikeaa argparse-puuta. Tarkoitus on vain koota yhteen
ne komennot, joita vector-haun ymmärtämisessä kannattaa katsoa.

Komennot on jaettu kolmeen käytännön ryhmään:

1. hakutestit
2. embeddingien hallinta
3. embedding-jobien hallinta
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CommandReference:
    command: str
    purpose: str
    group: str


VECTOR_COMMANDS: tuple[CommandReference, ...] = (
    CommandReference(
        group="Hakutestit",
        command=(
            "python -m bot.app.operator "
            "knowledge debug-search "
            "--tenant-id 1001 "
            "--query \"loot rules\" "
            "--backend hybrid "
            "--json"
        ),
        purpose=(
            "Näyttää keyword-, vector- tai hybrid-haun raportin. "
            "Tätä komentoa käytetään kun halutaan nähdä"
            "miksi jokin chunkki päätyi tai ei päätynyt mukaan hakutuloksiin. "
            "Raportista näkee myös mahdollisen fallback-syyn."
        ),
    ),
    CommandReference(
        group="Embedien hallinta",
        command=(
            "python -m bot.app.operator "
            "knowledge embeddings-build "
            "--tenant-id 1001 "
            "--dry-run"
        ),
        purpose=(
            "Laskee, kuinka monta embediä pitäisi rakentaa ilman että "
            "tietokantaan kirjoitetaan mitään. Tämä on 'turvallinen ensimmäinen "
            "komento ennen varsinaista embedien rakentamista."
        ),
    ),
    CommandReference(
        group="Embedien hallinta",
        command=(
            "python -m bot.app.operator "
            "knowledge embeddings-build "
            "--tenant-id 1001"
        ),
        purpose=(
            "Rakentaa puuttuvat tai vanhentuneet embedit suoraan. "
            "Käytetään silloin, kun embedit halutaan muodostaa heti "
            "ilman tausta jobi putkea."
        ),
    ),
    CommandReference(
        group="Embedien hallinta",
        command=(
            "python -m bot.app.operator "
            "knowledge embeddings-status "
            "--tenant-id 1001"
        ),
        purpose=(
            "Näyttää aktiivisten chunkien määrät, rakennettujen embedien määrät "
            "sekä missing- ja stale-tilanteen. Auttaa myös hahmottamaan, onko tenantin "
            "tietopankki valmis vector-hakuun."
        ),
    ),
    CommandReference(
        group="Embeddin hallinta",
        command=(
            "python -m bot.app.operator "
            "knowledge embeddings-clear "
            "--tenant-id 1001 "
            "--yes"
        ),
        purpose=(
            "Poistaa tenantin embedit hallitusti uudelleenrakennusta varten. "
            "Tätä voidaan käyttää esim. silloin, kun embedin-malli vaihdetaan "
            "tai halutaan rakentaa koko vector-indeksi alusta."
        ),
    ),
    CommandReference(
        group="Embeding-jobien hallinta",
        command=(
            "python -m bot.app.operator "
            "embeddings jobs enqueue-missing "
            "--tenant-id 1001"
        ),
        purpose=(
            "Luo pending-tilaiset embeding-jobit chunkeille, joilta puuttuu "
            "ajantasainen embedding tai joiden embedding on stale."
        ),
    ),
    CommandReference(
        group="Embeding-jobien hallinta",
        command=(
            "python -m bot.app.operator "
            "embeddings jobs status "
            "--tenant-id 1001"
        ),
        purpose=(
            "Näyttää embeding-jobiputken statuksen. Näyttää "
            "pending-, running-, completed- ja failed-jobien määrät."
        ),
    ),
    CommandReference(
        group="Embeding-jobien hallinta",
        command=(
            "python -m bot.app.operator "
            "embeddings jobs run "
            "--tenant-id 1001 "
            "--limit 50" # <--- Limit
        ),
        purpose=(
            "Ajaa embeding-workerin 1 kerran ja käsittelee limitin mukaisen määrän "
            "pending-jobeja. Tätä voi käyttää manuaaliseen testaukseen tai ajastetun "
            "workerin logiikan tarkistamiseen."
        ),
    ),
    CommandReference(
        group="Embeding-jobien hallinta",
        command=(
            "python -m bot.app.operator "
            "embeddings jobs reset-stale-running "
            "--tenant-id 1001"
        ),
        purpose=(
            "Palauttaa jumiin jääneet running statuksen-jobit takaisin käsiteltäviksi. "
            "Pää käyttö on tilainteihin, jos/kun worker on kaatunut kesken ajon "
            "ja jobi ovat jääneet lukittuun tilaan."
        ),
    ),
)


def render_status_example() -> list[str]:
    # Tuotantokoodi voi tulostaa koneystävällisiä avain=arvo-pareja.
    #
    # Tällainen muoto on helppo lukea sekä ihmiselle että skriptille.
    # Esimerkiksi monitorointi, lokitus tai CI/CD-putki voi poimia näistä arvoista
    # embeding-putken tämänhetkisen tilanteen.
    return [
        "tenant_id=1001",
        "provider=ollama",
        "model=nomic-embed-text",
        "active_chunk_count=42",
        "embedded_chunk_count=40",
        "missing_embedding_count=2",
        "stale_embedding_count=0",
        "pending_jobs=2",
        "running_jobs=0",
        "failed_jobs=0",
    ]


def render_command_list() -> list[str]:
    # Apumetodi tekemään komennoista helposti tulostettava lista.
    #
    # Tuotantokoodissa saman tiedon voi esim. tulostaa CLI:n help-tekstinä,
    # vaikka dokumentaationa tai diagnostiikkakomennon yhteydessä.
    lines: list[str] = []

    current_group: str | None = None

    for item in VECTOR_COMMANDS:
        if item.group != current_group:
            current_group = item.group
            lines.append(f"\n# {current_group}")

        lines.append(item.command)
        lines.append(f"  - {item.purpose}")

    return lines