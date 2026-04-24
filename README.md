# Vector-haku ja embeddingit

Tämä paketti kokoaa yhteen ai-botin vector-hakuun, embeddingeihin ja hybridihakuun liittyvät osat. Tarkoitus ei ole olla täydellinen kopio koko tuotantokoodista, vaan selkeä referenssi siitä, miten haku toimii ja mitkä tiedostot liittyvät kokonaisuuteen.

## Mitä vector-haku tarkoittaa ai-botissa?

Tietopankin haku voidaan ajaa kolmella eri tavalla:

- `keyword` – perinteinen avainsanahaku
- `vector` – semanttinen haku embedding-vektoreilla
- `hybrid` – keyword- ja vector-haun yhdistelmä

Vector-haku ei poista keyword-hakua käytöstä. Sen tarkoitus on tuoda haun rinnalle semanttinen taso, joka löytää osumia myös silloin, kun käyttäjä ei käytä samoja sanoja kuin dokumentissa.

Semanttinen taso tarkoittaa tässä sitä, että haku ei etsi vain samoja sanoja, vaan yrittää löytää samaa tarkoittavia sisältöjä. Jos dokumentissa lukee esimerkiksi “salasanan palautus” ja käyttäjä kysyy “miten saan uuden kirjautumistunnuksen”, vector-haku voi silti ymmärtää, että aiheet liittyvät toisiinsa.

Eli tarkoituksena on, että käyttäjä voi kysyä asian eri tavalla kuin miten se on kirjoitettu tietopankkiin. Keyword-haku ei välttämättä löydä silloin hyvää osumaa, mutta vector-haku voi tunnistaa, että kysymys ja dokumentin sisältö tarkoittavat suunnilleen samaa asiaa.

## Mitä embedding tarkoittaa ai-botissa?

Embedding tarkoittaa tekstistä muodostettua numeerista vektoria. Käytännössä teksti muutetaan lukujonoksi, jota voidaan vertailla toisiin teksteihin matemaattisesti.

Ai-botissa embeddingit toimivat näin:

- knowledge-dokumentti pilkotaan pienempiin chunkkeihin
- jokaisesta aktiivisesta chunkista voidaan luoda embedding
- embedding tallennetaan tietokannan `knowledge_embeddings`-tauluun
- käyttäjän kysymyksestä luodaan oma embedding
- kysymyksen embeddingiä verrataan chunkkien embeddingeihin kosinisimilariteetilla

Tämän avulla voidaan arvioida, mitkä dokumentin osat ovat merkitykseltään lähimpänä käyttäjän kysymystä.

## Miten knowledge chunk muuttuu embeddingiksi?

Yksinkertaistettuna prosessi menee näin:

1. Dokumentti tallennetaan `knowledge_documents`- ja `knowledge_chunks`-tauluihin.
2. Chunkille lasketaan `content_hash`, jonka avulla voidaan tunnistaa sisällön muutokset.
3. `EmbeddingService` pyytää AI-bridgeä kutsumaan Ollaman embeddings-rajapintaa.
4. Ollama palauttaa tekstistä muodostetun vektorin.
5. Vektori tallennetaan `knowledge_embeddings`-tauluun yhdessä providerin, mallin ja `content_hash`-arvon kanssa.
6. Jos chunkin sisältö muuttuu myöhemmin, myös `content_hash` muuttuu.
7. Kun hash ei enää täsmää vanhaan embeddingiin, embedding tulkitaan vanhentuneeksi eli stale-tilaan.

Tällä yritetään varmistetaa, ettei haku käytättäisi vanhoja embeddingejä muuttuneeseen sisältöön.

## Miten käyttäjän kysymys haetaan vector-haulla?

Vector-haussa käyttäjän kysymystä ei verrata dokumentteihin pelkkinä sanoina. Sen sijaan sekä kysymys että tietopankin chunkit muutetaan embedding-vektoreiksi, joita voidaan vertailla matemaattisesti.

Yksinkertaistettuna haku etenee näin:

1. käyttäjän kysymys siistitään ja normalisoidaan
2. kysymyksestä luodaan embedding-vektori
3. tietokannasta haetaan saman tenantin aktiiviset chunkit, joilla on ajantasainen embedding
4. kysymyksen embeddingiä verrataan jokaisen chunkin embeddingiin
5. jokaiselle chunkille lasketaan similarity-arvo
6. similarity-arvo muunnetaan hakupisteiksi
7. riittävän vahvat osumat lisätään mukaan promptin kontekstiksi

Lopputuloksena ai-botti saa käyttöönsä ne tietopankin kohdat, jotka ovat merkitykseltään lähimpänä käyttäjän kysymystä.

### 1. Kysymyksestä luodaan embedding

Kun käyttäjä kysyy esimerkiksi:

```text
Miten käyttäjä voi palauttaa salasanansa?
```

kysymys muutetaan ensin embedding-vektoriksi.

Yksinkertaistettuna vektori voidaan ajatella lukulistana:

```text
q = [0.12, -0.04, 0.88, 0.31, ...]
```

Tässä:

- `q` tarkoittaa käyttäjän kysymyksen embedding-vektoria
- jokainen luku kuvaa pientä osaa tekstin merkityksestä
- todellisuudessa vektorissa voi olla satoja tai tuhansia ulottuvuuksia

Ihminen ei yleensä tulkitse näitä lukuja suoraan. Oleellista on, että samanlaiset merkitykset päätyvät embedding-avaruudessa lähelle toisiaan.

### 2. Chunkilla on oma embedding

Myös jokaisella tietopankin chunkilla on oma embedding-vektori.

Esimerkiksi tietopankissa voi olla chunkki:

```text
Salasanan voi uusia kirjautumissivun "Unohtuiko salasana?" -linkistä.
```

Tästä muodostetaan oma vektori:

```text
c = [0.10, -0.02, 0.84, 0.29, ...]
```

Tässä:

- `c` tarkoittaa chunkin embedding-vektoria
- samaan aiheeseen liittyvät tekstit saavat yleensä samansuuntaisia vektoreita
- tekstien ei tarvitse käyttää täsmälleen samoja sanoja, jotta ne voivat olla lähellä toisiaan

Tämän takia vector-haku voi löytää osuman, vaikka käyttäjä kysyy “salasanan palautuksesta” ja dokumentissa puhutaan “salasanan uusimisesta”.

### 3. Vektoreita verrataan kosinisimilariteetilla

Kun kysymyksellä ja chunkilla on omat vektorinsa, niitä verrataan kosinisimilariteetilla.

Kosinisimilariteetti mittaa kahden vektorin välistä suuntaa. Tässä yhteydessä se tarkoittaa sitä, kuinka samaan “merkityssuuntaan” käyttäjän kysymys ja tietopankin chunkki osoittavat.

Kaava on:

$$
\text{cosine\_similarity}(q, c) =
\frac{q \cdot c}{\|q\| \|c\|}
$$

Missä:

- `q` = käyttäjän kysymyksestä muodostettu embedding-vektori
- `c` = tietopankin chunkista muodostettu embedding-vektori
- `q · c` = vektorien pistetulo
- `||q||` = kysymysvektorin pituus
- `||c||` = chunk-vektorin pituus

Käytännössä kaava kysyy:

```text
Kuinka samaan suuntaan nämä kaksi vektoria osoittavat?
```

Jos vektorit osoittavat lähes samaan suuntaan, niiden merkitys on todennäköisesti samankaltainen. Jos ne osoittavat eri suuntiin, yhteys on heikompi.

### 4. Pistetulo kertoo, kuinka paljon vektorit liikkuvat samaan suuntaan

Kaavan yläosa on pistetulo:

$$
q \cdot c
$$

Pistetulo lasketaan kertomalla vektorien vastaavat arvot keskenään ja laskemalla tulokset yhteen.

Yksinkertaistettu esimerkki:

$$
q = [0.2, 0.4, 0.1]
$$

$$
c = [0.3, 0.5, 0.2]
$$

Pistetulo lasketaan näin:

$$
q \cdot c =
(0.2 \times 0.3) +
(0.4 \times 0.5) +
(0.1 \times 0.2)
$$

$$
q \cdot c =
0.06 + 0.20 + 0.02 = 0.28
$$

Mitä suurempi pistetulo on, sitä enemmän vektoreilla on samaa suuntaa. Pelkkä pistetulo ei kuitenkaan vielä riitä, koska pitkät vektorit voisivat saada suurempia arvoja vain kokonsa takia.

Siksi tulos jaetaan vektorien pituuksilla.

### 5. Vektorien pituudet normalisoivat tuloksen

Kaavan alaosa on:

$$
\|q\| \|c\|
$$

Tämä tarkoittaa, että kysymysvektorin ja chunk-vektorin pituudet kerrotaan keskenään.

Yhden vektorin pituus lasketaan näin:

$$
\|q\| =
\sqrt{q_1^2 + q_2^2 + q_3^2 + ... + q_n^2}
$$

Yksinkertaisella vektorilla:

$$
q = [0.2, 0.4, 0.1]
$$

pituus on:

$$
\|q\| =
\sqrt{0.2^2 + 0.4^2 + 0.1^2}
$$

$$
\|q\| =
\sqrt{0.04 + 0.16 + 0.01}
$$

$$
\|q\| =
\sqrt{0.21}
$$

Tämä normalisointi tekee vertailusta reilumman. Silloin ei mitata sitä, kuinka “isoja” vektorit ovat, vaan sitä, kuinka samaan suuntaan ne osoittavat.

### 6. Similarity-arvo kertoo osuman vahvuuden

Kosinisimilariteetin tulos on yleensä välillä `-1` ja `1`.

Tulkinta on yksinkertaistettuna:

- `1` tarkoittaa, että vektorit osoittavat samaan suuntaan
- `0` tarkoittaa, että vektoreilla ei ole selvää yhteistä suuntaa
- `-1` tarkoittaa, että vektorit osoittavat vastakkaisiin suuntiin

Tekstihaussa kiinnostavia ovat yleensä arvot, jotka ovat lähempänä arvoa `1`.

Esimerkiksi:

```text
0.91 = erittäin vahva osuma
0.78 = hyvä osuma
0.52 = mahdollisesti käyttökelpoinen osuma
0.20 = yleensä heikko osuma
0.00 = ei selvää yhteyttä
```

Nämä rajat eivät ole universaaleja. Sopiva raja riippuu embedding-mallista, dokumenttien sisällöstä ja siitä, kuinka tarkkaa haun halutaan olevan.

### 7. Similarity voidaan muuttaa pisteiksi

Kun similarity-arvo on laskettu, se voidaan muuttaa hakupisteiksi. Tämä helpottaa tulosten järjestämistä ja yhdistämistä hybrid-haussa.

Yksinkertainen pisteytys voi näyttää tältä:

$$
\text{vector\_score} =
\max(0, \text{cosine\_similarity}) \times 100
$$

Jos similarity on esimerkiksi:

```text
0.82
```

niin pisteet ovat:

$$
0.82 \times 100 = 82
$$

Tällöin chunkin vector-score olisi:

```text
82
```

`max(0, ...)` tarkoittaa, ettei negatiivisia similarity-arvoja oteta mukaan pisteytykseen negatiivisina pisteinä. Jos similarity on alle nollan, se käsitellään käytännössä nollana.

### 8. Heikot osumat voidaan suodattaa pois

Kaikkia osumia ei kannata lisätä promptiin. Jos mukaan päästetään liian heikkoja osumia, ai-botti voi saada kontekstiinsa asiaan liittymätöntä tietoa.

Siksi haussa voidaan käyttää minimirajaa:

$$
\text{cosine\_similarity}(q, c) \geq \text{min\_similarity}
$$

Esimerkiksi:

```text
min_similarity = 0.60
```

Tällöin vain chunkit, joiden similarity on vähintään `0.60`, pääsevät mukaan jatkokäsittelyyn.

Jos chunkin similarity on:

```text
0.72
```

se hyväksytään.

Jos similarity on:

```text
0.41
```

se jätetään pois.

### 9. Parhaat osumat lisätään promptiin

Kun jokaiselle chunkille on laskettu similarity ja pisteet, tulokset järjestetään parhaasta heikoimpaan.

Yksinkertaistettuna:

$$
\text{top\_chunks} =
\text{sort\_by\_score}(\text{chunks})
$$

Tämän jälkeen promptiin lisätään vain rajattu määrä parhaita osumia.

Esimerkiksi:

```text
top_k = 5
```

Tällöin ai-botti saa vastauksensa tueksi enintään viisi parhaiten käyttäjän kysymykseen sopivaa tietopankin kohtaa.

Tämä pitää promptin tiiviimpänä ja vähentää riskiä, että mukaan päätyy turhaa tai harhaanjohtavaa sisältöä.

### Yhteenveto

Vector-haun idea on muuttaa sekä käyttäjän kysymys että tietopankin sisältö embedding-vektoreiksi. Sen jälkeen järjestelmä mittaa kosinisimilariteetilla, kuinka lähellä nämä vektorit ovat merkitykseltään.

Yksinkertaistettuna:

$$
\text{kysymys} \rightarrow \text{embedding} \rightarrow q
$$

$$
\text{chunk} \rightarrow \text{embedding} \rightarrow c
$$

$$
\text{similarity} =
\frac{q \cdot c}{\|q\| \|c\|}
$$

Mitä korkeampi similarity-arvo on, sitä todennäköisemmin chunkki liittyy käyttäjän kysymykseen. Parhaat osumat annetaan ai-botille kontekstiksi, jotta se voi muodostaa vastauksen tietopankin sisällön perusteella.


## Miten hybrid-haku toimii?

Hybrid-haku yhdistää kaksi eri hakutapaa:

- keyword-score
- vector-score

Keyword-score perustuu siihen, kuinka hyvin hakusanat löytyvät dokumentista. Vector-score taas perustuu siihen, kuinka lähellä kysymys ja dokumentin sisältö ovat merkityksen perusteella.

Lopullinen pisteytys lasketaan painotettuna yhdistelmänä:

- `hybrid_keyword_weight`
- `hybrid_vector_weight`

Chunk voi päästä mukaan kolmella eri tavalla:

- se löytyy vain keyword-haulla
- se löytyy vain vector-haulla
- se löytyy molemmilla tavoilla, jolloin pisteet yhdistetään

Hybrid-haun idea on hyödyntää molempien hakutapojen vahvuuksia. Keyword-haku on hyvä täsmällisiin termeihin, kuten nimiin, komentoihin ja teknisiin arvoihin. Vector-haku taas toimii paremmin, kun käyttäjän kysymys on muotoiltu eri tavalla kuin dokumentin sisältö.

## Miksi keyword fallback on edelleen mukana?

Keyword fallback on ai-botissa tärkeä varmistusmekanismi.

Järjestelmä voi palata keyword-hakuun esimerkiksi silloin, kun:

- vector-haku ei ole tenantille sallittu
- embedding-palvelu ei ole käytettävissä
- embeddingejä ei ole vielä rakennettu
- vector- tai hybrid-haku ei löydä riittävän hyviä osumia
- tenantin hakuympäristö on vasta osittain käytössä

Tämän tarkoitus on, että ai-botti pystyy vastaamaan kysymyksiin myös, kun semanttinen haku ei ole mahdollista. Tämä pyrkii varmistamaan, että botin vastaukset olisivat turvallisia ja tietoisia. Esimerkiksi, ai-botin komento `/prompt`-polku ei hajoa, vaikka embedding-puoli ei olisi vielä täysin valmis tai käytettävissä pystyy Ai-botti edelleen hakemaan tietoa perinteisellä keyword-haulla.

## Miten tenant-eristys toimii?

Tenant-eristys on mukana koko hakuketjussa.

(tent ja tena = tenant)

Kaikki oleelliset haut rajataan tenantin mukaan:

- dokumentit haetaan `tent_id`:n perusteella
- chunkit haetaan `tent_id`:n perusteella
- embeddingit haetaan `tent_id`:n perusteella
- embedding-jobit haetaan `tent_id`:n perusteella
- näkymät on rakennettu tenant-kohtaisen näkyvyyden ympärille

Vector-haussa tämä on erityisen tärkeää. Semanttinen haku ei saa koskaan verrata käyttäjän kysymystä toisen tenantin sisältöön.

Toisin sanoen yhden tenantin kysymys saa osua vain kyseisen tenantin omiin dokumentteihin ja embeddingeihin.

## Miten embedding-jobit toimivat?

Embeddingit voidaan rakentaa kahdella tavalla:

- suoraan komennolla `knowledge embeddings-build`
- taustajobien kautta komennolla `embeddings jobs ...`

Jobiputki toimii näin:

1. Järjestelmä tunnistaa chunkit, joilta puuttuu embedding tai joiden embedding on vanhentunut.
2. Nämä chunkit lisätään `embedding_jobs`-tauluun.
3. Worker hakee seuraavan pending-tilassa olevan jobin.
4. Worker tarkistaa vielä ennen ajoa, että chunk on edelleen aktiivinen.
5. Worker tarkistaa myös, että chunkin sisältö vastaa edelleen samaa `content_hash`-arvoa.
6. Embedding rakennetaan.
7. Tulos upsertoidaan `knowledge_embeddings`-tauluun.
8. Job merkitään lopuksi johonkin lopputilaan.

Mahdollisia lopputiloja ovat esimerkiksi:

- `completed`
- `skipped`
- `cancelled`
- `failed`

Tarkistukset ennen embeddingin raentamista on tärkeitä, koska chunk on voinut muuttua tai poistua sen jälkeen, kun job lisättiin jonoon.

## Mitkä tiedostot kannattaa lukea ensin?

Suositeltu lukujärjestys:

1. `README.md`
2. `vector_retrieval_reference.py`
3. `embedding_service_reference.py`
4. `bridge_client_reference.py`
5. `ai_bridge_embedding_reference.py`
6. `embedding_jobs_reference.py`
7. `runtime_wiring_reference.py`
8. `operator_cli_reference.py`
9. `diagnostics_reference.py`
10. `tests_reference.py`
11. `source_map.md`

Tämä järjestys alkaa kokonaiskuvasta ja etenee vähitellen kohti toteutusta, tietokantamallia, CLI-komentoja, diagnostiikkaa ja testejä.

## Mitkä osat ovat yksinkertaistettuja kopioita?

Seuraavat tiedostot ovat yksinkertaistettuja referenssiversioita:

- `vector_retrieval_reference.py`
- `embedding_jobs_reference.py`
- `operator_cli_reference.py`
- `diagnostics_reference.py`
- `tests_reference.py`
- `runtime_wiring_reference.py`

Näissä tiedostoissa on pyritty säilyttämään varsinainen logiikan runko, mutta poistamaan ympäriltä projektin muuta kuormaa. Tarkoitus on, että lukija pystyy seuraamaan vector-haun, embeddingien ja jobiputken toimintaa ilman, että hänen täytyy ensin ymmärtää koko sovelluksen rakennetta.

## Yhteenveto

Tämän referenssipaketin tärkein ajatus on näyttää, miten semanttinen haku on rakennettu ai-botin sisälle turvallisesti ja vaiheittain.

Keyword-haku toimii edelleen perustasona. Vector-haku tuo mukaan merkityspohjaisen haun. Hybrid-haku yhdistää molemmat. Embedding-jobit mahdollistavat sen, että embeddingejä voidaan rakentaa hallitusti myös taustalla.

Tenant-eristys kulkee mukana koko toteutuksen läpi, jotta haku pysyy aina oikean asiakkaan datassa.