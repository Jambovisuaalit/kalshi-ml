# kalshi-ml

Kalshin KXBTC15M-sopimusten paper/live-kaupankäyntijärjestelmä. Python 3.11+,
Coinbase/Kraken-spot-data, yhdeksän kausaalista piirrettä ja
HistGradientBoostingClassifier. Krypto-markkinoiden oletussiru on **2**.
Lisenssi: kaupallinen **Proprietary / Single User**; katso [LICENSE](LICENSE).

**Mukana olevan mallin tila:** `models/hgb_btc15m_prod.joblib` on oikea
esikoulutettu **SPOT_PROXY-tutkimusmalli**. Tiedostonimi ei tarkoita todettua
tuotantokelpoisuutta. Sen synteettiset OOS-backtestit olivat kulujen jälkeen
tappiollisia. Alkuperä, mittarit ja rajat ovat [mallikortissa](models/MODEL_CARD.md).
Oikeita toimeksiantoja ei ole käytetty tämän julkaisun testaamiseen.

**OBI-päivityksen mallivaatimus:** version 0.2.0 mukana tullut kahdeksan
piirteen malli on säilytetty muuttamattomana tutkimushistoriaa varten. Tämä
koodi vaatii uuden yhdeksän piirteen mallin, ajallisen kalibroinnin ja
opetusdatasta lasketun volatiliteettiviitteen. Vanha malli hylätään sekä
paper- että live-ajossa. Älä päivitä käynnissä olevaa positiota hallitsevaa
daemonia kesken position; vanhan julkaisun riskisulut ovat edelleen sen vastuulla.
Katso [OBI-, volatiliteetti- ja kalibrointiohje](docs/ML_SAFETY.md).

GitHubin testit ja ostajapaketin julkaisukäytäntö: [julkaisuohje](docs/RELEASE.md).

## Pika-aloitus

Pura lähdekoodipaketti ja aja sen juuressa Python 3.11:llä tai uudemmalla:

```sh
python3.12 quickstart.py
source .venv/bin/activate
# Kun uusi malli on koulutettu aidolla OBI-historialla:
python main.py --mode paper --model models/hgb_btc15m_latest.joblib
```

Alustus luo paikallisen `.venv`-ympäristön, asentaa lukitut riippuvuudet ja
CLI-komennot, luo puuttuvan `.env`:n ja tarkistaa konfiguraation sekä mallin.
Olemassa olevaa `.env`:iä, mallia tai journalia ei korvata. Järjestelmän
Python-asennukseen ei asenneta paketteja. Asennus tarvitsee verkon.

```sh
python quickstart.py --check             # paikalliset tarkistukset, ei muutoksia
python quickstart.py --check --model models/hgb_btc15m_latest.joblib
python quickstart.py --check --network   # lisäksi julkinen markkinahaku
kalshi-ml-run --mode paper --model models/hgb_btc15m_latest.joblib --once
kalshi-ml-run --mode paper --model models/hgb_btc15m_latest.joblib --iterations 5
```

Paper käyttää tuotannon julkista tarjouskirjaa ja 1000 dollarin virtuaalisaldoa.
Se ei tarvitse Kalshi-avaimia, lähetä pörssitoimeksiantoja tai siirrä varoja.
Oletuskulu on 0,02 $ per toteutettu sopimus per puoli. Täyttö rajoittuu näkyvään
bid/ask-kokoon; passiivista jonotäyttöä ei keksitä. Lopeta Ctrl+C:llä.

## Konfiguraatio ja avaimet

`.env.example` sisältää asetuspohjan. Oletukset ovat:

```dotenv
KALSHI_ENV=PROD
KALSHI_TRADING_MODE=paper
KALSHI_LIVE_TRADING_ENABLED=false
KALSHI_SUBACCOUNT=0
KALSHI_EXCHANGE_INDEX=2
ALERTS_ENABLED=false
```

Tiukka Pydantic-konfiguraatio luetaan järjestyksessä oletukset → `.env` →
prosessin ympäristömuuttujat. Rahaluvut käsitellään Decimal-arvoina.
Suhteellinen avainpolku ratkaistaan `.env`-tiedoston hakemistosta.

Live-käytössä tarvitaan Kalshin Key ID ja oma RSA-/Ed25519-privaattiavain PEM-
muodossa. RSA-avaimen tulee olla vähintään 2048-bittinen. Demo- ja tuotantoavaimet
ovat erillisiä. Aseta `KALSHI_KEY_ID` ja `KALSHI_PRIVATE_KEY_PATH` esimerkiksi
polkuun `secrets/kalshi-prod.pem`. Älä lisää avaimen sisältöä lähdekoodiin.

```sh
chmod 600 .env secrets/kalshi-prod.pem
kalshi-ml check-config --check-credentials
```

`.gitignore`, `.dockerignore` ja julkaisun tiedostolista rajaavat salaisuudet,
journalit ja ajonaikaiset tiedot pois paketista. Älä silti jaa omaa työhakemistoasi
asiakkaalle sellaisenaan.

## Live-käyttö

Oikeita toimeksiantoja varten kaikki seuraavat asetukset tarvitaan:

```dotenv
KALSHI_ENV=PROD
KALSHI_TRADING_MODE=live
KALSHI_LIVE_TRADING_ENABLED=true
```

```sh
kalshi-ml-run --mode live
```

Komento voi lähettää oikeita toimeksiantoja. Paper ei vaihdu liveksi
itsestään. Live-kirjanpito perustuu pörssin vahvistamiin toteutumiin ja
`fee_cost`-kuluihin, ei paperin oletuskuluihin tai pelkkään kuittaukseen.
GTC-avauksen lepäävä jäännös perutaan. Poistuminen käyttää reduce-only IOC:ta.

## Strategia ja tilakone

| Ehto | Oletus |
| --- | --- |
| Avausikkuna | 180–800 sekuntia jäljellä |
| YES | P(YES) ≥ 0,55 ja delta ≥ +5 $ |
| NO | P(YES) ≤ 0,45 ja delta ≤ −5 $ |
| Ostettavan puolen ask | 45–66¢ |
| Take profit | bid ≥ 88¢ |
| Hard stop | hintatappio ≥ 14¢ |
| Delta stop | YES: delta ≤ 0; NO: delta ≥ 0 |

Yksi positio ja yksi avausyritys per sykli. Poistumisprioriteetti on
Delta-SL → Hard-SL → TP. Riskitarkistus tehdään ennen mallia. Malli- tai
kynttilävirhe estää avaukset mutta ei poista riskivalvontaa. TP/Hard-SL voivat
toimia ticker-häiriössä; Delta-SL tarvitsee tuoreen spotin.

Pollaus on nimellisesti 4 sekuntia. HTTP-viiveet voivat ylittää tämän eikä REST-
pollaus takaa välitöntä stop-toteutusta tai stop-hinnan täyttymistä. Puuttuvaa
hintaa/strikeä ei korvata nollalla. Syklin tulos kirjataan virallisesta
finalisoidusta payoutista; sitä ei päätellä Coinbasen spotista.

Statusrivi näyttää tilan, hinnan, P(YES):n, position, käteisen ja realisoidun
PnL:n. Kauppa-, intent-, osatäyttö-, settlement- ja virhetapahtumat kirjataan
kokonaisina UTC-aikaleimattuina riveinä ja SQLite-tapahtumina.

## Journalit ja palautuminen

```text
logs/paper.log
logs/paper-exchange-2.sqlite3
logs/live.log
logs/live-exchange-2.sqlite3
```

`--log-file` ja `--state` vaihtavat polut. Intent tallennetaan ennen lähetystä.
Uudelleenkäynnistys täsmäyttää alkuperäisen client_order_id:n eikä lähetä
avausta uudelleen. Älä poista live-journalia tuntemattoman toimeksiannon
kiertämiseksi. HALTED vaatii tilin ja journalin tarkistuksen.

Käytä yhtä omistavaa prosessia per tili/siru/strategia. Journalikohtainen
prosessilukko tukee macOS/Linux-ympäristöjä; eri state-polut eivät lukitse
toisiaan. Uudet paperin alkusaldot/kulut tarvitsevat uuden state-polun.

Ctrl+C/SIGTERM estää uudet avaukset ja yrittää täsmäyttää nykyisen intentin.
Avoimia positioita ei väkisin likvidoida sammutuksessa. Ne säilyvät journalissa,
mutta sammutettu prosessi ei valvo stoppeja.

## Telegram ja Discord

```dotenv
ALERTS_ENABLED=true
TELEGRAM_BOT_TOKEN=oma_bot_token
TELEGRAM_CHAT_ID=oma_chat_id
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/ID/TOKEN
```

Voit käyttää yhtä tai molempia palveluja. Telegram tarvitsee tokenin ja chat ID:n.
Kun hälytykset on erikseen aktivoitu, daemon lähettää kauppa-, settlement- ja
virhetiivistelmät näihin kohteisiin. `quickstart --check` vain validoi asetukset
eikä lähetä viestejä. Älä käytä esimerkkiarvoja.

Taustajono on rajattu, epäonnistuneita lähetyksiä ei uusita ja sammutuksen
jonopurku on rajattu kahteen sekuntiin. Hälytyksiä voi kadota; SQLite-journali
on kirjanpidon ensisijainen lähde. Tokenit ja webhookit ovat salaisuuksia.

## Varojen siirto 0 → 2

Esikatselu ei käytä tiliä eikä lähetä mitään:

```sh
python scripts/transfer_collateral.py --amount 100.00
```

Todellinen siirto on erillinen toiminto ja vaatii live-asetukset sekä UUID:n:

```sh
python scripts/transfer_collateral.py --amount 100.00 --request-id UUID_ESIKATSELUSTA --execute
```

Intent/kuittaus säilyy polussa `logs/transfers/<uuid>.json`. Hyväksyntä on
asynkroninen: tarkista siirtohistoria ja kohdesaldo ennen kauppaa. Älä toista
ratkaisematonta siirtoa uudella UUID:llä. Kalshin määräyksikkö on centicent:
100 $ = 1 000 000 centicent-yksikköä.
[Virallinen siirtorajapinta](https://docs.kalshi.com/api-reference/portfolio/intra-account-transfer).

## Koulutus ja raportointi

Erillinen neliluokkainen poistumien tutkimuslabeler ja sen proxy-rajoitukset:
[MULTICLASS_LABELS.md](docs/MULTICLASS_LABELS.md). Sen targetit eivät sovellu
nykyisen binäärisen P(YES)-mallin koulutukseen.

```sh
# Kerää ensin oikeaa, koulutuskynttilöiden kanssa päällekkäistä OBI-historiaa:
python -m kalshi_ml.data.orderbook --output data/raw/coinbase_obi.csv
# Lopeta keruu Ctrl+C:llä vasta riittävän historian jälkeen.
python scripts/train_model.py --candles 3000 --obi-csv data/raw/coinbase_obi.csv
python scripts/train_model.py --candles-csv data/raw/btc.csv --cycle-prices data/raw/official_cycles.csv --obi-csv data/raw/coinbase_obi.csv
python -m kalshi_ml.backtest.engine --candles-csv data/raw/btc.csv --obi-csv data/raw/coinbase_obi.csv
python scripts/view_performance.py --state logs/paper-exchange-2.sqlite3
kalshi-ml-performance --state logs/live-exchange-2.sqlite3 --json
```

Uudelleenkoulutus tallentaa oletuksena `models/hgb_btc15m_latest.joblib` eikä
korvaa toimitettua mallia. Valitse ajossa oma tiedosto `--model`-parametrilla.
Jaon yksikkö on kokonainen sykli ja testijakson alkuun ulottuvat opetusvasteet
poistetaan. Viimeiset 30 % ovat OOS-testiä. Niitä edeltävän 70 %:n viimeiset
20 % varataan sigmoidikalibroinnille; perusmalli ja volatiliteettiviite sovitetaan
vain sitä edeltävään lohkoon. Ilman virallista cycle-CSV:tä käytetään merkittyä SPOT_PROXY-vastetta.
`--end` rajaa historiaa. Puuttuvia minuutteja ei täytetä keksityillä hinnoilla.

Suorituskykyraportti avaa journalin vain luku -tilassa. Se näyttää kaupat,
nettotuloksen, kulut, Win %:n, Profit Factorin ja realisoidun pääomakäyrän
drawdownin. Todellista mark-to-market-drawdownia ei voi palauttaa ilman
historiallisia tarjouskirjahavaintoja.
Tarkat kaavat ja CSV-skeemat: [tekninen viite](docs/TECHNICAL_REFERENCE.md).

## Docker ja systemd

```sh
mkdir -p logs secrets
export KALSHI_UID=$(id -u) KALSHI_GID=$(id -g)
docker compose -f deploy/docker-compose.yml build
docker compose -f deploy/docker-compose.yml up -d
docker compose -f deploy/docker-compose.yml logs -f
docker compose -f deploy/docker-compose.yml down
```

Luo `.env` ensin. UID/GID:n tulee sallia avaimien lukeminen ja lokien
kirjoittaminen. Kontti käynnistyy paper-tilaan, ajaa ilman root-oikeuksia ja
mounttaa salaisuudet/mallit vain luku -tilassa. Live vaatii lisäksi compose-
komennon `--mode live` -muutoksen. Image ei sisällä omia avaimiasi/journalejasi.

`deploy/kalshi-bot.service` on Linuxin paper-palvelupohja. Asenna lähteet ja
venv polkuun `/opt/kalshi-ml`, luo `kalshi-bot`-palvelukäyttäjä ja asetukset
polkuun `/etc/kalshi-ml/kalshi.env`. Käytä avaimen absoluuttista polkua ja
rajoita lukuoikeus palvelukäyttäjälle. State/log-hakemistot luo systemd.

```sh
sudo cp deploy/kalshi-bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now kalshi-bot
sudo journalctl -u kalshi-bot -f
```

Palvelupohja sisältää `--mode paper`; asennus ei aktivoi live-kaupankäyntiä.
Dockeria tai systemd-palvelua ei käynnistetä quickstart-alustuksessa.

## Rakenne ja julkaisupaketti

```text
kalshi-ml/
|-- .env.example, .gitignore, .dockerignore
|-- README.md, LICENSE, THIRD_PARTY_NOTICES.md
|-- requirements.txt, pyproject.toml, quickstart.py, main.py
|-- kalshi_ml/
|   |-- config.py, daemon.py, runner.py, setup.py, cli.py
|   |-- data/          collector.py, features.py
|   |-- execution/     kalshi_client.py, collateral.py
|   |-- model/         trainer.py, inference.py, dataset.py
|   |-- backtest/      engine.py
|   |-- monitoring/    notifier.py
|   `-- storage/       journal.py
|-- models/            pretrained joblib, metadata, manifest, MODEL_CARD.md
|-- scripts/           train_model.py, transfer_collateral.py,
|                      view_performance.py, build_release.py
|-- deploy/            Dockerfile, docker-compose.yml, kalshi-bot.service
|-- docs/              TECHNICAL_REFERENCE.md
`-- tests/
```

```sh
python -m unittest discover -s tests -v
python -m pip check
python scripts/build_release.py
```

Julkaisuskripti tekee `dist/kalshi-ml-0.2.0.zip`:n sallittujen tiedostojen
listalta: lähteet, testit, dokumentit ja nimetty malli. Mukaan ei tule omia
asetuksia, avaimia, SQLite-tiedostoja, raakadataa, lokeja tai virtuaaliympäristöä.

Lisenssinantaja yksilöidään ostosopimuksessa/laskulla, koska virallista nimeä
 ei ole vielä valittu. LICENSE ei korvaa myyjän toimitus-, hinta-, vero- tai
asiakastietoja. Kolmansien osapuolten lisenssit säilyvät erillisinä.

## Kalshin tarjous- ja selvityshavaintojen keruu

Lue [havaintolokin ohje](docs/KALSHI_OBSERVATIONS.md). Käynnistä julkisen,
vain lukuja tekevän rajapinnan kerääjä komennolla
`python -m kalshi_ml.data.kalshi_observer --environment PROD --output data/raw/kalshi`.
Valitsin `--once` tekee yhden kierroksen. Paikallinen loki on Gitin ohittama.

Haaran julkaisutila ja jäljellä olevat datavaatimukset ovat
[julkaisuvalmiuden yhteenvedossa](docs/RELEASE_READINESS.md).
