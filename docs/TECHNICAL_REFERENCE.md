# Technical reference

The OBI/calibration schema supersedes the v0.2.0 eight-feature inference and
entry rules below. See [ML_SAFETY.md](ML_SAFETY.md) for nine-feature training,
two-sigma volatility filtering, fee-adjusted edge and required model migration.

Detailed strategy, API and backtest documentation. For release installation,
CLI defaults, deployment and the bundled model, follow the root README.md.

## Datankeruu ja piirteet

`MarketDataCollector.fetch_candles(start, end)` palauttaa suljetut 1m-kynttilät
puoliavoimelta väliltä `[start, end)`. Rajojen tulee olla minuutintarkkoja ja
aikavyöhykkeellisiä. Indeksi on UTC-minuutin **alkuaika**. Coinbase-haut
jaetaan 250 minuutin sivuihin, viimeinen sivu voi olla lyhyempi.
Puuttuva minuutti, ristiriitainen duplikaatti tai noutovirhe vaihtaa koko
pyydetyn ikkunan Krakeniin. Lähteitä ei yhdistetä yhden ikkunan sisällä.
Lähde on `df.attrs["source"]`. Krakenin 720 viimeisimmän rivin raja tarkoittaa,
ettei varalähteellä voi täydentää mielivaltaisen vanhaa opetusdataa.

HTTP-lukupyynnöillä on rajattu eksponentiaalinen backoff ja jitter.
`Retry-After` hyväksyy sekunnit tai HTTP-päivämäärän. Jos palvelimen edellyttämä
odotus ylittää budjetin, pyyntö epäonnistuu eikä sitä uusita liian aikaisin.
`fetch_ticker()` palauttaa `SpotTicker`-olion Coinbasen viimeisimmästä kaupasta
ja torjuu yli 10 sekuntia vanhan hinnan oletuksena.

```python
import pandas as pd

from kalshi_ml.data.collector import MarketDataCollector, RetryPolicy

end = pd.Timestamp.now(tz="UTC").floor("min")
with MarketDataCollector(retry=RetryPolicy(max_retries=3)) as collector:
    candles = collector.fetch_candles(end - pd.Timedelta(minutes=90), end)
    ticker = collector.fetch_ticker()
```

Kerää kynttilät inkrementaalisesti; historiallista REST-dataa ei ole tarkoitus
ladata kokonaan jokaisella ticker-pollauksella. Konstruktorin `RetryPolicy`
voi muodostaa konfiguraation `http_timeout_sec`- ja `http_max_retries`-arvoista.
Ticker-haku ja kynttilähaku ovat erilliset pyynnöt: ajosilmukan tulee varmistaa,
että kynttiläikkuna kattaa tickerin aikaleimaa edeltävän minuutin myös
minuutinvaihteen yli.

### Viralliset hinnat ja opetusdatan aikarajat

Coinbasen tai Krakenin kynttilän avaushinta **ei ole** Kalshin virallinen strike.
`CyclePrices` pitää virallisen `strike_open`-arvon erillään myöhemmin julkaistavasta
`cycle_settle`-arvosta ja sisältää kummankin havaittavuusaikaleiman.
Syötä nämä arvot kyseisen Kalshi-sopimuksen virallisesta metadatasta/tuloksesta;
niiden Kalshi-noutaja toteutetaan pörssiadapterin yhteydessä.

```python
from kalshi_ml.data.features import (
    CyclePrices, bind_cycle_prices, compute_features, cycle_start,
)

# official_strike ja strike_observed_at tulevat Kalshi-metadatasta.
start = cycle_start(ticker.timestamp)
official = CyclePrices(
    cycle_start_ts=start,
    strike_open=official_strike,
    strike_available_at=strike_observed_at,
    source="kalshi:<market-ticker>",
)
candles = bind_cycle_prices(candles, [official])
features = compute_features(candles, float(ticker.price), ticker.timestamp, start)
```

Puuttuvaa tai päätöshetken jälkeen havaittua strikeä ei korvata spot-proxyllä:
piirrelaskenta nostaa `FeatureDataError`-virheen. `bind_cycle_prices` säilyttää
viralliset tietueet muistissa DataFrame-attribuuttina; tallenna ne erillisenä
aineistona ja liitä uudelleen CSV/Parquet-latauksen jälkeen.
`CyclePrices.settlement_delta(as_of)` palauttaa `cycle_settle - strike_open`
vasta tuloksen julkaisun jälkeen. Tämä erotus on opetusvastetta varten,
eikä sitä käytetä piirrelaskennassa. Tasatuloksen ja pyöristyksen säännöt
kuuluvat myöhempään sopimuskohtaiseen label-moduuliin.

`aggregate_15m(df, as_of)` muodostaa täydet UTC-jaksot 00/15/30/45 ja nimeää
spot-hinnat `spot_open`/`spot_close`, jotta niitä ei sekoiteta virallisiin hintoihin.
Osittaiset jaksot ja aukolliset jaksot jätetään pois.

`compute_features` tarvitsee 61 peräkkäistä suljettua 1m-kynttilää.
Sulkeutumaton tai tuleva kynttilä ei vaikuta piirteisiin. Valinnainen
`available_at`-sarake mahdollistaa todellisten dataviiveiden huomioinnin;
ilman sitä oletetaan saatavuus minuutin päättyessä. Historiallinen OHLCV ei
itsessään todista todellista saatavuutta, joten tallenna vastaanottoajat live-ajossa.
Historiallisessa opetusajossa `current_spot` täytyy ottaa kyseisen päätöshetken
havainnosta, eikä syklin lopullisesta päätöshinnasta.

| Piirre | Täsmällinen määritelmä |
| --- | --- |
| `delta_usd` | spot − virallinen strike |
| `remaining_sec` | syklin loppu − päätöshetki |
| `vol_15m` | viimeisten 15 suljetun 1m-logtuoton otoskeskihajonta × √15 |
| `vol_60m` | viimeisten 60 suljetun 1m-logtuoton otoskeskihajonta × √60 |
| `vol_ratio` | vol_15m / vol_60m; täysin tasaiselle historialle 1 |
| `z_score` | delta / max(0.01 USD, spot × std_15 × √(remaining_sec/60)) |
| `mom_1m` | spot / viimeinen sulkeutunut hinta hetkellä t−1min − 1 |
| `mom_3m` | spot / viimeinen sulkeutunut hinta hetkellä t−3min − 1 |

Volatiliteetit ja momentit ovat suhteellisia lukuja. Momentin vertailuhetki
pyöristyy alaspäin minuutin sulkeutumisrajaan. Z-luvun yhden sentin
nimittäjälattia pitää tuloksen äärellisenä nollavolatiliteetissa.
Nämä määritelmät on säilytettävä samoina opetus- ja live-ajossa.

Rajapintalähteet: [Coinbase candles](https://docs.cdp.coinbase.com/api-reference/exchange-api/rest-api/products/get-product-candles),
[Coinbase ticker](https://docs.cdp.coinbase.com/api-reference/exchange-api/rest-api/products/get-product-ticker),
[Kraken OHLC](https://docs.kraken.com/api-reference/market-data/get-ohlc-data).

## Mallin koulutus ja historiallinen backtest

```sh
python -m kalshi_ml.model.trainer --candles 3000
python -m kalshi_ml.backtest.engine --candles 3000
```

Ensimmäinen komento noutaa 3000 suljettua 1m-kynttilää, muodostaa piirteet,
kouluttaa HGB-mallin, tulostaa OOS-mittarit ja ajaa molemmat backtest-skenaariot.
Se ei lähetä pörssitoimeksiantoja. `--skip-backtest` tekee vain koulutuksen.
`--max-iter 150` on oletus. `.env`:n strategiarajoja käytetään backtestissä.
Toinen komento lataa tallennetun mallin ja käyttää vain sen opetushetken
jälkeisiä havaintoja. `--model` ja koulutuksen `--output` vaihtavat mallipolun.
`--end 2026-10-30T19:55:00Z` rajaa latauksen toistettavasti annetun minuutin
alkuun (eksklusiivinen). Oletus päättyy minuutin nykyistä minuuttirajaa aiemmin,
jotta juuri sulkeutuneen kynttilän REST-julkaisulle jää aikaa.
Jos pitkässä ikkunassa on puuttuvia minuutteja, nouto keskeytetään;
valitse aukoton `--end`-ikkuna tai toimita tarkistettu paikallinen CSV.
Puuttuvia kynttilöitä ei täytetä keinotekoisilla hinnoilla.

Tallennettavat tulokset:

```text
data/raw/btc_usd_1m_latest.csv
models/hgb_btc15m_latest.joblib
models/hgb_btc15m_latest.metadata.json
models/hgb_btc15m_latest.oos.csv
models/hgb_btc15m_latest.backtest.txt
models/hgb_btc15m_latest.trades.csv
```

Joblib-paketti sisältää `model`, `scaler=None` ja `metadata`.
Puumallille ei soviteta turhaa skaalainta. Metatiedot sisältävät piirteiden
järjestyksen ja version, hyperparametrit, kirjastoversiot, aineiston tiivisteen,
train/test-aikarajat, viimeisen käytetyn vasteen julkaisuajan, hintojen lähteen
sekä AUC-ROC-, Brier Score- ja Log Loss -mittarit. Vakioennusteen vertailumalli
käyttää vain opetusjoukon YES-osuutta. AUC ilmoitetaan puuttuvaksi, jos koko
testijoukossa on vain yksi luokka. Brier Score ja Log Loss lasketaan silti.
OOS-CSV sisältää tarkistettavat ennusteet. Mallia ei soviteta uudelleen
testidatalla mittauksen jälkeen. Backtestin tekstitiedosto sisältää molemmat
raportit sekä käytetyt strategia- ja toteutusparametrit; trades-CSV sisältää
molempien skenaarioiden toteutuneet simulaatiokaupat.

**Jaon yksikkö on kokonainen 15 minuutin sykli.** `TimeSeriesSplit`:n viimeinen
laajeneva jako varaa noin 70 % sykleistä opetukseen ja uusimmat 30 % testiin.
Yhden syklin rivejä ei koskaan jaeta molemmille puolille. Syklirajojen pyöristys
ja saatavuuspurku voivat muuttaa rivimäärien suhdetta. Opetusrivi poistetaan,
jos sen vaste julkaistiin vasta ensimmäisen testihavainnon hetkellä tai sen
jälkeen. HGB:n `early_stopping=False` estää satunnaisen sisäisen validointijaon.
Hyperparametreja tai strategiarajoja ei optimoida testijaksolle.

### Virallinen aineisto ja eksplisiittinen spot-proxy

Pelkkä Coinbase/Kraken-data ei sisällä Kalshin virallisia strikejä, selvityksiä
tai tarjouskirjaa. Siksi `--candles 3000` toimii oletuksena selvästi merkityssä
`SPOT_PROXY`-tutkimustilassa: syklin ensimmäinen spot-open on proxy-strike ja
viimeinen spot-close on proxy-settlement. Avaushinta tulee havaittavaksi vasta
ensimmäisen minuutin sulkeuduttua. Nämä eivät ole Kalshin toteutuneita tuloksia.
3000 minuuttia on noin 50 tuntia: se on putken käyttökoe, ei todiste mallin
yleistymisestä tai kaupankäynnin kannattavuudesta.

Virallinen aineisto annetaan näin:

```sh
python -m kalshi_ml.model.trainer \
  --candles-csv data/raw/btc_usd_1m_latest.csv \
  --cycle-prices data/raw/official_cycles.csv
python -m kalshi_ml.backtest.engine \
  --candles-csv data/raw/btc_usd_1m_latest.csv \
  --cycle-prices data/raw/official_cycles.csv
```

Kynttilä-CSV:n sarakkeet ovat `timestamp,open,high,low,close,volume` ja
valinnainen `available_at`. Indeksi on UTC-minuutin alkuaika; kaikki ajat
sisältävät aikavyöhykkeen. Viivästyneet havainnot käsitellään julkaisuhetken
mukaan, eikä vanhaa kynttilää käytetä seuraavan syklin ajantasaisena spotina.
Puuttuvista tai liian myöhään saatavista riveistä ei keksitä piirteitä.

Virallisten syklihintojen CSV:n sarakkeet ovat:

```text
cycle_start_ts,strike_open,strike_available_at,cycle_settle,settle_available_at,source
```

Source-kenttä tunnistaa virallisen sopimuksen/lähteen. Opetusvaste on
`cycle_settle > strike_open`. Täsmälliset tasatulokset jätetään pois eikä niille
arvata sopimuskohtaista tulosta. Varmista hintojen pyöristys ja vertailusääntö
kyseisen sopimuksen ehdoista ennen virallisen aineiston käyttöä. Valmiiksi
vahvistettu binääritulos voidaan antaa suoraan backtestin `Settlement`-oliolle.

### Toteutussimulaation oletukset

`BacktestEngine.run` hyväksyy aikajärjestyksessä olevat `MarketSnapshot`-oliot
ja erilliset `Settlement`-tulokset. Snapshot sisältää spotin, striken,
OOS-ennusteen, YES/NO bid/ask-hinnat sekä saatavilla olevan sopimusmäärän.
Toisiaan vastaavien single-book-hintojen tulee olla komplementaarisia.
Syötä vain päätöshetkellä tunnettuja havaintoja: rajapinta ei voi itsenäisesti
todistaa ulkopuolelta annetun todennäköisyyden tai hintojen kausaalisuutta.
`inference_not_before` torjuu opetusjaksolle ulottuvan replayn.

CLI käyttää synteettisiä tarjoushintoja myös silloin, kun strike/settlement on
virallinen. Keskiarvohinta on `NormalCDF(z_score)`, **ei koulutetun mallin
ennuste**. Keskihinta pyöristetään senttiin ja rajataan välille 1–99¢; bid on
keskihinta −1¢ ja ask +1¢. YES- ja NO-spread on täsmälleen 2¢.
Tämä on hintamallin skenaario, ei Kalshin toteutuneen likviditeetin backtest.

Kynttilänsisäisestä tuntemattomasta järjestyksestä ajetaan molemmat vaihtoehdot:
`open → low → high → close` sekä `open → high → low → close`, oletetuilla
0/20/40/60 sekunnin aikaleimoilla. Jaetulla minuuttirajalla edellisen minuutin
close saa etusijan seuraavan minuutin openiin nähden. Ääriarvojen sijoittaminen
sekunteihin on skenaario-oletus; tarkka todellinen järjestys vaatii tick-dataa.
Tulevan kynttilän close/high/low ei päädy päätöshetken historiallisiin piirteisiin.
Molemmat raportit tulostetaan eikä tuottavampaa polkua valita tulokseksi.

Toteutussäännöt:

- Aloituspääoma 1000 $, yksi sopimus ja enintään yksi avoin positio.
- Enintään yksi avaus per sykli; avaus ostettavan puolen askista.
- Kulu 0,02 $ per sopimus **per toteutettu toimeksianto**: osto + myynti 0,04 $.
  Tämä on pyydetty kuluskenaario, ei ajantasaisen Kalshi-hinnaston toteutus.
- Settlement maksaa 0 tai 1 $ per sopimus ilman myyntitoimeksiantoa tai poistumiskulua.
- Delta-SL tarkistetaan ensin, sitten Hard-SL, sitten TP. STOP toteutuu
  senhetkisestä bidistä; hinnan hypätessä ei oleteta täyttöä stop-rajalla.
- IOC-poistuminen jää toteutumatta, jos näkyvä sopimusmäärä ei riitä.
  Se voidaan yrittää seuraavassa snapshotissa uudelleen. Ei osatäyttöjä.
- GTC-avaus oletetaan täytetyksi vain heti markkinakelpoisena riittävään
  ask-kokoon. Passiivisen jonon täyttöjä, latenssia tai piilotettua likviditeettiä
  ei oleteta. Synteettisessä tilassa koko on yksi sopimus.
- `ExecutionConfig` mahdollistaa sopimusmäärän, alkupääoman, kulun ja
  epäedullisen lisäslippagen vaihtamisen ohjelmallisesti.
- Käteisraja sisältää ostohinnan sekä varauksen osto- ja myyntikuluille;
  vivutusta ei käytetä edes nollahintaisessa poistumisessa.
- Avoin positio aineiston lopussa aiheuttaa virheen, ellei sen julkaistua
  settlement-tulosta voida käsitellä annettuun päättymisaikaan mennessä.

Raportti näyttää kauppamäärän, kulujen jälkeisen Win %:n, nettotuloksen,
Profit Factorin, Max Drawdownin sekä TP/Hard-SL/Delta-SL/Settlement-erittelyn.
Profit Factor käyttää nettovoittojen ja nettotappioiden suhdetta. Drawdown
mitataan jokaisessa saatavilla olevassa snapshotissa bid-hintaan merkitystä
likvidaatioarvosta poistumiskuluineen. Erääntynyt mutta vielä julkaisematon
positio merkitään konservatiivisesti nollaan, kunnes payout on saatavilla.
Ei kauppoja → Profit Factor N/A; voittoja mutta ei tappioita → inf.

Lataa joblib-malleja vain luotetusta lähteestä: formaatti voi suorittaa koodia.
Lataaja tarkistaa mallin tyypin, sklearn-version ja piirteiden version/järjestyksen.
Koulutuksessa käytetyt lähdeaineistot ja mallipaketti on säilytettävä yhdessä
toistettavuutta varten.

Mallikirjaston lähteet:
[HistGradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html),
[TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html),
[probability metrics](https://scikit-learn.org/stable/modules/model_evaluation.html).

## Kalshi V2 REST -asiakas

`kalshi_ml.execution.kalshi_client.KalshiClient` tukee osoitteita
`https://external-api.kalshi.com` (PROD) ja
`https://external-api.demo.kalshi.co` (DEMO). PEM-avaimen todellinen tyyppi
valitsee RSA-PSS/SHA-256- tai Ed25519-allekirjoituksen. RSA-avaimen tulee olla
vähintään 2048-bittinen. Salattu PEM voidaan avata `password`-bytes-parametrilla;
salasanaa ei tallenneta konfiguraatioon tai lokiin.

Julkinen markkinahaku toimii ilman tunnisteita ja ilman kaupankäynnin aktivointia:

```python
from kalshi_ml.config import KalshiEnvironment, load_settings
from kalshi_ml.execution.kalshi_client import KalshiClient

with KalshiClient(environment=KalshiEnvironment.DEMO) as client:
    market = client.get_active_15m_market()

# Avaimet luetaan olemassa olevasta .env-konfiguraatiosta.
# Paper-tila sallii saldohaut mutta estää send_order-kutsun.
with KalshiClient.from_settings(load_settings()) as client:
    balance = client.get_balance()
    available_dollars = balance.balance_dollars  # Decimal, ei float
```

`get_balance()` pyytää `subaccount=0` ja konfiguroidun `exchange_index`-arvon
(oletuksena 2) ja valitsee
`balance_breakdown`-listalta täsmälleen yhden valitun sirun tietueen. Sen `balance`
on dollarimääräinen merkkijono. Jos breakdown on vastauksessa mutta valittu siru
puuttuu tai toistuu, syntyy `InvalidResponseError`; yhteenlaskettuun saldoon
ei vaihdeta. Breakdownin puuttuessa käytetään eksplisiittisesti valittuun siruun
rajattua `balance_dollars`-kenttää tai vanhaa senttimääräistä `balance`-kenttää.
Desimaalit säilyttävät myös sentin murto-osat. `portfolio_value_dollars` on
erillinen kenttä; sitä ei lasketa käytettävissä olevaksi käteiseksi.

`get_active_15m_market()` sivuttaa kaikki vastaukset ja valitsee lähimpänä
sulkeutuvan **jo alkaneen** UTC-varttisyklin. Tilaksi hyväksytään `open` tai
`active`; tuleva sykli, vanhentunut/ratkaistu/keskeytetty markkina, provisional
sekä muu kuin konfiguroitu siru rajataan pois. Listauksen aikaisempi open_time ei
korvaa syklin alkua (`close_time - 15 min`). Ei sopivaa markkinaa → `None`.
Virheellinen vastausskeema tai numero aiheuttaa poikkeuksen. Puuttuva strike
ja puuttuvat tarjoushinnat jäävät arvoiksi `None`, eivät nollaksi.

Markkinan `exchange_index` luetaan API-vastauksesta ja palautetaan
`ActiveMarket.exchange_index`-kentässä. Puuttuvan kentän oletus on 2. Asiakas
valitsee konfiguroidun sirun markkinan, jotta saldo ja toimeksiannot kohdistuvat
samaan siruun; kovakoodattua sirun 0 vaatimusta ei ole.
Markkinan löytyminen ei yksin todista tarjoushintojen tuoreutta tai tilin oikeuksia.

`safe_to_cents("0.5600")` palauttaa 56. Vanha senttikenttä `"56"` tarvitsee
`unit="cents"`; yksikköä ei arvata luvun suuruudesta. Tarkka muunnos hylkää
sentin murto-osat. Markkinatarjouksissa bid pyöristetään alas ja ask ylös, jotta
kokonaissenttinen hintaraja ei parane virheellisesti pyöristyksessä.

### Toimeksiantokutsu ja vastaus

Metodin allekirjoitus on
`send_order(ticker, target_side, is_closing, count, price_cents, *, client_order_id=None)`.
`target_side` on `"YES"` tai `"NO"`, `count` positiivinen kokonaisluku ja
`price_cents` kokonaisluku välillä 1–99. Hinta tarkoittaa aina valitun oman
YES/NO-puolen hintaa myös suljettaessa.

| Toiminto | Single-Book side | Lähetetty YES-hinta | TIF | reduce_only |
| --- | --- | --- | --- | --- |
| YES-osto | bid | p / 100 | good_till_canceled | false |
| NO-osto | ask | (100 − p) / 100 | good_till_canceled | false |
| YES-sulku | ask | p / 100 | immediate_or_cancel | true |
| NO-sulku | bid | (100 − p) / 100 | immediate_or_cancel | true |

Päätepiste on `/trade-api/v2/portfolio/events/orders`. `price` lähetetään
esimerkiksi merkkijonona `"0.5600"` ja `count` merkkijonona `"1.00"` nykyisen
V2-skeeman mukaan. Mukana ovat aina `subaccount: 0`, `exchange_index` (oletuksena 2),
`self_trade_prevention_type: "taker_at_cross"` ja yksilöllinen `client_order_id`.
`OrderResult` sisältää täsmälliset `fill_count`- ja `remaining_count`-arvot:
IOC-vastaus tai vastaanottokuittaus ei tarkoita koko position sulkeutumista.

Paper-tilassa kaikki lähetykset estetään. Demo-lähetys tarvitsee DEMO-hostin
ja `KALSHI_TRADING_MODE=demo`. Tuotantolähetys tarvitsee PROD-hostin,
`KALSHI_TRADING_MODE=live` ja `KALSHI_LIVE_TRADING_ENABLED=true`.
Asiakkaan luonti tai konfiguraation muuttaminen ei itsessään lähetä toimeksiantoja.

### Virheiden käsittely

GET-pyynnöt käyttävät aikakatkaisua, rajattua eksponentiaalista backoffia ja
jitteriä. 408/429/5xx-tilanteita ja verkkovirheitä voidaan uusia. Jokainen uusi
yritys saa uuden allekirjoituksen ja senhetkisen millisekuntiaikaleiman.
`Retry-After` huomioidaan sekunteina tai HTTP-päivämääränä; yli budjetin menevää
odotusta ei lyhennetä ennenaikaiseksi uusinnaksi. 401/403 aiheuttaa
`AuthenticationError`-poikkeuksen ilman automaattista uusintaa.

**POST lähetetään vain kerran.** Aikakatkaisu, yhteysvirhe, 408/409/5xx,
uudelleenohjaus tai tulkitsematon onnistumisvastaus aiheuttaa
`OrderSubmissionUnknown`-poikkeuksen. Sen `client_order_id` täytyy ensin
täsmäyttää pörssin toimeksiantoihin/toteutumiin. Pääsilmukan täsmäytys on
`daemon.py`-moduulissa. Älä kutsu send_order-metodia automaattisesti
uudella ID:llä: aikaisempi toimeksianto on saattanut toteutua.
Myöskään saman ID:n uudelleenlähetykselle ei luvata dokumentoimatonta
deduplikointia. POSTin 429 aiheuttaa `RateLimitError`-poikkeuksen, jossa ovat
`client_order_id` ja `retry_after_sec`; sitä ei uusita automaattisesti.

HTTP-uudelleenohjauksia ei seurata, TLS-varmennus vaaditaan ja session
adapterien piilotetut retryt torjutaan. Asiakas omistaa vain itse luomansa
session; injektoitu sessio jää kutsujan suljettavaksi. Käytä yhtä asiakasta
per säie. Oletussessio ei käytä ympäristön proxy-/netrc-asetuksia; tarvittaessa
injektoi organisaation asetuksilla rakennettu `requests.Session`.
Älä lokita allekirjoitettuja pyyntöjä, avaintietoja tai kokonaisia API-vastauksia.

Rajapintalähteet: [authentication](https://docs.kalshi.com/getting_started/quick_start_authenticated_requests),
[Create Order V2](https://docs.kalshi.com/api-reference/orders/create-order-v2),
[balance](https://docs.kalshi.com/api-reference/portfolio/get-balance),
[markets](https://docs.kalshi.com/api-reference/market/get-markets).

## Päävalvontaprosessi

Käynnistä projektin juuresta Python 3.11+ -virtuaaliympäristössä:

```sh
python main.py --mode paper
```

Paper käyttää **tuotannon** julkista tilauskirjaa ilman Kalshi-avaimia,
virtuaalisaldoa 1000 $ ja oletuskulua 0,02 $ per toteutettu sopimus per puoli.
Konfiguraation DEMO-oletus ei vaihda paperin hintalähdettä demoon.
Alkusaldo ja simulaatiokulu ovat vaihdettavissa uuden paper-journalin luonnissa:

```sh
python main.py --mode paper --paper-cash 500 --paper-fee 0.02 --contracts 1
python main.py --mode paper --once
python main.py --mode paper --iterations 5
```

Live-käynnistys käyttää oikeaa tiliä ja oikeita toimeksiantoja. Se tarvitsee
`.env`:ssä `KALSHI_ENV=PROD`, `KALSHI_TRADING_MODE=live`,
`KALSHI_LIVE_TRADING_ENABLED=true` sekä tuotannon Key ID:n ja privaattiavaimen:

```sh
python main.py --mode live
```

`--model` valitsee luotetun mallipaketin, oletus on
`models/hgb_btc15m_latest.joblib`. Malli ladataan kerran käynnistyksessä;
sen piirre- ja sklearn-versiot tarkistetaan. Mallin opetuslähde
(`SPOT_PROXY`/`OFFICIAL_PRICES`) kirjataan START-lokiin. Pääsilmukka käyttää
aina Kalshin virallista havaittua strikeä eikä korvaa sitä spot-proxyllä.
`--env-file` valitsee asetustiedoston. Pääsilmukka käyttää
`KALSHI_EXCHANGE_INDEX`-asetusta (oletus 2) markkinahaussa, saldossa,
toimeksiannoissa, peruutuksissa ja täsmäytyksessä. Jos valitulla sirulla ei ole
sopivaa markkinaa, tilaksi tulee WAITING_MARKET.

### Tilat, pollaus ja piirteet

Tilat ovat `WARMING`, `WAITING_MARKET`, `FLAT`, `OPENING`, `OPEN`, `CLOSING`,
`SETTLING`, `HALTED` ja `STOPPED`. Neljän sekunnin pollaus käyttää monotonista
kelloa. Jos pollaus kestää liian pitkään, väliin jääneitä kutsuja ei ajeta
peräkkäisenä purskeena. Pääsilmukan HTTP-lukupyynnöissä on enintään 3 sekunnin
connect- ja read-aikakatkaisut ilman sisäistä retryä; seuraava pollaus yrittää
uudelleen. Muun REST-asiakkaan normaali retry-politiikka säilyy ennallaan.
Useat peräkkäiset HTTP-kutsut voivat silti ylittää neljän sekunnin välin.
REST-pollaus ei takaa sekunnin murto-osien stop-toteutusta.

Jokainen pollaus tarkistaa ensin keskeneräisen toimeksiannon ja erääntyneet
settlementit. Reaaliaikainen Coinbase-ticker ja avoimen position tarjouskirja
palvelevat riskipoistumisia ennen kynttilöiden päivitystä tai mallia.
TP ja Hard-SL voivat toimia myös ticker-häiriössä; Delta-SL vaatii tuoreen spotin.
Malli- tai kynttilävirhe ei poista positiota muistista eikä poista riskivalvontaa.

Suljetut 1m-kynttilät haetaan 75 minuutin ikkunassa kerran minuutinvaihteessa,
eivät jokaisella ticker-pollauksella. Epäonnistunut päivitys voidaan uusia
seuraavalla pollauksella. Koko ikkuna säilyttää yhden lähteen ja vastaanottoajan.
Piirteet lasketaan nykyisen päätöshetken saatavilla olevista suljetuista
kynttilöistä ja aikaleimatusta virallisesta strikestä. `predict_proba` käyttää
täsmälleen mallin kahdeksan piirteen järjestystä. Puuttuva, vanhentunut tai
virheellinen havainto estää uuden avauksen. Spotin ja kirjan tuoreusraja on
oletuksena 8 sekuntia. Avausehdot ja aikaikkuna tarkistetaan ennen lähetystä.

Syklin vaihtuessa vanha sykli siirretään odottamaan virallista tulosta ja uusi
sykli lukitaan omalla tickerillään, aikarajoillaan ja strikellään. Striken
muuttuminen lukituksen jälkeen pysäyttää kaupankäynnin HALTED-tilaan.
Kalshin `closed` tai `determined` ei vielä merkitse maksettua settlementiä:
PnL päivitetään vasta `settled`/`finalized`-tilan virallisesta payoutista.
Myös mahdollinen osittainen/mitätöity payout säilytetään desimaalina.
Vanha positio estää uuden avauksen, kunnes pääoma on täsmäytetty.
Muidenkin havaittujen syklien lopullinen tulos kirjataan, vaikka kauppaa ei tehty.

### Toteutukset ja PnL

Avaus käyttää samoja aika-, todennäköisyys-, delta- ja hintarajoja kuin
konfiguraatio/backtest. Enintään yksi avausyritys sallitaan per sykli ja yksi
positio kerrallaan. Riskipoistumisten prioriteetti on Delta-SL → Hard-SL → TP.
Hard-SL vertaa omistetun puolen toteutuneeseen keskimääräiseen ostohintaan;
hintaraja ei sisällä kuluja. PnL sisältää kulut.

- Paperin täyttö käyttää näytettyä askia/bidiä ja enintään kyseisen tason kokoa.
  Passiivista jonotäyttöä tai puuttuvaa likviditeettiä ei keksitä.
- Live avaa GTC-toimeksiannon. Heti vastaanottokuittauksen jälkeen mahdollinen
  lepäävä jäännös perutaan ja lopullinen toimeksianto sekä toteutumat luetaan.
  Avausta ei jätetä lepäämään seuraavien pollausten yli tarkoituksellisesti.
- Live-IOC-poistuminen käyttää `reduce_only=True`. Kokonaislukumäärä pyöristetään
  tarvittaessa ylöspäin, mutta pörssin reduce-only-raja estää osasopimusposition
  ylittämisen. Kirjanpito vähentää vain vahvistetun toteutuneen määrän.
- Vastaanottokuittaus ei muuta saldoa. Lopullinen filled-määrä ja deduplikoitu
  täyttörekisteri on saatava täsmäämään ennen kirjausta. Pörssin mahdollinen
  toteutumarekisterin viive jättää tilan OPENING/CLOSING-tilaan odottamaan.
- Live-PnL käyttää valitun YES/NO-puolen toteutushintoja ja `fee_cost`-kuluja;
  paperin 0,02 $ oletusta ei käytetä live-kuluina.
- Osasulun entry-kustannuspohja kohdistetaan suhteessa suljettuun määrään.
  Jäljelle jäävää positiota valvotaan ja suljetaan myöhemmissä pollauksissa.
- Settlementissä jäljellä oleva määrä arvotetaan viralliseen oman puolen
  payoutiin. Tulos kirjataan täsmälleen kerran, eikä spotin suunta ratkaise sitä.

Statuskonsoli päivittää `\r`-rivillä tilan, ajan, tickerin, jäljellä olevat
sekunnit, spotin, P(YES):n, position, käteisen ja realisoidun PnL:n.
Täydelliset UTC-aikaleimatut START/CYCLE/ORDER/FILL/SETTLEMENT/STOP-tapahtumat
tulostetaan omille riveilleen ja kirjoitetaan lokiin. Kustannuspohjan,
todellisten kulujen, osatäyttöjen ja PnL-muutosten rahaluvut ovat `Decimal`-arvoja.
Live-tilan cash on tämän prosessin täsmäytetty käteisledger; ulkopuolisia
talletuksia tai muiden instrumenttien kauppoja ei tulkita strategian voitoksi.
Uuden avauksen saatavuus tarkistetaan myös pörssin saldosta.

### Jatkaminen ja hallittu lopetus

Oletustiedostot:

```text
logs/paper.log
logs/paper-exchange-2.sqlite3
logs/live.log
logs/live-exchange-2.sqlite3
```

`--state` ja `--log-file` vaihtavat polut. SQLite-journalissa ovat checkpoint
ja täydet tapahtumat. Samalle journalille sallitaan vain yksi prosessi.
Tämä lukitustoteutus tukee macOS/Linux-ympäristöjä. Pidä live-tili/siru/series
yhden omistavan prosessin hallinnassa; eri state-polut eivät lukitse toisiaan.
Sirut käyttävät erillisiä oletusjournaleja; vanhaa sirun 0 journalia ei
siirretä sirulle 2. Journalin identiteetti sitoo tilan moodiin, tiliin, siruun ja kokoon/
paperin saldo- sekä kuluasetuksiin. Olemassa oleva saldo palautuu journalista.
Konfiguraation vaihtaminen ei nollaa PnL:ää.

Toimeksiantoaikomus commitoidaan FULL-durability-tilassa **ennen** POST-kutsua.
Jos prosessi kaatuu, jatkava prosessi etsii saman `client_order_id`:n pörssistä
ja lukee/peruu/täsmäyttää alkuperäisen toimeksiannon. Se ei lähetä avausta
uudelleen. Kun lopullinen toteutuma ja sen hinta/kulut tunnetaan, kirjanpito ja
intent-merkinnän poisto tehdään samassa SQLite-transaktiossa.
Jos ID:tä ei vielä löydy, uudet toimeksiannot pysyvät estettyinä. Älä poista
journalia tämän ohittamiseksi; todellinen tuntematon lopputulos vaatii
tilin/toimeksiantojen tarkistuksen pörssin puolelta.

Live-käynnistys sekä avaukset ja poistumiset tarkistavat hallitun position ja
lepäävät KXBTC15M-toimeksiannot konfiguroidulla sirulla. Avoimelle positiolle tehdään lisäksi
määräajoin täsmäytys. Ulkopuolista positiota/toimeksiantoa ei adoptoida
automaattisesti. Poikkeama tai muuttunut strike tuottaa HALTED-tilan ja
vaatii operatorin tarkistuksen. Tiedostoon jo kirjattu HALTED ei katoa restartissa.
Käytä erillistä paper-journalia tutkimukseen; live-journalia ei saa nollata,
ennen kuin sen toimeksiannot ja tilipositiot on selvitetty.

Ctrl+C ja SIGTERM pyytävät hallittua lopetusta. Nykyinen intent yritetään
täsmäyttää/perua, mutta uutta positiota ei enää avata. Tunnettu avoin positio
ei saa automaattista markkinasulkua sammutuksessa: se ja mahdollinen
keskeneräinen intent säilytetään journalissa ja STOP-lokissa seuraavaa
käynnistystä varten. Sammutettuna prosessi ei valvo stoppeja. Levyvirhe estää
uuden lähetysten tekemisen ja säilyttää viimeisen kestävän intentin.

Lisärajapinnat:
[orderbook](https://docs.kalshi.com/getting_started/orderbook_responses),
[order state](https://docs.kalshi.com/api-reference/orders/get-order),
[cancel V2](https://docs.kalshi.com/api-reference/orders/cancel-order-v2),
[fills](https://docs.kalshi.com/api-reference/portfolio/get-fills),
[positions](https://docs.kalshi.com/api-reference/portfolio/get-positions).
