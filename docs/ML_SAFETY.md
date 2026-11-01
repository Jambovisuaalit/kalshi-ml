# Yhdeksän piirrettä, volatiliteettivahti ja todennäköisyyskalibrointi

Tämä on v0.2.0:n jälkeinen skeemamuutos. Vanhaa tagia, mallipainoja tai
niiden suorituskykymittareita ei ole muutettu uudeksi yhdeksän piirteen malliksi.
Uutta aitoa OBI-historialla opetettua mallia ei toimiteta tämän muutoksen mukana.
Vanha artifact_version=1 hylätään; uusi versio on 2. Mallin puuttuessa
käynnistys/quickstart-tarkistus päättyy koodiin 2 ennen kaupankäyntiä.

## OBI:n kerääminen ja kohdistaminen

Piirteet järjestyksessä: `delta_usd`, `remaining_sec`, `z_score`, `vol_15m`,
`vol_60m`, `vol_ratio`, `mom_1m`, `mom_3m`, `obi`.
OBI = (sum bid BTC size - sum ask BTC size) / (sum bid BTC size + sum ask BTC size).
Molemmilta puolilta otetaan kymmenen parasta uniikkia hintatasoa Coinbasen
BTC-USD L2 REST -kirjasta, neljän sekunnin pollaussyklissä. Tyhjä, risteävä,
vajaa, virheellinen tai vanhentunut kirja estää uuden avauksen.
Kyseessä on pollaava snapshot, ei WebSocket-tason jatkuva tilauskirjarekonstruktio.
[Coinbase book API](https://docs.cdp.coinbase.com/api-reference/exchange-api/rest-api/products/get-product-book).

```sh
python -m kalshi_ml.data.orderbook --output data/raw/coinbase_obi.csv
# Rajattu julkisen datan tarkistus:
python -m kalshi_ml.data.orderbook --output data/raw/coinbase_smoke.csv --samples 3
```

CSV: `timestamp,received_at,bid_volume,ask_volume,sequence,depth`.
Ajat ovat aikavyöhykkeellisiä; `timestamp` tulee pörssistä ja `received_at`
kirjataan vastauksen saavuttua. Koulutus käyttää vain päätöshetkellä jo
vastaanotettua snapshotia; molempien aikojen ikä saa olla enintään 10 s.
Tulevaa kirjaa ei käytetä taaksepäin eikä puuttuvaa OBI:a korvata nollalla,
OHLCV-proxyllä tai nykyhetken kirjalla. Puuttuvan OBI:n näytteet ohitetaan;
ilman käyttökelpoista historiaa koulutus keskeytyy.

Kerää riittävän pitkä, candle-CSV:n kanssa päällekkäinen historia ja varmista
että erillisissä sovitus- ja kalibrointilohkoissa on molemmat lopputulosluokat.
3000 yhden minuutin kynttilää kattaa noin 50 tuntia, ei muutaman snapshotin
keruuta. Julkinen REST-päätepiste ei palauta tätä historiallista OBI-dataa jälkikäteen.

```sh
python -m kalshi_ml.model.trainer --candles 3000 --obi-csv data/raw/coinbase_obi.csv
python quickstart.py --check --model models/hgb_btc15m_latest.joblib
python main.py --mode paper --model models/hgb_btc15m_latest.joblib
```

## Kahden sigman avausraja

Nykyisen 15 minuutin syklin volatiliteettitilasto on

```
v = sqrt((sum(log(P_i/P_(i-1))**2)
          + sum(log(H_j/L_j)**2 / (4*log(2)))) / elapsed_minutes)
```

`P` alkaa ensimmäisen sulkeutuneen Coinbase-minuutin avauksesta, sisältää
syklin sulkeutuneet minuutin päätökset ja viimeisen spot-havainnon. `H/L`
sisältää vain jo sulkeutuneet, saatavilla olevat minuutit. Näin viimeisin
hyppy havaitaan ennen minuutin sulkeutumista ja sulkeutuneen minuutin
crash/recovery näkyy myös päätöshinnan palautuessa. Muodostuvan kynttilän
lopullisia high/low-arvoja ei lueta. Syklin ensimmäinen minuutti ei riitä
tilastolle eikä kuulu strategian avausikkunaan.

Viitteen keskiarvo `mu` ja otoskeskihajonta `sigma` lasketaan vain HGB:n
sovituslohkon vastaavista havainnoista. Kalibrointi- ja OOS-data eivät vaikuta
niihin. Avaus sallitaan vain, jos `abs(v-mu) <= 2*sigma`; myös poikkeuksellisen
matala volatiliteetti estää avauksen. Rajan yhtäsuuruus sallitaan, ja
nollavarianssissa sallitaan vain sama arvo numeerisella toleranssilla.
`STRATEGY_VOLATILITY_SIGMA_LIMIT` voi tiukentaa rajaa, enintään arvoon 2.
Puuttuva viite estää avauksen. Tämä ei takaa flash crashien havaitsemista
pollausten välissä eikä kahden sigman kaista tarkoita normaalijakaumaoletusta.

TP, Hard-SL ja Delta-SL käsitellään ennen OBI-hakua tai mallipäätelmää.
Volatiliteetti-, OBI- ja edge-suodattimet eivät estä suojaavaa sulkua.

## Ajallinen sigmoidikalibrointi ja hintaan sidottu etu

Kokonaiset syklit jaetaan kronologisesti 70 % ennen testiä / 30 % OOS-testiin.
Ensimmäisen osan viimeiset noin 20 % varataan erilliseksi kalibrointilohkoksi
(vähintään kaksi sykliä). Malli sovitetaan vanhempaan lohkoon; sen vasteet,
jotka eivät olleet saatavilla ennen kalibroinnin ensimmäistä päätöstä,
poistetaan. Myös kalibrointivasteiden on oltava tiedossa ennen OOS-alkua.
Suhteet ovat kokonaisista sykleistä ja purgauksesta johtuen likimääräisiä.

`CalibratedClassifierCV(FrozenEstimator(hgb), method="sigmoid")` sovitetaan
ainoastaan kalibrointilohkoon. Runtime ja backtest käyttävät sen
`predict_proba`-ulostuloa. Tallennetaan raaka- ja kalibroidut AUC/Brier/log-loss-
mittarit sekä OOS reliability -pisteet. Kalibroinnin ei oleteta parantavan
mittareita automaattisesti. OOS-tuloksia ei käytetä menetelmän valintaan tai
mallin uudelleensovitukseen.
[scikit-learn calibration](https://scikit-learn.org/stable/modules/calibration.html).

Kalibrointi on empiirinen luotettavuuskorjaus, ei täsmällisen todellisen
todennäköisyyden tai voitollisuuden takuu. Vanhojen P/delta/aika/hintarajojen
lisäksi vaaditaan:

```
owned_probability = p_yes for YES, 1-p_yes for NO
edge_cents = 100*owned_probability - rounded_executable_ask_cents
             - entry_fee_buffer_cents - exit_fee_buffer_cents
edge_cents >= minimum_edge_cents
```

Oletukset ovat 2 c avauskulu, 2 c sulkukulun varaus ja vähintään 1 c
ennustettu etu sopimusta kohti. Live-portti tarkistetaan uudelleen
tili/saldohakujen jälkeen ennen pysyvää toimeksiantoaikomusta. Paper- ja
backtest-kulujen ylittäessä varaukset portti käyttää suurempia kuluja.
Kulupuskureiden on katettava todellinen tilikohtainen kulurakenne;
live-kulujen automaattista hakua ei tässä lisätä.

Esimerkiksi p=0.55 ja YES ask=66 c tuottavat -15 c arvioidun edun
kulujen jälkeen ja estyvät. Ask=50 c tuottaa samoilla oletuksilla +1 c.
Tämä on settlement-arvon odotusarvo kustannusvarauksella: TP/SL-politiikan
todellinen polkuriippuvainen tuotto vaatii erillisen OOS-backtestin.

Synteettinen replay syöttää malliin vain havaittua OBI:a. Puuttuvan OBI:n
välipisteessä avaus estyy, mutta synteettinen hintapiste säilyy riskisuluille.
Z-score-hintaproxy on erillinen mallipäätelmästä eikä käytä OBI:a. Replay
ei ole täydellinen flash-crash- tai toteutumasimulaatio.
`BacktestEngine` vaatii `volatility_reference`-viitteen ja snapshotin
`cycle_volatility`-arvon avatakseen kaupan; puuttuvat arvot estävät avauksen.
