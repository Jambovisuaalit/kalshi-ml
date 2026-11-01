# Neliluokkainen triple-barrier-labeler

`kalshi_ml/model/labeler.py` tuottaa erilliset kohteet hypoteettisille YES-
tai NO-avauksille. Luokat kuvaavat omistetun puolen poistumaa:

| target | Nimi | Poistuma |
|---|---|---|
| 0 | HIT_SL | Hard-SL tai Delta-SL ennen TP:tä/erääntymistä |
| 1 | HIT_TP | Take-profit ennen SL:ää/erääntymistä |
| 2 | EXP_0 | Pidetty erääntymiseen, omistetun puolen voitto-osuus 0 c |
| 3 | EXP_100 | Pidetty erääntymiseen, omistetun puolen voitto-osuus 100 c |

Tämä on OHLCV-/hintaproxyyn perustuva tutkimustarget. Minuutin OHLCV ei
sisällä Kalshin toteutettavia osto-/myyntihintoja, kirjan syvyyttä, fill-
viivettä tai high/low-havaintojen järjestystä ja tarkkaa aikaa. Labeler ei
todista oikeita poistumia eikä korvaa todelliseen tarjouskirjaan perustuvaa
triple-barrier-labelointia tai backtestiä. Kukin rivi on itsenäinen
hypoteettinen avaus; rivit eivät muodosta yhden position kauppahistoriaa.

## Käyttö

```sh
python -m kalshi_ml.model.labeler \
  --candles-csv data/raw/btc_usd_1m_latest.csv \
  --side YES \
  --output data/processed/yes_labels.csv

python -m kalshi_ml.model.labeler \
  --candles-csv data/raw/btc_usd_1m_latest.csv \
  --cycle-prices data/raw/official_cycles.csv \
  --side NO \
  --output data/processed/no_labels.csv
```

Funktiorajapinta:

```python
from kalshi_ml.model.labeler import generate_multiclass_labels

labels = generate_multiclass_labels(
    candles,
    tp_cents=88,
    hard_sl_distance_cents=14,
    use_delta_sl=True,
    side="YES",
    cycle_prices=official_records,  # Mapping[Timestamp, CyclePrices], tai None proxy-ajossa
)
```

Syötteessä tarvitaan `open,high,low,close` sekä joko aikavyöhykkeellinen
DatetimeIndex tai `timestamp`-sarake. Numeerinen timestamp tarkoittaa Unix-
sekunteja; merkkijonojen pitää sisältää aikavyöhyke. `volume` saa olla mukana.
Indeksin ja aikaleimojen on oltava uniikkeja. Järjestämätön syöte lajitellaan
aikaleiman mukaan ja alkuperäinen rivi-indeksi säilytetään tuloksessa.

## Korjatut aikaleimat ja volatiliteetti

- Timestamp tarkoittaa minuutin **avausta**. Rivin `m` close on tiedossa vasta
  `cycle_start + (m+1)*60s`. Rajat 180..800 s jäljellä sallivat indeksit 1..11
  eli päätökset 120..720 s syklin alusta. Rajat ovat inklusiivisia.
- Vaaditaan täsmälleen 15 peräkkäistä, sulkeutunutta ja saatavilla olevaa
  minuuttia. Vajaat, aukolliset tai `as_of`-hetkellä keskeneräiset syklit
  ohitetaan. Tyhjä tulos säilyttää dokumentoidut sarakkeet ja targetin int8-tyypin.
- `vol_15m` jäädytetään päätösrivin arvoon. Koko syklin mediaania ei lueta.
  Annetun sarakkeen pitää olla laskettu kausaalisesti. Jos sarake puuttuu,
  lasketaan vain 15 aiemmasta minuuttituotosta `std(log_return, ddof=1)*sqrt(15)`;
  tämä edellyttää 16 peräkkäistä päätöshavaintoa ja saatavuustarkistusta.
- Projektin volatiliteetin yksikkö on suhteellinen, esimerkiksi 0.001 tarkoittaa
  0.1 %:n 15 minuutin hajontaa. Proxy käyttää dollarimääräistä hajontaa
  `strike * vol_15m * sqrt(remaining_sec/900)` ja 0.01 dollarin numeerista
  nimittäjän lattiaa. Jo valmiiksi dollarimääräiselle volatiliteetille on
  eksplisiittinen `volatility_unit="usd"` / `--volatility-unit usd`.
- `available_at` ja `volatility_available_at` ovat valinnaisia tietojen
  vastaanotto-/julkaisuaikoja. Myöhässä saapuvaa entry-kynttilää ei käytetä
  avaukseen: seuraavan minuutin kokonainen high/low ei kuvaa enää yksin
  viivästyneen päätöksen jälkeistä polkua. Puuttuva aika olettaa saatavuuden
  välittömästi minuutin sulkeutuessa; tämä on tutkimusajon ajoitusoletus.

## Strike, settlement ja hintapolun epävarmuus

Ilman `cycle_prices`-tietueita avaushinta on spot-proxy ja settlement on viimeisen
minuutin close. `label_mode="SPOT_AND_QUOTE_PROXY"` erottaa tämän virallisista
hinnoista. Erillisistä `CyclePrices`-tietueista käytetään saatavilla olevaa
strikea ja julkaistua settlementiä; markkinan puuttuvaa tietuetta ei korvata
spot-hinnalla. Näissäkin riveissä osto-/myyntihinnat ovat **QUOTE_PROXY**.
Tasan strikessä tapahtuva settlement ohitetaan oletuksena; `settlement_tie`
voi olla `yes` tai `no`, kun sopimuksen YES-tasatulos on vahvistettu.

Entry on midpoint-proxy + puolet spreadista ja sulun bid on midpoint-proxy
- puolet spreadista; spreadin oletus on 2 c. Hinnat rajataan 0..100 c.
Take-profit, hard-stop ja delta-stop koskevat omistettua YES-/NO-puolta.
Rivit, joilla delta-stop on voimassa jo entry-hetkellä, saavat välittömän SL-
labelin. Strategian P-, delta- ja 45..66 c entry-suodattimia ei lisätä labeleriin;
ne voivat rajata näytteitä erikseen vain entry-hetkellä tunnetuista tiedoista.

Tulevan palkin tiedetty avausgap käsitellään ensin. Jos TP osuu jo avauksessa,
myöhempi saman minuutin SL ei muuta sitä jälkikäteen SL:ksi. Muuten palkin
high/low arvioidaan eksplisiittisesti **palkin avaushetken jäljellä olevalla
ajalla** ja entryssä jäädytetyllä volatiliteetilla. Tämä on approksimaatio:
aikakato minuutin sisällä ja erityisesti viimeisen minuutin osumat eivät ole
täsmällisesti palautettavissa OHLCV:stä. Kun TP ja SL ovat mahdollisia samassa
palkissa, SL voittaa ja `ambiguous_barrier=True`. Lippu kuvaa tätä törmäystä,
ei kaikkia hintaproxyn epävarmuuksia.

Tulos sisältää `entry_price_proxy`, `cycle_strike`, `decision_ts`,
`exit_price_proxy`, `exit_time_lower`, `exit_time_upper` ja
`label_available_at`. Intrabar-osumalle ei keksitä sekunnintarkkaa aikaa.
SL-range-osuman hintakenttä on epäedullinen proxyraja, ei toteutushinta.
Labelin saatavuus huomioi myös sitä edeltävien palkkien myöhästyneen
vastaanoton. OOS-jaossa/purgauksessa käytetään `label_available_at`-kenttää.

Piirredatasettiin liittämisessä käytetään täsmällistä `decision_ts`-hetkeä
sekä sykliä ja omistettua puolta. OHLCV:n avaustimestamp ei ole päätöshetki.
Jaa aineisto kokonaisina sykleinä ja poista opetuksesta vasteet, joiden
saatavuus ulottuu kalibrointi-/testijaksoon.

## Mallin kytkentä on erillinen vaihe

Oletus-Delta-SL tekee `EXP_0`-luokasta tässä spot-proxyssa rakenteellisesti
tyhjän: jos viimeinen close on striken väärällä puolella, myös viimeisen
minuutin adverse-extreme käy siellä ja konservatiivinen range-sääntö antaa
SL-labelin. Oikeassa toteutuksessa sulku voi jäädä täyttymättä, spotin ja
virallisen settlementin välillä voi olla ero tai käännös tapahtua vasta
erääntymishetkellä. OHLCV ei ratkaise näitä tapauksia. Älä keksi puuttuvaan
luokkaan havaintoja tai oleta neljän luokan esiintyvän. Luokkajakauma pitää
tarkistaa ennen koulutusta; `EXP_0` voidaan tutkia esimerkiksi erikseen
Delta-SL pois kytkettynä, mikä on eri poistumapolitiikka.

Neliluokkainen target ei ole `P(YES)`. Nykyinen binäärikouluttaja hylkää
`target_schema="triple-barrier-four-class-v1"`-rivit myös silloin, kun pienessä
otoksessa esiintyy vain luokat 0 ja 1. Schema tallennetaan CSV-sarakkeeseen,
jotta DataFrame-attribuuttien katoaminen ei poista tätä tarkistusta.

Erillinen neliluokkainen malli tarvitsee oman kalibroinnin, side-ehdollistuksen
ja tuottoarvon validoinnin. Pelkät neljä luokkatodennäköisyyttä eivät määrää
tuottoarvoa: SL-hinta ja gapit, entryhinta, kulut ja toteutuminen vaikuttavat
luokkakohtaiseen nettotuottoon. Labeler ei vaihda daemonin mallia tai lähetä
toimeksiantoja. Tuloksen exit-/target-/settlement-auditointisarakkeita ei saa
käyttää ennustavina piirteinä.
