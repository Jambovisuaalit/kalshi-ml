# Ostajapaketin julkaiseminen

Kaupallinen lähdekoodi toimitetaan oletuksena yksityisestä repositoriosta.
Julkinen repositorio tekee myös proprietary-lisensoidun lähdekoodin kaikkien
ladattavaksi; lisenssi ei estä teknistä lataamista.

GitHub Actionsin **Buyer package verification** suorittaa offline-testit
Python 3.11:llä ja 3.12:lla, rakentaa wheel- ja ZIP-paketit sekä tarkistaa
Docker-kuvan käynnistyskomennot ilman verkkoa. Työnkulku ei käynnistä
kaupankäyntiä, siirrä vakuuksia tai lähetä hälytyksiä.

Julkaisukandidaatti hyväksytään vasta, kun kaikki kolme työtä (`tests`,
`package`, `docker`) onnistuvat. Paketit ja SHA-256-tarkisteet saa saman
ajon `kalshi-ml-0.2.0`-artefaktista. Työnkulun voi käynnistää Actions-välilehden
Run workflow -painikkeesta; se on oltava repositorion oletushaarassa.
Ohje: https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow

Luo v0.2.0-julkaisu juuri tarkistetusta commitista ja liitä sen ZIP, wheel
ja tarkistetiedostot. Älä liitä työaseman .env-tiedostoa, avaimia,
tietokantoja tai lokeja. Kirjaa julkaisukuvaukseen CI-ajon osoite ja
mallikortin tutkimusrajoitukset. Ostajalle toimitetaan myös lasku tai
ostosopimus, jossa myyjä ja lisenssinsaaja yksilöidään.

GitHub-julkaisu on ohjelmistopaketti. Varsinainen daemon otetaan käyttöön
ostajan Linux-palvelimella README:n Docker- tai systemd-ohjeella.
Oletus on paper-tila. Mukana oleva SPOT_PROXY-malli on tutkimusesimerkki,
eikä onnistunut CI tai paketin julkaisu osoita sen kaupankäyntikelpoisuutta.
