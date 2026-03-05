# Codex Session Architecture Rules

- Kein API-spezifischer Code außerhalb von `ingestion/` und `execution/`.
- Alle Daten laufen durch `data/normalizer.py`.
- Neue Datenquellen: neues File in `data/ingestion/`, muss `BaseFeed` implementieren.
- Neue Broker: neues File in `execution/`, muss `BaseBroker` implementieren.
- Kein Live-Trading ohne vorherigen Paper-Trading Schritt.
- Secrets immer aus `config/secrets.env` via `python-dotenv` laden.
