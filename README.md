# quantNet Trading Platform

Asset-agnostic Trading-Plattform für Swing Trading und Factor-Momentum Strategien.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Wenn `python scripts/start_app.py` mit `No module named uvicorn` fehlschlaegt, fehlen die Backend-Abhaengigkeiten in der aktiven Python-Umgebung. Dann:

```bash
pip install -r requirements.txt
```

## Trading Platform UI

Die neue UI besteht aus einem React-Frontend unter `frontend/` und einer FastAPI unter `api/`.

### Alles mit einem Befehl starten

```bash
npm install
python scripts/start_app.py
```

Alternativ:

```bash
npm install
make app
```

Das startet:
- Frontend auf `http://localhost:5173`
- Backend auf `http://localhost:8000`

Der Ein-Befehl-Start nutzt bewusst kein `--reload`, damit `uvicorn` nicht an Dateisystem-Watchern scheitert.

### Frontend starten

```bash
npm install
npm run frontend
```

Frontend URL:
- `http://localhost:5173`

### Backend starten

```bash
cd api
pip install fastapi uvicorn pydantic python-dotenv --break-system-packages
uvicorn main:app --reload --port 8000
```

Backend URLs:
- `http://localhost:8000`
- `http://localhost:8000/docs`

### Umgebungsvariablen

Backend:

```bash
LIVE_MODE=false
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGIN=http://localhost:5173
```

Frontend:

```bash
VITE_API_URL=http://localhost:8000
```

`LIVE_MODE=false` ist der Default. In diesem Modus liefert die API realistische Mock-Daten, damit das Frontend ohne laufende Trading-Plattform entwickelt werden kann.

## Backtest starten

```bash
python backtest_runner.py
```

Erwartete Ausgabe:
- Sharpe Ratio
- Max Drawdown
- CAGR
- Anzahl Trades
- `equity_curve.png`

## Architekturprinzipien

1. Lose Kopplung
2. Dependency Injection
3. Fail Loud
4. Type Hints überall
5. Keine Magic Numbers
