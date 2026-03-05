# quantNet Trading Platform

Asset-agnostic Trading-Plattform für Swing Trading und Factor-Momentum Strategien.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

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
