# swingbot

Research-first swing-trading system for crypto (now) and U.S. ETFs (later). **Paper trading only.**
The aim is the best result that survives honest out-of-sample testing after all costs — or a clear "no edge found".

Start here: `CLAUDE.md` → `docs/STATUS.md` → `docs/BLUEPRINT.md` → `docs/TESTING_PROTOCOL.md` → `docs/DECISIONS.md`.

```
uv sync && uv run pytest
uv run python scripts/fetch_crypto.py --interval 1d
uv run python scripts/run_experiment.py EXP-001
```

This is research, not financial advice.
