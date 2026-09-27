# Thesis-Briscola-RL

Simulatore di Briscola a due giocatori, base comune per due possibili direzioni di tesi:

1. **Briscola completa + reinforcement learning:** quanto aiuta un agente ricordare le carte già giocate?
2. **Briscola ridotta + equilibrio:** quanto bene CFR approssima un equilibrio di Nash in una versione piccola del gioco?

Documentazione completa del simulatore: [`docs/simulatore.md`](docs/simulatore.md).

## Uso rapido

```bash
pip install -e ".[dev]"          # oppure: pip install pytest
python -m pytest                 # esegue i test
python -m briscola.play          # gioca contro il bot nel terminale
python -m briscola.compare --a greedy --b random --deals 5000
```

Richiede Python ≥ 3.10; il simulatore non ha dipendenze esterne.
