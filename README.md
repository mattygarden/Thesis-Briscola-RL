# Thesis-Briscola-RL

Simulatore di Briscola a due giocatori, base comune per due possibili direzioni di tesi:

1. **Briscola completa + reinforcement learning:** quanto aiuta un agente ricordare le carte già giocate?
2. **Briscola ridotta + equilibrio:** quanto bene CFR approssima un equilibrio di Nash in una versione piccola del gioco?

Documentazione:

- [`docs/simulatore.md`](docs/simulatore.md): il simulatore (regole, stato, osservazioni, agenti di base, valutazione);
- [`docs/ambiente.md`](docs/ambiente.md): l'ambiente di addestramento RL, il learner lineare, il ciclo di training e l'esperimento sulla memoria.

## Uso rapido

```bash
pip install -e ".[dev]"          # oppure: pip install pytest
python -m pytest                 # esegue i test
python -m briscola.play          # gioca contro il bot nel terminale
python -m briscola.compare --a greedy --b random --deals 5000
python -m briscola.train --memory --episodes 50000 --out runs/memory-s0
python -m briscola.experiment --episodes 100000 --seeds 4 --out results/memory-linear
```

Richiede Python ≥ 3.10; il simulatore non ha dipendenze esterne.
