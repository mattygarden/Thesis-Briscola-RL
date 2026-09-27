# Ambiente di addestramento RL — documentazione

Questo documento descrive l'ambiente di addestramento per agenti di reinforcement learning costruito sopra il simulatore ([`simulatore.md`](simulatore.md)): cosa fa ogni componente, come si usa e quali scelte sono state fatte. La teoria dietro (RL, POMDP, algoritmi, valore dell'informazione, statistica) è spiegata nel documento separato *Agenti RL per la Briscola: teoria e processo*.

Tutto resta in **Python puro, senza dipendenze**.

---

## 1. I componenti

| File | Contenuto | Ruolo |
|---|---|---|
| `briscola/env.py` | `BriscolaEnv` | Una partita vista da un agente che impara: `reset()` / `step(azione)` |
| `briscola/learners.py` | `LinearQLearner` | Agente di riferimento: Q-learning con approssimazione lineare |
| `briscola/train.py` | `train()`, CLI `python -m briscola.train` | Ciclo di addestramento con valutazioni periodiche |
| `briscola/experiment.py` | CLI `python -m briscola.experiment` | Esperimento memoria: agente base contro agente con memoria, più seed, in parallelo |
| `briscola/stats.py` | `welch_t_test()`, `holm_adjust()` | Test statistici per confrontare le condizioni tra seed |

Il flusso è: l'**ambiente** genera partite, il **learner** sceglie le carte e aggiorna i pesi, **train** ripete per molte partite e valuta, **experiment** ripete tutto per ogni condizione e ogni seed.

---

## 2. `BriscolaEnv`

### 2.1 Idea

Un problema di RL ha un solo agente, mentre la Briscola ne ha due. L'ambiente risolve la cosa mettendo **l'avversario dentro l'ambiente**. Dopo ogni carta dell'agente, `step()` gioca da solo le carte dell'avversario e le pescate, finché tocca di nuovo all'agente o la partita finisce. Per l'agente un passo va quindi dalla sua carta alla sua decisione successiva.

L'interfaccia imita **Gymnasium**, lo standard del settore, senza dipenderne:

```python
from briscola import GreedyAgent, LowestCardAgent, RandomAgent
from briscola.env import BriscolaEnv

env = BriscolaEnv([RandomAgent(), LowestCardAgent(), GreedyAgent()],
                  memory=True, reward="trick_points", seed=0)
x, info = env.reset()                      # x: lista di 208 float
done = False
while not done:
    legali = [a for a, ok in enumerate(info["action_mask"]) if ok]
    x, reward, done, truncated, info = env.step(legali[0])
print(info["scores"], info["outcome"])     # es. (64, 56) e +1
```

### 2.2 Parametri

| Parametro | Default | Significato |
|---|---|---|
| `opponents` | — | Un agente o una lista di agenti. Con una lista, a ogni partita se ne estrae uno a caso (miscela fissa di avversari) |
| `config` | `STANDARD` | Mazzo completo o ridotto (`GameConfig.reduced()`) |
| `memory` | `True` | Encoding dell'osservazione: `False` = base (168 feature), `True` = con le carte già giocate (208) |
| `agent_seat` | `None` | `0` o `1` per fissare il posto dell'agente (il posto 0 apre la prima presa); `None` = estratto a caso a ogni partita |
| `reward` | `"win"` | Modalità di reward, vedi 2.4 |
| `seed` | `None` | Seed del generatore interno (mazzi, posti, avversari, scelte casuali degli avversari) |

### 2.3 Metodi e valori restituiti

| Metodo / attributo | Restituisce |
|---|---|
| `reset(seed=None)` | `(x, info)`: prima osservazione dell'agente. Se l'avversario è di mano, ha già giocato la sua carta |
| `step(azione)` | `(x, reward, terminated, truncated, info)`. `truncated` è sempre `False`, perché le partite hanno lunghezza fissa |
| `observation_size` | 168 o 208 (mazzo completo) |
| `action_size` | 40 (una azione per carta del mazzo) |
| `observation()` | L'oggetto `Observation` grezzo dietro il vettore, utile per debug e analisi |
| `action_mask()` | 40 booleani, veri sulle carte in mano |

Il dizionario `info` contiene sempre:

- `action_mask`: la maschera delle azioni legali per la prossima decisione;
- `seat`: il posto dell'agente;
- `opponent`: il nome dell'avversario di questa partita.

A fine partita contiene anche `scores` (punti propri, punti avversari) e `outcome` (+1, 0, −1).

**Azioni.** L'azione è l'indice della carta in `config.deck`. Un'azione illegale, cioè una carta non in mano, solleva `IllegalMoveError`. Anche chiamare `step()` prima di `reset()` o a partita finita solleva un errore. Gli agenti devono scegliere tra le azioni con `action_mask` vera.

### 2.4 Modalità di reward

| `reward` | Quando | Valore |
|---|---|---|
| `"win"` | Solo a fine partita | +1 vittoria, 0 pareggio, −1 sconfitta |
| `"points"` | Solo a fine partita | (punti propri − punti avversari) / 120 |
| `"trick_points"` | A ogni passo in cui si chiude una presa | Variazione della differenza punti / 120 |

In `"trick_points"` i reward di una partita sommano esattamente al reward `"points"`; un test lo verifica. È reward shaping basato su un potenziale. L'addestramento lo usa di default perché è un segnale **denso**: in una prova a parità di tutto, dopo 20.000 partite l'agente lineare batteva il random di +0,44 con `trick_points` e solo di +0,10 con `win`. La **valutazione** usa invece sempre vittorie e sconfitte. La teoria (e il perché massimizzare i punti non equivale a massimizzare le vittorie) è nel documento di teoria.

### 2.5 Uso con librerie esterne

L'ambiente restituisce liste di float e maschere di booleani. Per PyTorch basta `torch.tensor(x)`. Per usare Gymnasium o Stable-Baselines3 (per esempio `MaskablePPO` di sb3-contrib) serve un wrapper di poche righe che dichiari:

- `observation_space = Box(0, 1, (observation_size,))`;
- `action_space = Discrete(action_size)`;
- un metodo `action_masks()` che restituisce `action_mask()`.

---

## 3. `LinearQLearner`

L'agente di riferimento stima un valore Q per ogni carta con un modello lineare:

```
Q(s, a) = b[a] + somma_i w[a][i] * x_i(s)
```

Dopo ogni transizione aggiorna i pesi della carta giocata con il **Q-learning** (target `r + γ · max Q(s', a')` sulle azioni legali, solo `r` a fine partita). Lo scalino è normalizzato per `1 + ||x||²` (regola NLMS): così l'encoding con memoria, che ha più feature attive, non riceve di fatto un learning rate più alto. Il calcolo usa solo le feature non nulle, per velocità.

| Metodo | Uso |
|---|---|
| `select_action(x, mask, rng, epsilon)` | Scelta ε-greedy tra le azioni legali (addestramento) |
| `greedy_action(x, mask, rng)` | La carta con Q più alto; pareggi rotti a caso |
| `update(x, a, r, x_next, mask_next, done)` | Un passo di Q-learning; restituisce il TD error |
| `act(obs, rng)` | Gioco greedy da un'`Observation`: è un `Agent` a tutti gli effetti |
| `save(path)` / `LinearQLearner.load(path)` | Pesi in JSON |

Siccome implementa `act`, un learner addestrato si può valutare con `evaluate()`, far giocare con `briscola.play`, oppure usare come **avversario** di un altro learner (self-play).

Parametri: `lr` (default 0,2, scelto con prove preliminari tra 0,05, 0,2 e 0,5) e `gamma` (default 1, perché le partite sono finite e di lunghezza fissa).

---

## 4. `train`

```python
from briscola import GreedyAgent, LowestCardAgent, RandomAgent
from briscola.env import BriscolaEnv
from briscola.learners import LinearQLearner
from briscola.train import train

opponents = [RandomAgent(), LowestCardAgent(), GreedyAgent()]
env = BriscolaEnv(opponents, memory=True, reward="trick_points")
learner = LinearQLearner(memory=True)
history = train(learner, env, episodes=50_000, eval_every=5_000,
                eval_opponents=opponents, eval_deals=500, seed=0)
```

- **Esplorazione:** ε scende linearmente da `epsilon_start` (0,30) a `epsilon_end` (0,02) nel primo 80% delle partite (`decay_fraction`), poi resta fisso.
- **Valutazione:** ogni `eval_every` partite, e alla fine, la policy greedy (senza esplorazione) gioca contro ogni avversario di test su `eval_deals` mazzi in formato duplicate. I mazzi di valutazione hanno un seed proprio (`EVAL_SEED`), quindi sono diversi da quelli di addestramento.
- **Riproducibilità:** `seed` fissa sia il generatore dell'ambiente sia quello delle scelte esplorative.

Ogni riga di `history` contiene: `episode`, `epsilon`, `opponent`, `mean_reward`, `reward_ci95`, `win_rate`, `mean_point_diff`, `train_outcome_avg` (esito medio delle partite di addestramento dall'ultima valutazione, esplorazione inclusa) ed `elapsed_s`.

Da terminale:

```bash
python -m briscola.train --memory --episodes 50000 --out runs/memory-s0
python -m briscola.train --no-memory --episodes 50000 --out runs/basic-s0
```

| Opzione | Default |
|---|---|
| `--memory` / `--no-memory` | con memoria |
| `--episodes` | 50.000 |
| `--opponents` | `random,lowest,greedy` |
| `--eval-opponents` | `random,lowest,greedy` |
| `--eval-every`, `--eval-deals` | 5.000, 500 |
| `--reward` | `trick_points` |
| `--lr`, `--epsilon-start`, `--epsilon-end` | 0,2, 0,30, 0,02 |
| `--seed` | 0 |
| `--reduced` | mazzo completo |
| `--out` | nessun salvataggio; se indicato salva `weights.json` e `history.csv` |

Un agente salvato si può sfidare così:

```python
from briscola import GreedyAgent, evaluate
from briscola.learners import LinearQLearner

agent = LinearQLearner.load("runs/memory-s0/weights.json")
print(evaluate(agent, GreedyAgent(), 2000))
```

---

## 5. `experiment`

```bash
python -m briscola.experiment --episodes 100000 --seeds 4 --out results/memory-linear
```

Per ogni seed addestra due learner identici, uno per condizione (`basic` e `memory`), cambiando solo l'encoding. I run girano in processi paralleli (`--workers`, di default tutti i core). Alla fine ogni agente è valutato su `--final-deals` mazzi nuovi (default 2.000, cioè 4.000 partite per avversario), con un seed di valutazione diverso da quello delle curve.

File prodotti nella cartella `--out`:

| File | Contenuto |
|---|---|
| `config.csv` | I parametri di ogni run, per la riproducibilità |
| `curves.csv` | Curve di apprendimento: una riga per run, momento di valutazione e avversario |
| `final.csv` | Valutazione finale: una riga per run e avversario, con intervallo di confidenza |
| `summary.csv` | Media e deviazione standard tra seed del reward finale, per condizione e avversario |
| `comparison.csv` | Memoria meno base per avversario: differenza, test t di Welch tra seed, p-value corretto con Holm; più una riga con la media sui tre avversari |
| `<condizione>-seed<k>.json` | I pesi di ogni agente (non versionati in git) |

Le altre opzioni sono le stesse di `train` (`--opponents`, `--eval-opponents`, `--eval-every`, `--eval-deals`, `--reward`, `--lr`, `--reduced`).

Per rianalizzare risultati già salvati, senza addestrare: `python -m briscola.experiment --summarize results/memory-linear`. Le funzioni statistiche (test t di Welch con p-value esatto, correzione di Holm) sono in `briscola/stats.py`, in Python puro.

---

## 6. Test

| File | Verifica |
|---|---|
| `tests/test_env.py` | Una partita completa: 20 decisioni dell'agente, maschera coerente con la mano, reward solo alla fine in modalità `win`. In `trick_points` i reward sommano alla differenza punti finale. Posti e avversari vengono estratti davvero; con l'avversario di mano c'è già una carta sul tavolo. Seed riproducibili, errori su azioni illegali e usi scorretti, mazzo ridotto |
| `tests/test_learners.py` | L'aggiornamento sposta Q verso il target e solo per la carta giocata; le scelte sono sempre legali e seguono Q; salvataggio e caricamento; calendario di ε; 3.000 partite di addestramento bastano a battere il random con significatività |

`tests/test_stats.py` verifica il test t di Welch contro le tavole della t di Student, la correzione di Holm e il confronto tra condizioni.

In tutto il progetto ha 57 test (`python -m pytest`), che girano in circa 5 secondi.

---

## 7. Prestazioni

Su un core, in Python puro:

- **addestramento:** circa 650 partite al secondo (100.000 partite in circa 2,5 minuti);
- **valutazione** di un learner: diverse centinaia di partite al secondo.

L'esperimento completo (2 condizioni × 4 seed × 100.000 partite, con le valutazioni) richiede meno di 6 minuti su 4 core: ogni run dura circa 2–2,5 minuti.

---

## 8. Primi risultati

Esperimento `results/memory-linear` (2 condizioni × 4 seed × 100.000 partite, `trick_points`, lr 0,2). Valutazione finale su 2.000 mazzi duplicate per avversario; confronto tra condizioni con test t di Welch sui 4 run.

| Avversario | Base | Memoria | Differenza | p (Welch) | p (Holm) |
|---|---|---|---|---|---|
| random | +0,543 | +0,531 | −0,012 | 0,35 | 0,35 |
| lowest | +0,524 | +0,477 | −0,047 | 0,011 | 0,034 |
| greedy | +0,048 | +0,113 | +0,066 | 0,11 | 0,22 |
| Media dei tre | +0,371 | +0,374 | +0,002 | 0,84 | — |

Con il modello lineare la memoria **non dà un vantaggio netto**. Cambia però il profilo dell'agente: impara più lentamente all'inizio, gioca meglio contro greedy (non significativo con 4 seed, ma con varianza tra seed molto più bassa) e peggio contro lowest (significativo). L'interpretazione è nel documento di teoria.

---

## 9. Limiti e prossimi passi

- Il learner lineare **non può combinare feature**: per esempio non può contare le briscole uscite, perché quali carte sono briscole dipende dalla partita. È il motivo principale per passare a una rete neurale.
- Gli avversari di test coincidono con quelli di addestramento. Per misurare la generalizzazione servono avversari tenuti fuori dall'addestramento.
- Manca la metrica di **accuratezza nel finale**, che richiede un risolutore esatto delle ultime 3 prese.

Passi successivi, in ordine:

1. **DQN in PyTorch** (con maschera, replay buffer, target network), sullo stesso ambiente. Basta una classe con gli stessi metodi del learner lineare (`select_action`, `update`, `act`): `train()` la usa così com'è, mentre `experiment` va esteso con un'opzione per scegliere il learner.
2. **Risolutore del finale** e metrica di accuratezza nelle ultime 3 prese.
3. **Avversari di test aggiuntivi** e self-play (un learner congelato come avversario).
4. Più seed (8–10 per condizione) e test statistici tra condizioni.
