# Simulatore di Briscola a 2 — documentazione

Questo documento descrive cosa fa il simulatore, come è costruito e perché è costruito così. Il simulatore è la base comune alle due direzioni di tesi da presentare al prof. Celli:

| | Direzione 1 — Memoria e RL | Direzione 2 — Equilibrio su gioco ridotto |
|---|---|---|
| Gioco | Briscola completa (40 carte) | Briscola ridotta (poche carte) |
| Domanda | Quanto migliora un agente RL se ricorda le carte già uscite? | Quanto bene CFR approssima un equilibrio di Nash? |
| Cosa usa del simulatore | Osservazioni, encoding con/senza memoria, agenti baseline, valutazione | Configurazione ridotta, mazzo esplicito, chiavi degli information set, `clone()` |

Tutto è scritto in **Python puro (≥ 3.10), senza dipendenze esterne**. Per i test serve solo `pytest`.

---

## 1. Regole implementate

Briscola classica a due giocatori, mazzo da 40 carte con semi italiani (Coppe, Denari, Bastoni, Spade).

**Valori delle carte** (totale 120 punti):

| Carta | Asso | Tre | Re | Cavallo | Fante | 7, 6, 5, 4, 2 |
|---|---|---|---|---|---|---|
| Punti | 11 | 10 | 4 | 3 | 2 | 0 |

**Forza di presa** all'interno di un seme, dalla più debole alla più forte:
2 < 4 < 5 < 6 < 7 < Fante < Cavallo < Re < Tre < Asso.

**Svolgimento:**

1. Il mazzo mescolato viene distribuito alternando le carte: 3 a testa, a partire da chi gioca per primo.
2. La carta successiva viene girata scoperta: è la **briscola** e il suo seme è il seme di briscola. Va sotto il tallone ed è l'ultima carta pescata.
3. Chi è di mano gioca una carta, l'avversario risponde. **Non c'è obbligo di rispondere al seme.**
4. Chi prende:
   - se la seconda carta è dello stesso seme della prima, vince la più forte;
   - se la seconda è di briscola e la prima no, vince la seconda;
   - in tutti gli altri casi vince la prima.
5. Chi vince la presa raccoglie le due carte e **pesca per primo**, poi pesca l'altro. Chi vince la presa gioca anche per primo la presa successiva.
6. Dopo 17 prese il tallone è finito: chi perde la 17ª presa pesca la briscola scoperta. Le ultime 3 prese si giocano senza pescare.
7. Dopo 20 prese vince chi ha più punti (più di 60). 60–60 è pareggio.

**Non implementato** (di proposito, per avere un'unica regola fissa): varianti regionali come lo scambio della briscola scoperta con il 2 o il 7 di briscola, briscola a 3/4 giocatori, briscola chiamata, segnali tra compagni.

---

## 2. Struttura del codice

```
briscola/
  cards.py        carte, punti, forza di presa
  rules.py        GameConfig (mazzo completo o ridotto) e regola della presa
  game.py         GameState: stato completo e transizioni
  observation.py  Observation: ciò che un giocatore può sapere
  encoding.py     vettori di feature e id delle azioni per l'RL
  agents.py       agenti di base: random, lowest, greedy
  arena.py        partite tra agenti e valutazione "duplicate"
  compare.py      confronto tra agenti da terminale
  play.py         giocare contro un bot da terminale
  env.py, learners.py, train.py, experiment.py
                  ambiente di addestramento RL (vedi docs/ambiente.md)
tests/            test (pytest)
```

### 2.1 Carte (`cards.py`)

Una carta è una `NamedTuple` immutabile, `Card(suit, rank)`: per esempio `Card("Denari", 1)` è l'Asso di Denari. Il rango va da 1 a 10, con Fante = 8, Cavallo = 9, Re = 10. Punti e forza sono proprietà (`card.points`, `card.strength`) lette da due tabelle. Ho scelto le `NamedTuple` perché sono leggibili, confrontabili e usabili come chiavi di dizionario, che serve a CFR.

### 2.2 Configurazione e gioco ridotto (`rules.py`)

`GameConfig` stabilisce **quali carte ci sono nel mazzo** (`suits`, `ranks`) e **quante carte ha in mano ciascuno** (`hand_size`).

```python
GameConfig()                      # standard: 4 semi x 10 ranghi, mano da 3
GameConfig.reduced()              # 2 semi x (Asso, Tre, Re), mano da 2 -> 6 carte, 3 prese
GameConfig.reduced(ranks=(1, 3, 10, 2))   # 8 carte, 4 prese
```

Nel gioco ridotto le regole sono **identiche**, compresi briscola scoperta, pescata e "chi vince pesca per primo". Cambia solo il mazzo. Punti e forza restano quelli standard, quindi un mazzo con Asso, Tre e Re conserva la gerarchia 11 / 10 / 4. La configurazione rifiuta i mazzi impossibili: numero di carte dispari, oppure nessuna carta rimasta per la briscola dopo la distribuzione.

`trick_winner(lead, follow, trump_suit)` restituisce 0 se prende la prima carta e 1 se prende la seconda. È la regola del punto 4 della sezione 1.

### 2.3 Stato completo (`game.py`)

`GameState` contiene **tutto**: le mani di entrambi, l'ordine del tallone, le carte sul tavolo, le prese, i punti e la storia. Si usa così:

```python
from briscola import GameState

state = GameState.new_game(seed=42)          # mazzo mescolato con un seed
while not state.is_terminal:
    mosse = state.legal_actions()            # le carte in mano a chi deve giocare
    state.play(mosse[0])
print(state.scores, state.winner(), state.returns())
```

- `new_game(config, seed=…, rng=…, deck=…, first_player=…)` accetta anche **un ordine del mazzo esplicito** (`deck`). Serve a due cose: enumerare tutte le distribuzioni possibili di un gioco ridotto (CFR) e far giocare due agenti con le stesse identiche carte (valutazione duplicate).
- `play(card)` gioca una carta per chi è di turno, risolve la presa e fa pescare. Una mossa illegale solleva `IllegalMoveError`.
- `returns()` restituisce il payoff a somma zero: +1 vittoria, 0 pareggio, −1 sconfitta.
- `clone()` crea una copia indipendente, utile per l'esplorazione dell'albero di gioco (CFR, risolutore del finale).
- `render()` stampa lo stato completo, per il debug.

### 2.4 Osservazione: il vincolo più importante (`observation.py`)

Un agente **non deve mai vedere lo stato completo**, altrimenti imparerebbe barando. Per questo gli agenti ricevono `state.observation(player)`, un oggetto `Observation` immutabile che contiene solo le informazioni legittime:

| Campo | Contenuto |
|---|---|
| `hand` | le proprie carte |
| `trump_card` | la briscola scoperta |
| `table` | la carta giocata dall'avversario, se ha già aperto la presa |
| `scores` | i punti di entrambi (ricostruibili dalle prese, che sono pubbliche) |
| `stock_size` | carte rimaste da pescare |
| `plays`, `tricks` | la sequenza pubblica delle carte giocate |
| `known_opponent_cards` | la briscola scoperta, se l'ha pescata l'avversario e non l'ha ancora giocata |
| `initial_hand`, `my_draws` | la propria mano iniziale e le proprie pescate (informazione privata) |

Due dettagli meritano attenzione:

- **La briscola scoperta è l'unica carta avversaria che si può conoscere.** Tutti vedono chi la pesca all'ultimo giro, quindi da quel momento è informazione pubblica finché non viene giocata.
- **Il finale diventa a informazione perfetta.** `unseen_cards()` restituisce le carte di cui il giocatore non conosce la posizione, cioè quelle divise tra la mano avversaria e il tallone. Quando il tallone è vuoto, queste carte sono esattamente la mano dell'avversario. È l'osservazione della chat: chi conta le carte conosce tutto nelle ultime 3 prese. Un test lo verifica.

Un test verifica anche che l'osservazione non tradisca nulla. Prende due mazzi che differiscono solo per una carta scambiata tra la mano dell'avversario e il tallone, e controlla che il giocatore 0 riceva osservazioni identiche.

### 2.5 Information set e perfect recall (per CFR)

`obs.info_state_key()` restituisce una chiave che identifica l'**information set** del giocatore. Due stati hanno la stessa chiave esattamente quando il giocatore non riesce a distinguerli. La chiave contiene:

- la mano iniziale;
- la briscola;
- la sequenza pubblica delle giocate;
- le proprie pescate in ordine.

Le pescate avvengono sempre subito dopo ogni presa, quindi questi elementi ricostruiscono l'intera sequenza di ciò che il giocatore ha osservato. La chiave rispetta così la **perfect recall**, cioè la condizione sotto cui CFR converge a un equilibrio nei giochi a due giocatori a somma zero. È il punto segnalato come delicato nella chat: se si fondessero storie diverse per risparmiare memoria, la garanzia teorica cadrebbe.

### 2.6 Encoding per l'RL (`encoding.py`)

Le reti neurali vogliono vettori di lunghezza fissa. `encode(obs, memory=...)` produce una lista di float, trasformabile con `numpy.asarray` o `torch.tensor`. Le due modalità corrispondono alle due condizioni dell'esperimento della direzione 1:

| Blocco | Dimensione (40 carte) | `memory=False` (base) | `memory=True` (card counting) |
|---|---|---|---|
| Mano | 40 | ✓ | ✓ |
| Carta sul tavolo | 40 | ✓ | ✓ |
| Briscola scoperta | 40 | ✓ | ✓ |
| Carte avversarie note | 40 | ✓ | ✓ |
| Seme di briscola | 4 | ✓ | ✓ |
| Punti propri, punti avversari, tallone, "sono di mano" | 4 | ✓ | ✓ |
| **Carte già giocate** | 40 | — | ✓ |
| **Totale** | | **168** | **208** |

Come suggerito nella chat, l'agente base **non è privo di memoria**: punti e dimensione del tallone riassumono parte della storia. È "senza memoria esplicita delle carte". Le due codifiche sono identiche tranne l'ultimo blocco, così l'unica differenza tra i due agenti è l'informazione sulle carte uscite.

**Azioni:** un'azione è l'indice della carta in `config.deck` (40 azioni possibili). `action_mask(obs)` indica quali sono legali, cioè al massimo 3. Le funzioni `card_to_action` e `action_to_card` convertono in entrambe le direzioni.

### 2.7 Agenti baseline (`agents.py`)

Un agente è qualunque oggetto con un attributo `name` e un metodo `act(obs, rng) -> Card`.

| Agente | Strategia |
|---|---|
| `RandomAgent` | carta legale a caso; serve come controllo di sanità |
| `LowestCardAgent` | butta sempre la carta meno preziosa (non di briscola, pochi punti, debole); non cerca mai di prendere |
| `GreedyAgent` | se è di mano, butta la carta meno preziosa. Se risponde: prende con una carta dello stesso seme se può (scegliendo quella con più punti, visto che la presa si chiude subito); altrimenti, se sul tavolo c'è un Asso o un Tre, prende con la briscola più debole; altrimenti butta la carta meno preziosa |

Sono euristiche volutamente trasparenti. Servono come avversari di addestramento e di valutazione, e anche come avversari "mai visti in addestramento" per misurare la generalizzazione.

### 2.8 Valutazione duplicate (`arena.py`)

`evaluate(agente_a, agente_b, n_deals, config, seed=...)` confronta due agenti. Per default usa il formato **duplicate**, preso dal bridge: ogni mazzo viene giocato due volte, scambiando i posti. Così entrambi gli agenti ricevono esattamente le stesse carte e lo stesso turno di apertura. Gran parte della fortuna della distribuzione si annulla e servono molte meno partite per vedere una differenza.

Il risultato (`MatchStats`) riporta:

- vittorie, pareggi e sconfitte;
- il reward medio (+1/0/−1) e la differenza media di punti, ciascuno con un intervallo di confidenza al 95%.

L'intervallo è calcolato sulle **coppie** di partite: le due partite sullo stesso mazzo non sono indipendenti, e trattarle come tali darebbe intervalli troppo stretti.

Tutto è **riproducibile**: lo stesso seed dà gli stessi mazzi e le stesse scelte degli agenti casuali.

---

## 3. Come si usa

```bash
pip install -e ".[dev]"             # installa il pacchetto e pytest
python -m pytest                    # tutti i test, pochi secondi
python -m briscola.play             # gioca contro il bot greedy
python -m briscola.play --reduced   # ... sul mazzo ridotto da 6 carte
python -m briscola.compare --a greedy --b random --deals 5000
```

Opzioni di `briscola.play`: `--opponent {random,lowest,greedy}`, `--seed N`, `--second` (inizia il bot), `--reduced`.
Opzioni di `briscola.compare`: `--a`, `--b`, `--deals`, `--seed`, `--no-duplicate`, `--reduced`.

Esempi da codice:

```python
from briscola import GameConfig, GameState, GreedyAgent, RandomAgent, evaluate
from briscola.encoding import action_mask, encode

# Confronto tra agenti
print(evaluate(GreedyAgent(), RandomAgent(), n_deals=1000, seed=0))

# Osservazione ed encoding di un giocatore
state = GameState.new_game(seed=1)
obs = state.observation(0)
x_base = encode(obs, memory=False)      # 168 feature
x_mem = encode(obs, memory=True)        # 208 feature
mask = action_mask(obs)                 # 40 booleani, 3 veri

# Gioco ridotto: enumerare tutti i mazzi possibili (720 per 6 carte)
import itertools
cfg = GameConfig.reduced()
for deck in itertools.permutations(cfg.deck):
    s = GameState(cfg, deck)
    key = s.observation(s.current_player).info_state_key()
    break
```

---

## 4. Cosa verificano i test

| File | Verifica |
|---|---|
| `test_rules.py` | 40 carte e 120 punti; tabelle di punti e forza; 7 casi della regola di presa; gioco ridotto; configurazioni non valide rifiutate |
| `test_game.py` | distribuzione alternata e briscola in fondo al tallone. Su 300 partite casuali: ogni carta giocata una sola volta, punti totali 120, punti = somma delle carte prese, chi vince apre la presa successiva, mani 3-3 fino al tallone vuoto e poi 2-2 e 1-1. Chi vince pesca per primo; chi perde la 17ª presa pesca la briscola. **Tutte le 720 distribuzioni** del gioco ridotto; riproducibilità; mosse illegali; `clone()` indipendente; payoff |
| `test_observation.py` | l'osservazione non rivela mano avversaria né tallone; la briscola pescata dall'avversario è nota; a tallone vuoto le carte non viste coincidono con la mano avversaria; le pescate sono private |
| `test_encoding.py` | dimensioni 168/208; il blocco memoria contiene esattamente le carte delle prese chiuse; maschera e id delle azioni |
| `test_arena.py` | riproducibilità. Greedy contro sé stesso in duplicate dà **esattamente** 0, perché le due partite si specchiano. Greedy batte random con significatività statistica |

---

## 5. Primi numeri

**Agenti baseline** (5.000 mazzi in duplicate = 10.000 partite, seed 0, dal punto di vista del primo agente):

| Confronto | Vittorie | Pareggi | Sconfitte | Reward medio (IC 95%) | Diff. punti media |
|---|---|---|---|---|---|
| greedy vs random | 79,4% | 1,4% | 19,2% | +0,602 ± 0,015 | +31,1 |
| greedy vs lowest | 77,9% | 1,4% | 20,8% | +0,571 ± 0,016 | +34,1 |
| lowest vs random | 41,0% | 1,7% | 57,3% | −0,162 ± 0,019 | −8,6 |
| random vs random | 50,1% | 1,6% | 48,2% | +0,019 ± 0,019 | +0,7 |

Tre osservazioni:

- **La bravura conta**, anche con giocatori molto semplici: un'euristica di poche righe vince quasi l'80% delle partite contro il gioco casuale. È un primo dato contro l'idea che la briscola sia "solo fortuna".
- **Non cercare mai di prendere è peggio del caso.** Una strategia puramente difensiva perde contro il random.
- Random contro random dà un risultato compatibile con zero, come atteso.

**Velocità** (un solo core, Python puro): circa 2.500–3.300 partite complete al secondo sul mazzo da 40 carte, circa 11.000 sul mazzo ridotto da 6. Per l'RL basta: il collo di bottiglia sarà la rete neurale, non il simulatore.

**Dimensione dei giochi ridotti.** Ho esplorato l'albero completo per ogni distribuzione distinta: stesse mani come insiemi, stesso ordine del tallone.

| Gioco ridotto | Carte | Prese | Distribuzioni | Nodi dell'albero | Information set | Tempo |
|---|---|---|---|---|---|---|
| 2 semi × (A, 3, R), mano 2 | 6 | 3 | 180 | 11.340 | 6.000 | 0,1 s |
| 2 semi × (A, 3, R, 2), mano 2 | 8 | 4 | 10.080 | 2.570.400 | 673.008 | 21 s |
| 2 semi × (A, 3, R, F, 2), mano 2 | 10 | 5 | — | — | — | enumerazione semplice > 9 min |

Questo è **il dato chiave per la direzione 2**: la crescita è molto ripida. Con 6 carte CFR gira in pochi secondi, ma è quasi banale. Con 8 carte si arriva a centinaia di migliaia di information set, un gioco già interessante e ancora trattabile in Python con un'implementazione attenta. Oltre servono ottimizzazioni: rappresentazione compatta delle carte, gestione più furba dei nodi di caso, eventualmente Monte Carlo CFR. La scelta della dimensione del gioco ridotto è quindi una decisione di ricerca da discutere con Celli.

---

## 6. Limiti noti e prossimi passi

**Limiti attuali**

- Un'unica regola (quella classica); nessuna variante regionale.
- Gli agenti baseline sono deboli e deterministici, tranne random. Servirà almeno un avversario più forte, per esempio con ricerca nel finale.
- Le prestazioni sono sufficienti ma non ottimizzate (Python puro, carte come oggetti).

**Prossimi passi — direzione 1 (memoria e RL)**

1. ~~Un wrapper stile *Gym* (`reset` / `step`) che gestisce l'avversario e restituisce `encode(obs)`, `action_mask(obs)` e il reward.~~ Fatto: `BriscolaEnv`, vedi [`ambiente.md`](ambiente.md).
2. Un algoritmo RL (per esempio DQN con mascheramento delle azioni, oppure PPO) in PyTorch, addestrato con `memory=False` e con `memory=True` a parità di rete, budget e avversari.
3. Un risolutore esatto per le ultime 3 prese (minimax sullo stato a informazione perfetta, già possibile con `clone()`), per misurare l'accuratezza nel finale.
4. Valutazione con `evaluate` su più seed di addestramento e avversari esclusi dall'addestramento.

**Prossimi passi — direzione 2 (equilibrio)**

1. CFR "vanilla" sul gioco da 6 carte, usando `info_state_key()` come chiave e l'enumerazione delle distribuzioni come nodo di caso iniziale.
2. Calcolo della best response ed exploitability, per misurare la distanza dall'equilibrio.
3. Passaggio al gioco da 8 carte e misura di tempi e memoria al crescere del gioco.
4. Confronto tra la strategia di equilibrio e le euristiche: quanto perdono greedy e le altre contro una best response?
