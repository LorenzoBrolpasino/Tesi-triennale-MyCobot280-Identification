# Identificazione del comportamento dinamico del MyCobot280

Codice della tesi triennale in Ingegneria Industriale — curriculum Meccatronica e
Robotica, Università di Trento.

**Il problema.** Il MyCobot280 accetta come comando solo posizioni di giunto. Internamente
le usa come bersaglio per generare una traiettoria interpolata, che diventa il riferimento
dei controllori di motore. Per simulare il robot con accuratezza bisogna identificare
quella procedura di interpolazione.

**La risposta.** Ogni giunto esegue un **trapezoidale asimmetrico a tempo minimo**:

| grandezza | valore | come dipende dal comando |
|---|---|---|
| velocità di crociera | `vmax = 0.026388 · comando` rad/s | **lineare** nel comando 1–100 |
| accelerazione | `a_acc = 8.70` rad/s² | **indipendente** dal comando |
| decelerazione | `a_dec = 7.52` rad/s² | **indipendente** dal comando |

Il robot accelera il 16% più forte di quanto deceleri: è il risultato che distingue il
profilo asimmetrico da quello simmetrico, e che vince il confronto fra modelli su 122
movimenti su 132.

**Il simulatore** costruito su questi tre numeri riproduce il robot reale entro **3.97
quanti di encoder, cioè 0.32°**, su tutti i 132 movimenti del dataset.

---

## Struttura

```
thesis_project/
├── src/
│   ├── preprocessing.py        # 1. filtraggio, derivazione, stime preliminari
│   ├── trajectory_models.py    # 2. profili teorici a tempo minimo
│   ├── fitting.py              # 3. identificazione ai minimi quadrati
│   ├── validation.py           # 4. confronto fra modelli e residui
│   ├── simulator.py            # 5. il simulatore del robot
│   └── viewer.py               # 6. visualizzazione 3D (MeshCat)
├── scripts/
│   ├── run_identification.py   # rigenera tutti i risultati
│   └── make_video.py           # video della simulazione
├── tests/                      # suite pytest (101 test)
├── notebooks/
│   ├── 01_esplorazione_dati.ipynb          # com'è fatto il dataset
│   ├── 02_validazione_preprocessing.ipynb  # quanto ci si può fidare delle stime
│   └── 03_risultati_finali.ipynb           # i risultati
├── data/
│   ├── raw/dati_tesi/          # i dati grezzi, 11 sessioni
│   └── processed/              # CSV intermedi
├── robot_description/          # URDF e mesh del MyCobot280-M5
└── results/{figures,tables}/   # materiale per l'elaborato
```

Il repository è **autosufficiente**: contiene i dati grezzi e il modello del robot, quindi
un clone basta a rieseguire l'intera analisi e a produrre il video.

Il modello in `robot_description/` è di Elephant Robotics, distribuito con licenza
BSD 2-Clause (`robot_description/LICENSE`). Ne è inclusa la sola parte necessaria: l'URDF
del 280-M5 e le mesh che referenzia.

---

## Il dataset

Acquisito con [`scripts/data_collection_robot_angles.py`](scripts/data_collection_robot_angles.py),
che comanda a ciascun giunto un movimento ampio (−120° → +120° e ritorno) registrando le
posizioni a **25 Hz**. È l'unico script che parla con l'hardware, e per girare ha bisogno
di `pymycobot` e del wrapper di comunicazione `mycobot280_wrapper.py` usato nel corso, che
non fa parte di questo lavoro e non è incluso. È qui perché documenta il protocollo con cui
i dati in `data/raw/` sono stati raccolti; tutto il resto dell'analisi non richiede il
robot.

* **11 sessioni**, una per sottocartella di `dati_tesi/`. Il nome è
  `<ripetizione>_<velocità_comandata>`: `3_75` è la terza ripetizione a velocità 75. La
  cartella `25` è l'unica acquisizione a velocità 25.
* Ogni sessione contiene **12 movimenti** (6 giunti × 2 direzioni), separati da pause in
  cui l'acquisizione è sospesa, e i file `.csv` / `.npy` con la matrice
  `[timestamp, j1, …, j6]` — angoli in **radianti**, tempo in secondi.
* **132 movimenti** in totale.

> **Il Δt non è uniforme, ma non è jitter.** Sull'intera sessione il coefficiente di
> variazione supera il 300%: sono le pause fra un movimento e il successivo, che appaiono
> come buchi di alcuni secondi. Dentro i segmenti il passo è 40.1 ms (24.95 Hz) con CV
> dell'8%, dovuto a una dozzina di campioni persi dal loop di acquisizione.
> `segment_movements` usa proprio quei buchi per isolare i singoli movimenti.

---

## Installazione ed esecuzione

```bash
pip install -r requirements.txt
```

`meshcat` e `urdf-parser-py` servono al solo visualizzatore 3D: i moduli di
identificazione e il simulatore sono numpy/scipy puri e funzionano anche senza.

Tutti i comandi vanno lanciati dalla cartella `thesis_project/`.

```bash
# rigenera tutti i risultati da capo (~3 s)
python3 scripts/run_identification.py

# la suite di test
python3 -m pytest tests/ -q

# il video della simulazione
python3 scripts/make_video.py
python3 scripts/make_video.py --vel 50
```

I notebook vanno letti in ordine. Le figure possono essere **interattive** (zoom, pan): la
prima cella seleziona il backend `widget` di ipympl quando è installato, altrimenti
`inline`.

> **I notebook versionati sono salvati in modalità statica**, e vanno rigenerati così:
>
> ```bash
> MPL_STATIC=1 python3 -m jupyter nbconvert --to notebook --execute --inplace \
>     --ExecutePreprocessor.timeout=900 notebooks/03_risultati_finali.ipynb
> ```
>
> Le figure ipympl sono superfici di disegno vive, legate al kernel: salvate nel file
> producono un output `application/vnd.jupyter.widget-view+json` che VS Code preferisce al
> PNG presente nello stesso bundle, e su cui fallisce appena si apre il notebook senza
> eseguirlo. Lavorando dentro VS Code basta invece eseguire le celle normalmente.

---

## Uso del simulatore

```python
from src.simulator import MyCobot280Simulator

sim = MyCobot280Simulator()
t, Q = sim.send_q_ref(np.deg2rad([120, 0, 0, 0, 0, 0]), vel=100)   # (N,), (N, 6)
t, Q = sim.simulate_sequence([(q1, 100), (q2, 50)])                # comandi concatenati
```

`send_q_ref(q_ref, vel)` ha la stessa firma del comando che si manda al robot vero: è ciò
che rende il simulatore sostituibile all'hardware.

---

## Limiti del lavoro

* **Il campionamento a 25 Hz è il limite dominante.** Sotto ~5 campioni di rampa il
  residuo diventa insensibile ad `amax`, e succede su 46 movimenti su 132. Riacquisire a
  100 Hz lo renderebbe identificabile ovunque: è la raccomandazione operativa più concreta
  di questo lavoro.
* **L'asimmetria fra le rampe resta da spiegare.** Il confronto stabilisce *che* c'è, non
  *perché*. Il notebook 02 mostra che non c'è asimmetria fra i due versi di marcia, il che
  rende la spiegazione gravitazionale la meno probabile.
* **Nessun profilo triangolare nel dataset.** Tutti i movimenti hanno escursione ~4.2 rad,
  ben oltre la soglia `vmax²/amax` (margine minimo 4.6). Il ramo triangolare è validato
  solo su segnali sintetici.
* **La sincronizzazione fra giunti non è identificabile** da questo dataset, che muove un
  giunto per volta.
* **Il modello è cinematico**, quindi non riproduce l'assestamento lento del giunto `j4`,
  né descrive un comando che arrivi a movimento già in corso.

---

## Output prodotti

| file | contenuto |
|---|---|
| `data/processed/preprocessing_params.csv` | stime del Modulo 1, una riga per movimento |
| `data/processed/fit_results.csv` | parametri identificati, residui, flag di convergenza |
| `data/processed/model_comparison.csv` | una riga per (movimento, candidato) |
| `data/processed/validation_results.csv` | residui e metriche per movimento |
| `data/processed/simulator_accuracy.csv` | errore del simulatore per movimento |
| `results/tables/` | tabelle riassuntive per l'elaborato |
| `results/figures/` | grafici per l'elaborato |

---

## Convenzioni

* Angoli in **radianti**, tempi in **secondi**, velocità in rad/s, accelerazioni in rad/s².
* Quantizzazione dell'encoder: 1.4·10⁻³ rad (0.08°). È il pavimento fisico del residuo e il
  riferimento naturale per giudicarlo.
* Type hints ovunque; docstring in italiano.
