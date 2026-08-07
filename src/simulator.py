#!/usr/bin/env python3
"""Modulo 5 — Simulatore del MyCobot280.

Dato un comando di posizione e una velocita' 1-100, genera la traiettoria
interpolata che il firmware eseguirebbe: ogni giunto percorre un trapezoidale
asimmetrico a tempo minimo (``a_acc`` = 8.70, ``a_dec`` = 7.52 rad/s^2), con
parametri globali e non per giunto. Modello cinematico, giunti indipendenti.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

try:  # pragma: no cover - dipende da come e' avviato l'interprete
    from src.trajectory_models import asymmetric_position, asymmetric_times
except ImportError:  # pragma: no cover
    from trajectory_models import asymmetric_position, asymmetric_times

__all__ = ["SimulatorParams", "MyCobot280Simulator", "validate_simulator"]

N_JOINTS = 6


# I tre parametri identificati sperimentalmente, piu' il passo di campionamento.
@dataclass(frozen=True)
class SimulatorParams:
    vmax_per_unit: float = 0.026388  # [rad/s] per unita' di comando
    a_acc: float = 8.70              # [rad/s^2]
    a_dec: float = 7.52              # [rad/s^2]
    dt: float = 0.04                 # [s] passo di campionamento, 25 Hz

    # Converte il comando di velocita' 1-100 nella velocita' di crociera corrispondente.
    def vmax(self, velocita_comandata: float) -> float:
        v = float(np.clip(velocita_comandata, 1, 100))  # il comando sta in 1-100
        return self.vmax_per_unit * v


# Simulatore cinematico del MyCobot280, con la stessa interfaccia del robot vero.
class MyCobot280Simulator:

    # Crea il simulatore nella configurazione iniziale indicata.
    def __init__(
        self,
        q_init: Optional[Sequence[float]] = None,
        params: Optional[SimulatorParams] = None,
    ):
        self.params = params or SimulatorParams()
        self.q = (
            np.zeros(N_JOINTS)
            if q_init is None
            else np.asarray(q_init, dtype=float).ravel().copy()
        )
        if self.q.size != N_JOINTS:
            raise ValueError(f"q_init deve avere {N_JOINTS} elementi, ricevuti {self.q.size}.")
        self.t = 0.0  # istante corrente sull'asse dei tempi assoluto

    # -- il modello di un singolo giunto ------------------------------------

    # Durata del movimento di un singolo giunto verso il proprio bersaglio.
    def _durata_giunto(self, q0: float, qf: float, vmax: float) -> float:
        if q0 == qf:
            return 0.0
        return asymmetric_times(q0, qf, vmax, self.params.a_acc, self.params.a_dec).t_total

    # Durata di un comando, pari a quella del giunto piu' lento.
    def move_duration(self, q_ref, vel: float = 100) -> float:
        q_ref = self._as_q(q_ref)
        vmax = self.params.vmax(vel)
        return max(
            (self._durata_giunto(self.q[j], q_ref[j], vmax) for j in range(N_JOINTS)),
            default=0.0,
        )

    # -- interfaccia di comando ---------------------------------------------

    # Simula un comando di posizione e restituisce l'evoluzione temporale dei sei
    # giunti.
    def send_q_ref(self, q_ref, vel: float = 100, dt: Optional[float] = None):
        q_ref = self._as_q(q_ref)
        passo = float(dt) if dt is not None else self.params.dt
        if passo <= 0:
            raise ValueError(f"dt deve essere positivo, ricevuto {passo}.")

        vmax = self.params.vmax(vel)
        durata = self.move_duration(q_ref, vel)  # il comando dura quanto il giunto piu' lento

        n = max(2, int(np.ceil(durata / passo)) + 1)  # numero di campioni a 25 Hz
        tau = np.arange(n) * passo  # asse dei tempi relativo all'inizio del comando
        Q = np.empty((n, N_JOINTS), dtype=float)

        for j in range(N_JOINTS):
            q0, qf = float(self.q[j]), float(q_ref[j])
            if q0 == qf:
                Q[:, j] = q0
            else:
                Q[:, j] = asymmetric_position(
                    tau, q0, qf, vmax, self.params.a_acc, self.params.a_dec
                )

        # L'ultimo campione e' il bersaglio esatto: tau[-1] >= durata, quindi il
        # profilo e' gia' clippato, ma lo si impone per evitare derive numeriche
        # nella concatenazione di piu' comandi.
        Q[-1] = q_ref  # l'ultimo campione e' il bersaglio esatto
        t = self.t + tau  # riporta i tempi sull'orologio assoluto
        self.q = q_ref.copy()
        self.t = float(t[-1])
        return t, Q

    # Concatena piu' comandi su un unico asse dei tempi, come in una sessione reale.
    def simulate_sequence(self, commands: Sequence[tuple], dt: Optional[float] = None):
        tempi, posizioni = [], []
        for i, comando in enumerate(commands):
            q_ref, vel = comando if len(comando) == 2 else (comando[0], 100)
            t, Q = self.send_q_ref(q_ref, vel, dt=dt)
            if i > 0:  # il primo campione ripete l'ultimo del comando precedente
                t, Q = t[1:], Q[1:]
            tempi.append(t)
            posizioni.append(Q)

        if not tempi:
            return np.zeros(0), np.zeros((0, N_JOINTS))
        return np.concatenate(tempi), np.vstack(posizioni)

    # Riporta il simulatore alla configurazione iniziale e azzera il tempo.
    def reset(self, q_init: Optional[Sequence[float]] = None) -> None:
        self.q = (
            np.zeros(N_JOINTS)
            if q_init is None
            else np.asarray(q_init, dtype=float).ravel().copy()
        )
        self.t = 0.0

    # Restituisce la configurazione corrente dei giunti.
    def read_q(self) -> np.ndarray:
        return self.q.copy()

    # Converte l'ingresso in vettore di giunti, verificandone la dimensione.
    @staticmethod
    def _as_q(q) -> np.ndarray:
        q = np.asarray(q, dtype=float).ravel()
        if q.size != N_JOINTS:
            raise ValueError(f"attesi {N_JOINTS} valori di giunto, ricevuti {q.size}.")
        return q


# ---------------------------------------------------------------------------
# Validazione contro i dati reali
# ---------------------------------------------------------------------------


# Confronta il simulatore coi movimenti reali del dataset e ne misura l'errore,
# allineando il solo istante di partenza.
def validate_simulator(
    dataset_root: str | Path,
    params: Optional[SimulatorParams] = None,
    encoder_quantum: float = 1.4e-3,
) -> "object":
    import pandas as pd
    from scipy.optimize import minimize_scalar

    try:  # pragma: no cover
        from src.preprocessing import load_position_data, segment_movements
    except ImportError:  # pragma: no cover
        from preprocessing import load_position_data, segment_movements

    par = params or SimulatorParams()
    root = Path(dataset_root)
    if not root.is_dir():
        raise NotADirectoryError(f"{root} non e' una cartella.")

    righe = []
    for cartella in sorted(p for p in root.iterdir() if p.is_dir()):
        try:
            vel = float(cartella.name.split("_")[-1])
        except ValueError:
            continue

        df = load_position_data(cartella)
        tempo = df.iloc[:, 0].to_numpy()
        nomi = list(df.columns[1:])
        segmenti = segment_movements(tempo, df[nomi].to_numpy(), joint_names=nomi)

        for seg in segmenti:
            q_reale = seg.position
            t_reale = seg.time
            q0, qf = float(q_reale[0]), float(q_reale[-1])
            if q0 == qf:
                continue

            vmax = par.vmax(vel)
            durata = asymmetric_times(q0, qf, vmax, par.a_acc, par.a_dec).t_total

            # Errore fra simulato e misurato per un dato istante di partenza.
            def _rmse(t0: float) -> float:
                q_sim = asymmetric_position(t_reale, q0, qf, vmax, par.a_acc, par.a_dec, t0=t0)
                m = (t_reale >= t0) & (t_reale <= t0 + durata)  # confronta solo la fase di moto
                if int(np.count_nonzero(m)) < 10:
                    m = np.ones(t_reale.size, dtype=bool)
                return float(np.sqrt(np.mean((q_reale[m] - q_sim[m]) ** 2)))

            # Allineamento del solo t0, entro mezzo secondo dalla stima a soglia.
            t0_guess = _onset(t_reale, q_reale)  # punto di partenza della ricerca su t0
            esito = minimize_scalar(
                _rmse, bounds=(t0_guess - 0.5, t0_guess + 0.5), method="bounded"
            )
            t0, rmse = float(esito.x), float(esito.fun)

            in_moto = (t_reale >= t0) & (t_reale <= t0 + durata)
            righe.append({
                "session": cartella.name,
                "commanded_velocity": vel,
                "joint_name": seg.joint_name,
                "segment_index": seg.index,
                "delta_q": qf - q0,
                "n_samples": int(np.count_nonzero(in_moto)),
                "rmse": rmse,
                "rmse_quanti": rmse / encoder_quantum,
                "durata_simulata": durata,
                "t0_allineato": t0,
                "scarto_t0_campioni": (t0 - t0_guess) / par.dt,
            })

    return pd.DataFrame(righe)


# Stima l'istante di inizio del moto per soglia, senza dipendere dal modulo di fitting.
def _onset(t_data, q_data, threshold_ratio: float = 0.10, min_consecutive: int = 3) -> float:
    t = np.asarray(t_data, dtype=float).ravel()
    q = np.asarray(q_data, dtype=float).ravel()
    if t.size < 3:
        return float(t[0]) if t.size else 0.0

    pseudo_v = np.abs(np.gradient(q))
    picco = float(np.percentile(pseudo_v, 95))
    if picco <= 0:
        return float(t[0])

    sopra = pseudo_v > threshold_ratio * picco
    k = max(1, int(min_consecutive))
    if k > 1 and sopra.size >= k:
        indici = np.flatnonzero(np.lib.stride_tricks.sliding_window_view(sopra, k).all(axis=1))
    else:
        indici = np.flatnonzero(sopra)
    return float(t[indici[0]]) if indici.size else float(t[0])
