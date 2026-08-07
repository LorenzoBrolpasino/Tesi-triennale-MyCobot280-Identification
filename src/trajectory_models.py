#!/usr/bin/env python3
"""Modulo 2 — Profili di traiettoria teorici.

Risolve il problema diretto: date posizione iniziale, finale e vincoli
cinematici, genera la traiettoria a tempo minimo. Implementa il trapezoidale
(che degenera da solo in triangolare), il trapezoidale asimmetrico e il
quintico. E' la funzione obiettivo del fitting del Modulo 3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

__all__ = [
    "ProfileTimes",
    "classify_profile_type",
    "profile_times",
    "trapezoidal_position",
    "trapezoidal_velocity",
    "trapezoidal_acceleration",
    "trapezoidal_position_jacobian",
    "asymmetric_times",
    "asymmetric_position",
    "quintic_position",
    "model_position",
    "param_names",
    "n_params",
    "total_time",
    "ramp_time",
    "generate_trapezoidal_profile",
    "PROFILE_TYPES",
]

#: Profili candidati riconosciuti da :func:`model_position` e dal Modulo 4.
PROFILE_TYPES = ("trapezoidal", "triangular", "asymmetric_trapezoidal", "quintic")


# ---------------------------------------------------------------------------
# Strutture dati
# ---------------------------------------------------------------------------


# Tempi caratteristici di un profilo: durata delle rampe, del plateau e del movimento
# intero.
@dataclass
class ProfileTimes:
    profile_type: str
    t_acc: float
    t_flat: float
    t_dec: float
    t_total: float
    v_peak: float
    delta_q: float
    sign: float
    delta_q_crit: float


# Verifica che i vincoli cinematici ricevuti siano positivi e finiti.
def _check_positive(**valori: float) -> None:
    for nome, v in valori.items():
        if not np.isfinite(v) or v <= 0:
            raise ValueError(f"{nome} deve essere positivo e finito, ricevuto {v}.")


# ---------------------------------------------------------------------------
# Classificazione e tempi caratteristici
# ---------------------------------------------------------------------------


# Decide se il movimento e' trapezoidale o triangolare, confrontando lo spostamento con
# la soglia vmax^2/amax.
def classify_profile_type(delta_q: float, vmax: float, amax: float) -> str:
    _check_positive(vmax=vmax, amax=amax)
    # vmax^2/amax = spazio delle due rampe
    return "trapezoidal" if abs(delta_q) >= vmax**2 / amax else "triangular"


# Calcola i tempi caratteristici del profilo a tempo minimo, degenerando da solo nel
# caso triangolare.
def profile_times(q0: float, qf: float, vmax: float, amax: float) -> ProfileTimes:
    _check_positive(vmax=vmax, amax=amax)

    delta_q = float(qf) - float(q0)
    d = abs(delta_q)
    sign = 1.0 if delta_q >= 0 else -1.0
    d_crit = vmax**2 / amax  # spostamento minimo per raggiungere vmax

    if d == 0.0:
        return ProfileTimes("degenerate", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, sign, d_crit)

    if d >= d_crit:
        tipo, v_peak = "trapezoidal", float(vmax)
        t_acc = vmax / amax
        t_flat = (d - d_crit) / vmax
    else:
        tipo = "triangular"
        v_peak = float(np.sqrt(amax * d))  # nel triangolare vmax non si raggiunge
        t_acc = v_peak / amax
        t_flat = 0.0

    return ProfileTimes(
        profile_type=tipo,
        t_acc=float(t_acc),
        t_flat=float(t_flat),
        t_dec=float(t_acc),
        t_total=float(2 * t_acc + t_flat),
        v_peak=v_peak,
        delta_q=delta_q,
        sign=sign,
        delta_q_crit=float(d_crit),
    )


# Velocita' di crociera che rende il profilo esattamente triangolare, col plateau di
# durata nulla.
def _equivalent_vmax(q0: float, qf: float, amax: float) -> float:
    return float(np.sqrt(amax * abs(float(qf) - float(q0))))  # annulla il plateau


# ---------------------------------------------------------------------------
# Profilo trapezoidale (degenera da solo nel triangolare)
# ---------------------------------------------------------------------------


# Maschere delle cinque fasi: quiete, accelerazione, crociera, decelerazione, quiete.
def _phase_masks(tau: np.ndarray, tm: ProfileTimes) -> tuple[np.ndarray, ...]:
    t_acc, t_flat, t_tot = tm.t_acc, tm.t_flat, tm.t_total
    return (
        tau <= 0,
        (tau > 0) & (tau < t_acc),
        (tau >= t_acc) & (tau < t_acc + t_flat),
        (tau >= t_acc + t_flat) & (tau < t_tot),
        tau >= t_tot,
    )


# Posizione del profilo trapezoidale in forma chiusa; degenera in triangolare se lo
# spostamento e' troppo breve.
def trapezoidal_position(
    t, q0: float, qf: float, vmax: float, amax: float, t0: float = 0.0
) -> np.ndarray:
    tm = profile_times(q0, qf, vmax, amax)
    tau = np.asarray(t, dtype=float) - float(t0)

    if tm.profile_type == "degenerate":
        return np.full(tau.shape, float(q0))

    _, accel, flat, decel, after = _phase_masks(tau, tm)
    d_acc = 0.5 * amax * tm.t_acc**2  # spazio percorso nella sola rampa di salita

    q = np.zeros(tau.shape, dtype=float)
    q[accel] = 0.5 * amax * tau[accel] ** 2
    q[flat] = d_acc + tm.v_peak * (tau[flat] - tm.t_acc)
    u = tau[decel] - tm.t_acc - tm.t_flat  # tempo trascorso dentro la decelerazione
    q[decel] = d_acc + tm.v_peak * tm.t_flat + tm.v_peak * u - 0.5 * amax * u**2
    q[after] = abs(tm.delta_q)  # dopo l'arrivo la posizione resta al bersaglio

    return float(q0) + tm.sign * q


# Velocita' del trapezoidale: rampa lineare, plateau costante, rampa di ritorno a zero.
def trapezoidal_velocity(
    t, q0: float, qf: float, vmax: float, amax: float, t0: float = 0.0
) -> np.ndarray:
    tm = profile_times(q0, qf, vmax, amax)
    tau = np.asarray(t, dtype=float) - float(t0)

    v = np.zeros(tau.shape, dtype=float)
    if tm.profile_type == "degenerate":
        return v

    _, accel, flat, decel, _ = _phase_masks(tau, tm)
    v[accel] = amax * tau[accel]
    v[flat] = tm.v_peak
    v[decel] = tm.v_peak - amax * (tau[decel] - tm.t_acc - tm.t_flat)
    return tm.sign * v


# Accelerazione del trapezoidale: costante a tratti e discontinua ai raccordi.
def trapezoidal_acceleration(
    t, q0: float, qf: float, vmax: float, amax: float, t0: float = 0.0
) -> np.ndarray:
    tm = profile_times(q0, qf, vmax, amax)
    tau = np.asarray(t, dtype=float) - float(t0)

    a = np.zeros(tau.shape, dtype=float)
    if tm.profile_type == "degenerate":
        return a

    _, accel, _, decel, _ = _phase_masks(tau, tm)
    a[accel] = amax
    a[decel] = -amax
    return tm.sign * a


# Derivate della posizione rispetto a vmax, amax e t0, tenendo conto che gli istanti di
# commutazione dipendono dai parametri.
def trapezoidal_position_jacobian(
    t, q0: float, qf: float, vmax: float, amax: float, t0: float = 0.0
) -> np.ndarray:
    tm = profile_times(q0, qf, vmax, amax)
    tau = np.atleast_1d(np.asarray(t, dtype=float)) - float(t0)

    jac = np.zeros((tau.size, 3), dtype=float)
    if tm.profile_type == "degenerate":
        return jac

    d = abs(tm.delta_q)
    _, accel, flat, decel, _ = _phase_masks(tau, tm)
    u = tm.t_total - tau  # tempo che manca alla fine del movimento

    jac[accel, 1] = 0.5 * tau[accel] ** 2  # in salita la posizione dipende solo da amax
    jac[accel, 2] = -amax * tau[accel]

    if tm.profile_type == "trapezoidal":
        jac[flat, 0] = tau[flat] - vmax / amax  # sul plateau conta soprattutto vmax
        jac[flat, 1] = vmax**2 / (2 * amax**2)
        jac[flat, 2] = -vmax

        jac[decel, 0] = amax * u[decel] * d / vmax**2 - u[decel]
        jac[decel, 1] = -0.5 * u[decel] ** 2 + u[decel] * vmax / amax
    else:
        # t_tot = 2*sqrt(d/amax)  =>  d(t_tot)/d(amax) = -t_tot/(2*amax).
        jac[decel, 1] = -0.5 * u[decel] ** 2 + 0.5 * u[decel] * tm.t_total

    jac[decel, 2] = -amax * u[decel]
    return tm.sign * jac


# ---------------------------------------------------------------------------
# Trapezoidale asimmetrico
# ---------------------------------------------------------------------------


# Tempi caratteristici del trapezoidale con accelerazione e decelerazione di modulo
# diverso.
def asymmetric_times(
    q0: float, qf: float, vmax: float, a_acc: float, a_dec: float
) -> ProfileTimes:
    _check_positive(vmax=vmax, a_acc=a_acc, a_dec=a_dec)

    delta_q = float(qf) - float(q0)
    d = abs(delta_q)
    sign = 1.0 if delta_q >= 0 else -1.0
    d_crit = 0.5 * vmax**2 * (1.0 / a_acc + 1.0 / a_dec)

    if d == 0.0:
        return ProfileTimes("degenerate", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, sign, d_crit)

    if d >= d_crit:
        tipo, v_peak = "trapezoidal", float(vmax)
        t_flat = (d - d_crit) / vmax
    else:
        tipo = "triangular"
        v_peak = float(np.sqrt(2.0 * d * a_acc * a_dec / (a_acc + a_dec)))
        t_flat = 0.0

    t_acc, t_dec = v_peak / a_acc, v_peak / a_dec
    return ProfileTimes(
        profile_type=tipo,
        t_acc=float(t_acc),
        t_flat=float(t_flat),
        t_dec=float(t_dec),
        t_total=float(t_acc + t_flat + t_dec),
        v_peak=v_peak,
        delta_q=delta_q,
        sign=sign,
        delta_q_crit=float(d_crit),
    )


# Posizione del trapezoidale a rampe indipendenti: e' il profilo che descrive il
# MyCobot280.
def asymmetric_position(
    t, q0: float, qf: float, vmax: float, a_acc: float, a_dec: float, t0: float = 0.0
) -> np.ndarray:
    tm = asymmetric_times(q0, qf, vmax, a_acc, a_dec)
    tau = np.asarray(t, dtype=float) - float(t0)

    if tm.profile_type == "degenerate":
        return np.full(tau.shape, float(q0))

    v, t_acc, t_flat = tm.v_peak, tm.t_acc, tm.t_flat
    d_acc = 0.5 * v**2 / a_acc

    accel = (tau > 0) & (tau < t_acc)
    flat = (tau >= t_acc) & (tau < t_acc + t_flat)
    decel = (tau >= t_acc + t_flat) & (tau < tm.t_total)

    q = np.zeros(tau.shape, dtype=float)
    q[accel] = 0.5 * a_acc * tau[accel] ** 2
    q[flat] = d_acc + v * (tau[flat] - t_acc)
    u = tau[decel] - t_acc - t_flat
    q[decel] = d_acc + v * t_flat + v * u - 0.5 * a_dec * u**2
    q[tau >= tm.t_total] = abs(tm.delta_q)

    return float(q0) + tm.sign * q


# ---------------------------------------------------------------------------
# Quintico
# ---------------------------------------------------------------------------


# Posizione di un polinomio di grado cinque con estremi a riposo, senza tratto a
# velocita' costante.
def quintic_position(t, q0: float, qf: float, t_move: float, t0: float = 0.0) -> np.ndarray:
    _check_positive(t_move=t_move)
    tau = np.asarray(t, dtype=float) - float(t0)
    s = np.clip(tau / float(t_move), 0.0, 1.0)
    return float(q0) + (float(qf) - float(q0)) * (10 * s**3 - 15 * s**4 + 6 * s**5)


# ---------------------------------------------------------------------------
# Interfaccia comune ai candidati (usata dai Moduli 3 e 4)
# ---------------------------------------------------------------------------

_PARAM_NAMES = {
    "trapezoidal": ("vmax", "amax", "t0"),
    "triangular": ("amax", "t0"),
    "asymmetric_trapezoidal": ("vmax", "a_acc", "a_dec", "t0"),
    "quintic": ("t_move", "t0"),
}


# Nomi dei parametri liberi del profilo, nell'ordine in cui il fit se li aspetta.
def param_names(profile_type: str) -> tuple[str, ...]:
    try:
        return _PARAM_NAMES[profile_type]
    except KeyError:
        raise ValueError(
            f"profilo '{profile_type}' sconosciuto; attesi {sorted(_PARAM_NAMES)}."
        ) from None


# Numero di parametri liberi del profilo, per penalizzare i modelli in AIC/BIC.
def n_params(profile_type: str) -> int:
    return len(param_names(profile_type))


# Ingresso unico ai profili: instrada la chiamata al modello indicato da profile_type.
def model_position(profile_type: str, t, q0: float, qf: float, params: Sequence[float]) -> np.ndarray:
    p = np.asarray(params, dtype=float)
    attesi = n_params(profile_type)
    if p.size != attesi:
        raise ValueError(
            f"il profilo '{profile_type}' richiede {attesi} parametri, ricevuti {p.size}."
        )

    if profile_type == "trapezoidal":
        return trapezoidal_position(t, q0, qf, p[0], p[1], p[2])
    if profile_type == "triangular":
        return trapezoidal_position(t, q0, qf, _equivalent_vmax(q0, qf, p[0]), p[0], p[1])
    if profile_type == "asymmetric_trapezoidal":
        return asymmetric_position(t, q0, qf, p[0], p[1], p[2], p[3])
    return quintic_position(t, q0, qf, p[0], p[1])


# Durata totale del movimento per il profilo indicato.
def total_time(profile_type: str, q0: float, qf: float, params: Sequence[float]) -> float:
    p = np.asarray(params, dtype=float)
    if profile_type == "trapezoidal":
        return profile_times(q0, qf, p[0], p[1]).t_total
    if profile_type == "triangular":
        return profile_times(q0, qf, _equivalent_vmax(q0, qf, p[0]), p[0]).t_total
    if profile_type == "asymmetric_trapezoidal":
        return asymmetric_times(q0, qf, p[0], p[1], p[2]).t_total
    return float(p[0])


# Durata della rampa di accelerazione, da cui si giudica se amax sia identificabile.
def ramp_time(profile_type: str, q0: float, qf: float, params: Sequence[float]) -> float:
    p = np.asarray(params, dtype=float)
    if profile_type == "trapezoidal":
        return float(p[0] / p[1])
    if profile_type == "triangular":
        return float(np.sqrt(abs(float(qf) - float(q0)) / p[0]))
    if profile_type == "asymmetric_trapezoidal":
        return asymmetric_times(q0, qf, p[0], p[1], p[2]).t_acc
    return float("nan")


# ---------------------------------------------------------------------------
# Generazione di segnali sintetici
# ---------------------------------------------------------------------------


# Genera un segnale campionato a parametri noti, con quiete iniziale e finale ed
# eventuale rumore.
def generate_trapezoidal_profile(
    q0: float,
    delta_q: float,
    vmax: float,
    amax: float,
    fs: float,
    t_pre: float = 0.5,
    t_post: float = 0.5,
    noise_std: float = 0.0,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    _check_positive(vmax=vmax, amax=amax, fs=fs)

    qf = float(q0) + float(delta_q)
    tm = profile_times(q0, qf, vmax, amax)

    n = int(round((t_pre + tm.t_total + t_post) * fs)) + 1
    time = np.arange(n) / fs
    position = trapezoidal_position(time, q0, qf, vmax, amax, t0=t_pre)

    if noise_std > 0:
        position = position + np.random.default_rng(seed).normal(
            0.0, noise_std, size=position.shape
        )

    info = {
        "profile_type": tm.profile_type,
        "v_peak": tm.v_peak,
        "vmax_nominal": vmax,
        "amax_nominal": amax,
        "t_acc": tm.t_acc,
        "t_flat": tm.t_flat,
        "t_move": tm.t_total,
        "t_pre": t_pre,
        "delta_q": float(delta_q),
    }
    return time, position, info
