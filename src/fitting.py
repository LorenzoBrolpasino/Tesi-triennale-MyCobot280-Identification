#!/usr/bin/env python3
"""Modulo 3 — Identificazione dei parametri per fitting ai minimi quadrati.

Ottimizza i parametri del profilo teorico perche' riproduca la posizione
**grezza** misurata, non filtrata e non derivata: filtrare introdurrebbe il bias
che il Modulo 1 deve segnalare. Anche ``t0`` e' un parametro libero, perche' a
25 Hz un errore di un campione domina qualunque differenza di forma.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

try:  # pragma: no cover - dipende da come e' avviato l'interprete
    from src.preprocessing import load_position_data, segment_movements
    from src.trajectory_models import (
        PROFILE_TYPES, model_position, n_params, param_names, profile_times,
        ramp_time, trapezoidal_position_jacobian,
    )
except ImportError:  # pragma: no cover
    from preprocessing import load_position_data, segment_movements
    from trajectory_models import (
        PROFILE_TYPES, model_position, n_params, param_names, profile_times,
        ramp_time, trapezoidal_position_jacobian,
    )

__all__ = [
    "FitConfig",
    "FitResult",
    "estimate_t0",
    "build_initial_guess",
    "fit_single_movement",
    "fit_all_movements",
    "load_movements_from_processed",
    "save_fit_results",
]

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_META_KEYS = ("joint_name", "segment_index", "session", "source", "commanded_velocity")


# Impostazioni dell'ottimizzatore: bounds dei parametri, tolleranze e soglia di
# identificabilita'.
@dataclass
class FitConfig:
    vmax_bounds: tuple[float, float] = (1e-3, 20.0)
    amax_bounds: tuple[float, float] = (1e-3, 500.0)
    t_move_min: float = 1e-2
    t0_margin: float = 1.0
    tol: float = 1e-10
    use_analytic_jacobian: bool = True
    branch_margin: float = 0.01
    min_ramp_samples: int = 5


# Esito del fit di un movimento: parametri identificati, residuo e flag di convergenza.
@dataclass
class FitResult:
    profile_type: str
    q0: float
    qf: float
    delta_q: float
    n_samples: int

    vmax_fit: Optional[float] = None
    amax_fit: Optional[float] = None
    t0_fit: Optional[float] = None

    success: bool = False
    status: int = -1
    message: str = ""
    n_function_evals: int = 0
    n_jacobian_evals: int = 0

    rmse: float = np.nan
    max_abs_residual: float = np.nan
    cost: float = np.nan

    amax_identifiable: bool = False
    ramp_samples: float = np.nan
    branch_changed: bool = False

    initial_guess: tuple[float, ...] = ()
    joint_name: str = ""
    segment_index: int = -1
    session: str = ""
    source: str = ""
    commanded_velocity: float = np.nan

    # Parametri identificati con i nomi del profilo. Per il trapezoidale
    # coincidono con i campi propri; per l'asimmetrico e il quintico e' l'unico
    # posto dove finiscono a_acc, a_dec e t_move.
    params: dict[str, float] = field(default_factory=dict)
    residuals: Optional[np.ndarray] = field(default=None, repr=False)

    # Appiattisce l'esito in un dizionario per il CSV, senza il vettore dei residui.
    def to_record(self) -> dict:
        rec = {k: v for k, v in asdict(self).items() if k not in ("residuals", "params")}
        rec["initial_guess"] = " ".join(f"{v:.6g}" for v in self.initial_guess)
        rec.update({f"{nome}_fit": v for nome, v in self.params.items()})
        return rec


# Costruisce un esito di fallimento, cosi' un errore non interrompe il batch.
def _fallito(msg: str, profile_type: str, q0, qf, n: int, x0, **meta) -> FitResult:
    logger.warning("fit non eseguito: %s", msg)
    return FitResult(
        profile_type=profile_type,
        q0=float(q0), qf=float(qf), delta_q=float(qf) - float(q0), n_samples=int(n),
        message=msg,
        initial_guess=tuple(float(v) for v in np.asarray(x0, dtype=float).ravel()),
        **meta,
    )


# ---------------------------------------------------------------------------
# Punto di partenza
# ---------------------------------------------------------------------------


# Stima l'istante in cui il movimento comincia: primo campione in cui la pseudo-
# velocita' supera una frazione del proprio picco.
def estimate_t0(t_data, q_data, threshold_ratio: float = 0.10, min_consecutive: int = 3) -> float:
    t_data = np.asarray(t_data, dtype=float).ravel()
    q_data = np.asarray(q_data, dtype=float).ravel()
    if t_data.size == 0:
        return 0.0
    if t_data.size < 3:
        return float(t_data[0])

    pseudo_v = np.abs(np.gradient(q_data))  # spostamento per campione, non serve dt
    picco = float(np.percentile(pseudo_v, 95))  # picco robusto, immune agli spike
    if picco <= 0:
        return float(t_data[0])

    sopra = pseudo_v > threshold_ratio * picco  # dove il giunto si sta muovendo
    k = max(1, int(min_consecutive))
    if k > 1:
        if sopra.size < k:
            return float(t_data[0])
        indici = np.flatnonzero(np.lib.stride_tricks.sliding_window_view(sopra, k).all(axis=1))
    else:
        indici = np.flatnonzero(sopra)

    return float(t_data[indici[0]]) if indici.size else float(t_data[0])


# Costruisce il vettore dei parametri di partenza, nell'ordine richiesto dal profilo.
def build_initial_guess(
    vmax_est: float, amax_est: float, t0_est: float,
    profile_type: str = "trapezoidal", delta_q: float = np.nan,
) -> np.ndarray:
    nomi = param_names(profile_type)

    if not np.isfinite(amax_est) or amax_est <= 0:
        raise ValueError(f"amax_est deve essere positiva e finita, ricevuto {amax_est}.")
    if not np.isfinite(t0_est):
        raise ValueError(f"t0_est deve essere finito, ricevuto {t0_est}.")
    if "vmax" in nomi and (not np.isfinite(vmax_est) or vmax_est <= 0):
        raise ValueError(f"vmax_est deve essere positiva e finita, ricevuto {vmax_est}.")

    partenza = {
        "vmax": vmax_est, "amax": amax_est, "t0": t0_est,
        "a_acc": amax_est, "a_dec": amax_est,
    }
    if "t_move" in nomi:
        if not np.isfinite(delta_q) or delta_q == 0:
            raise ValueError(
                f"il profilo '{profile_type}' ha bisogno di delta_q per la durata iniziale."
            )
        # Durata del trapezoidale equivalente: stesso spostamento, stessi limiti.
        # durata del trapezoidale equivalente
        partenza["t_move"] = abs(delta_q) / vmax_est + vmax_est / amax_est

    x0 = np.array([float(partenza[p]) for p in nomi], dtype=float)
    if not np.all(np.isfinite(x0)):
        raise ValueError(f"profilo '{profile_type}': punto di partenza non finito {x0}.")
    return x0


# Calcola i limiti di ogni parametro; quelli su t0 e t_move dipendono dalla durata dei
# dati.
def _build_bounds(profile_type: str, t_data, cfg: FitConfig) -> tuple[np.ndarray, np.ndarray]:
    t_data = np.asarray(t_data, dtype=float).ravel()

    if t_data.size:
        t0_low, t0_high = float(t_data[0]) - cfg.t0_margin, float(t_data[-1])
    else:
        t0_low, t0_high = -cfg.t0_margin, cfg.t0_margin
    if t0_high <= t0_low:
        t0_high = t0_low + cfg.t0_margin

    # Un movimento non puo' durare piu' dei dati che lo contengono: senza questo
    # tetto il quintico, che non ha altri vincoli di forma, si allunga a piacere.
    durata = float(t_data[-1] - t_data[0]) if t_data.size > 1 else cfg.t_move_min * 2

    limiti = {
        "vmax": cfg.vmax_bounds,
        "amax": cfg.amax_bounds,
        "a_acc": cfg.amax_bounds,
        "a_dec": cfg.amax_bounds,
        "t_move": (cfg.t_move_min, max(durata, cfg.t_move_min * 2)),
        "t0": (t0_low, t0_high),
    }
    nomi = param_names(profile_type)
    lower = np.array([limiti[p][0] for p in nomi], dtype=float)
    upper = np.array([limiti[p][1] for p in nomi], dtype=float)
    if np.any(upper <= lower):
        raise ValueError(f"bounds incoerenti: lower={lower}, upper={upper}.")
    return lower, upper


# Sposta la vmax iniziale sul ramo trapezoidale se il guess cade nel triangolare, dove
# la derivata rispetto a vmax e' nulla.
def _proietta_sul_ramo(x0, lower, profile_type: str, q0: float, qf: float, cfg: FitConfig):
    nomi = param_names(profile_type)
    if "vmax" not in nomi or "amax" not in nomi:
        return x0

    iv = nomi.index("vmax")
    # confine fra i due rami
    limite = np.sqrt(x0[nomi.index("amax")] * abs(float(qf) - float(q0)))
    limite *= 1.0 - cfg.branch_margin  # un margine dentro il ramo trapezoidale
    if x0[iv] <= limite:
        return x0

    logger.debug("guess nel ramo degenere, vmax riportata sul ramo pieno: %.4g -> %.4g",
                 x0[iv], limite)
    x = x0.copy()
    x[iv] = max(limite, lower[iv])
    return x


# ---------------------------------------------------------------------------
# Funzione di costo e Jacobiano
# ---------------------------------------------------------------------------


# Funzione di costo: differenza fra la posizione del modello e quella misurata.
def _residuals(params, t_data, q_data, q0, qf, profile_type):
    # cio' che l'ottimizzatore minimizza
    return model_position(profile_type, t_data, q0, qf, params) - q_data


# Derivate dei residui rispetto ai parametri, in forma analitica (solo trapezoidale).
def _jacobian(params, t_data, q_data, q0, qf, profile_type):
    return trapezoidal_position_jacobian(t_data, q0, qf, params[0], params[1], params[2])


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------


# Identifica i parametri di un movimento ai minimi quadrati non lineari; un fallimento
# diventa un esito con success=False, senza eccezioni.
def fit_single_movement(
    t_data, q_data, q0: float, qf: float, initial_guess,
    bounds: Optional[tuple] = None,
    profile_type: str = "trapezoidal",
    config: Optional[FitConfig] = None,
    **meta,
) -> FitResult:
    cfg = config or FitConfig()
    t_data = np.asarray(t_data, dtype=float).ravel()
    q_data = np.asarray(q_data, dtype=float).ravel()
    x0 = np.asarray(initial_guess, dtype=float).ravel()
    n = t_data.size

    # Controlli preliminari: falliscono in modo pulito, non con un'eccezione.
    if profile_type not in PROFILE_TYPES:
        return _fallito(f"profilo '{profile_type}' sconosciuto", profile_type, q0, qf, n, x0, **meta)
    k = n_params(profile_type)
    if n != q_data.size:
        return _fallito(f"t_data ({n}) e q_data ({q_data.size}) hanno lunghezze diverse",
                        profile_type, q0, qf, n, x0, **meta)
    if n < k + 1:
        return _fallito(f"campioni insufficienti: {n} per {k} parametri",
                        profile_type, q0, qf, n, x0, **meta)
    if x0.size != k:
        return _fallito(f"initial_guess di lunghezza {x0.size}, attesa {k}",
                        profile_type, q0, qf, n, x0, **meta)
    if not np.all(np.isfinite(t_data)) or not np.all(np.isfinite(q_data)):
        return _fallito("t_data o q_data contengono valori non finiti",
                        profile_type, q0, qf, n, x0, **meta)
    if q0 == qf:
        return _fallito("spostamento nullo (q0 == qf): nessun parametro identificabile",
                        profile_type, q0, qf, n, x0, **meta)

    try:
        if bounds is None:
            lower, upper = _build_bounds(profile_type, t_data, cfg)
        else:
            lower = np.asarray(bounds[0], dtype=float).ravel()
            upper = np.asarray(bounds[1], dtype=float).ravel()
            if lower.size != k or upper.size != k:
                raise ValueError(f"bounds di lunghezza ({lower.size}, {upper.size}), attesa {k}.")
            if np.any(upper <= lower):
                raise ValueError(f"bounds incoerenti: lower={lower}, upper={upper}.")

        # 'trf' richiede che il punto di partenza stia dentro i bounds.
        x_start = _proietta_sul_ramo(np.clip(x0, lower, upper), lower, profile_type, q0, qf, cfg)

        usa_jac = cfg.use_analytic_jacobian and profile_type == "trapezoidal"
        sol = least_squares(
            _residuals,
            x0=x_start,
            jac=_jacobian if usa_jac else "2-point",
            bounds=(lower, upper),
            method="trf",
            ftol=cfg.tol, xtol=cfg.tol, gtol=cfg.tol,
            args=(t_data, q_data, q0, qf, profile_type),
        )
    except Exception as exc:  # noqa: BLE001 - un batch non deve mai interrompersi
        return _fallito(f"{type(exc).__name__}: {exc}", profile_type, q0, qf, n, x0, **meta)

    valori = dict(zip(param_names(profile_type), (float(v) for v in sol.x)))
    vmax_fit = valori.get("vmax", np.nan)
    amax_fit = valori.get("amax", np.nan)

    # Con profile_type fisso il ramo non dovrebbe cambiare: se accade, la stima
    # va guardata con sospetto.
    branch_changed = False
    if profile_type == "trapezoidal" and np.isfinite(vmax_fit):
        branch_changed = profile_times(q0, qf, vmax_fit, amax_fit).profile_type != "trapezoidal"
        if branch_changed:
            logger.warning("profilo degenerato in triangolare durante il fit "
                           "(vmax=%.4g, amax=%.4g)", vmax_fit, amax_fit)

    # Identificabilita': se la rampa non copre abbastanza campioni il residuo
    # diventa quasi insensibile all'accelerazione. Il fit converge lo stesso, ma
    # su un minimo piatto, e il valore restituito e' un artefatto numerico.
    t_rampa = ramp_time(profile_type, q0, qf, sol.x)
    dt_mediano = float(np.median(np.diff(t_data))) if n > 1 else np.nan
    ramp_samples = t_rampa / dt_mediano if np.isfinite(t_rampa) and dt_mediano > 0 else np.nan

    res = np.asarray(sol.fun, dtype=float)
    return FitResult(
        profile_type=profile_type,
        q0=float(q0), qf=float(qf), delta_q=float(qf) - float(q0), n_samples=int(n),
        vmax_fit=float(vmax_fit) if np.isfinite(vmax_fit) else None,
        amax_fit=float(amax_fit),
        t0_fit=float(valori["t0"]),
        success=bool(sol.success),
        status=int(sol.status),
        message=str(sol.message),
        n_function_evals=int(sol.nfev),
        n_jacobian_evals=int(sol.njev),
        rmse=float(np.sqrt(np.mean(res**2))),
        max_abs_residual=float(np.max(np.abs(res))),
        cost=float(sol.cost),
        amax_identifiable=bool(np.isfinite(ramp_samples) and ramp_samples >= cfg.min_ramp_samples),
        ramp_samples=float(ramp_samples),
        branch_changed=branch_changed,
        initial_guess=tuple(float(v) for v in x0),
        params=valori,
        residuals=res,
        **meta,
    )


# Applica il fit a tutti i movimenti del dataset e ne raccoglie gli esiti in una
# tabella.
def fit_all_movements(
    data_dict: dict[str, dict[str, Any]],
    config: Optional[FitConfig] = None,
    output_path: Optional[str | Path] = None,
) -> tuple[list[FitResult], pd.DataFrame]:
    cfg = config or FitConfig()
    results: list[FitResult] = []

    logger.info("avvio del fit su %d movimenti", len(data_dict))

    for chiave, mov in data_dict.items():
        meta = {k: mov[k] for k in _META_KEYS if k in mov}
        try:
            t_data = np.asarray(mov["t_data"], dtype=float).ravel()
            q_data = np.asarray(mov["q_data"], dtype=float).ravel()
            profile_type = str(mov.get("profile_type", "trapezoidal"))
            q0 = float(mov["q0"]) if "q0" in mov else float(q_data[0])
            qf = float(mov["qf"]) if "qf" in mov else float(q_data[-1])

            t0_est = mov.get("t0_est")
            if t0_est is None:
                t0_est = estimate_t0(t_data, q_data)

            x0 = build_initial_guess(
                mov.get("vmax_est", np.nan), mov.get("amax_est", np.nan), t0_est,
                profile_type=profile_type, delta_q=qf - q0,
            )
        except (KeyError, ValueError, TypeError, IndexError) as exc:
            results.append(_fallito(
                f"preparazione fallita - {type(exc).__name__}: {exc}",
                str(mov.get("profile_type", "unknown")),
                float(mov.get("q0", np.nan)), float(mov.get("qf", np.nan)), 0, (), **meta,
            ))
            continue

        res = fit_single_movement(
            t_data=t_data, q_data=q_data, q0=q0, qf=qf, initial_guess=x0,
            bounds=mov.get("bounds"), profile_type=profile_type, config=cfg, **meta,
        )
        results.append(res)
        if not res.success:
            logger.warning("fit NON convergente per '%s': %s", chiave, res.message)

    falliti = sum(1 for r in results if not r.success)
    logger.info("fit completato: %d/%d convergenti, %d falliti",
                len(results) - falliti, len(results), falliti)

    table = pd.DataFrame([r.to_record() for r in results])
    if output_path is not None:
        save_fit_results(table, output_path)
    return results, table


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


# Riunisce le stime del Modulo 1 con la posizione grezza corrispondente, pronte per il
# fit.
def load_movements_from_processed(
    params_csv: str | Path,
    raw_root: Optional[str | Path] = None,
    only_reliable: bool = True,
) -> dict[str, dict[str, Any]]:
    params_csv = Path(params_csv)
    params = pd.read_csv(params_csv)
    if only_reliable and "reliable" in params.columns:
        params = params[params["reliable"].astype(bool)]

    data: dict[str, dict[str, Any]] = {}

    for source, gruppo in params.groupby("source", sort=False):
        percorso = Path(str(source))
        if not percorso.exists() and raw_root is not None:
            candidato = Path(raw_root) / percorso.parent.name / percorso.name
            if candidato.exists():
                percorso = candidato
        if not percorso.exists():
            logger.warning("sorgente non trovata, %d movimenti saltati: %s", len(gruppo), source)
            continue

        df = load_position_data(percorso)
        tempo = df.iloc[:, 0].to_numpy()
        nomi = list(df.columns[1:])
        per_indice = {s.index: s for s in segment_movements(tempo, df[nomi].to_numpy(), joint_names=nomi)}

        for _, riga in gruppo.iterrows():
            seg = per_indice.get(int(riga["segment_index"]))
            if seg is None or seg.joint_name != riga["joint_name"]:
                logger.warning("segmento %s di %s non ricostruibile: movimento saltato",
                               riga["segment_index"], percorso.name)
                continue

            q_grezza = seg.position
            sessione = str(riga.get("session", percorso.parent.name))
            data[f"{sessione}/{seg.joint_name}/seg{seg.index}"] = {
                "t_data": seg.time,
                "q_data": q_grezza,  # GREZZA: nessun filtro, nessuna derivata
                "q0": float(q_grezza[0]),
                "qf": float(q_grezza[0]) + float(riga["delta_q"]),
                "vmax_est": float(riga["vmax_est"]),
                "amax_est": float(riga["amax_est"]),
                "profile_type": str(riga["profile_type"]),
                "joint_name": seg.joint_name,
                "segment_index": int(seg.index),
                "session": sessione,
                "source": str(percorso),
                "commanded_velocity": float(riga.get("commanded_velocity", np.nan)),
            }

    logger.info("caricati %d movimenti da %s", len(data), params_csv.name)
    return data


# Salva la tabella dei fit in CSV, creando le cartelle mancanti.
def save_fit_results(table: pd.DataFrame, path: str | Path, float_format: str = "%.6g") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, float_format=float_format)
    return path
