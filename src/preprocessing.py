#!/usr/bin/env python3
"""Modulo 1 — Preprocessing e stima preliminare dei parametri cinematici.

Dalle sole misure di posizione ricava una prima stima di ``vmax`` e ``amax`` per
ciascun movimento, che il Modulo 3 usera' come punto di partenza del fitting.
La catena e': caricamento, segmentazione, filtro passa-basso a fase nulla,
derivazione, rilevamento del regime, stime e classificazione del profilo.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.ndimage import find_objects, label
from scipy.signal import butter, filtfilt

try:  # pragma: no cover - dipende da come e' avviato l'interprete
    from src.trajectory_models import classify_profile_type as _classify
except ImportError:  # pragma: no cover
    from trajectory_models import classify_profile_type as _classify

__all__ = [
    "PreprocessingConfig",
    "MovementSegment",
    "JointMovementResult",
    "check_sampling_uniformity",
    "load_position_data",
    "segment_movements",
    "filter_position",
    "compute_derivatives",
    "detect_steady_state",
    "estimate_vmax",
    "estimate_amax",
    "classify_profile_type",
    "analyze_joint_movement",
    "analyze_session",
    "analyze_dataset",
    "save_results",
]

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Soglie di robustezza numerica della pipeline. Non sono grandezze da tarare
# caso per caso: restano qui perche' PreprocessingConfig raccolga solo i
# parametri che ha davvero senso variare.
JITTER_CV_THRESHOLD = 0.01      # CV del passo oltre il quale si avvisa
MIN_DURATION_SAMPLES_FLOOR = 5  # campioni minimi assoluti per una finestra di regime
FILTER_FLOOR_PERIODS = 1.5      # durata minima del regime, in periodi del cutoff
MIN_VELOCITY_RATIO = 0.50       # frazione del picco di |v| sotto cui non e' regime
MAD_K = 3.0                     # soglia robusta mediana + k*MAD per gli outlier
TRANSIENT_ACTIVE_RATIO = 0.50   # frazione del picco di |a| che identifica una rampa
AMAX_MOVING_RATIO = 0.10        # frazione del picco di pseudo-velocita' = "in moto"
AMAX_RAMP_PERIODS = 1.0         # periodi del cutoff sotto cui amax e' sottostimata


# Parametri regolabili della pipeline: frequenza di taglio, ordine del filtro e soglie.
@dataclass
class PreprocessingConfig:
    cutoff_hz: float = 3.0
    filter_order: int = 4
    accel_threshold_ratio: float = 0.25
    min_duration_ratio: float = 0.10
    max_gap_samples: int = 3
    plateau_cv_max: float = 0.25
    min_travel_rad: float = 0.05


# Un singolo movimento isolato dalla sessione, col giunto che si muove e la sua
# traiettoria.
@dataclass
class MovementSegment:
    index: int
    start: int
    stop: int
    time: np.ndarray = field(repr=False)
    positions: np.ndarray = field(repr=False)
    joint_index: int
    joint_name: str

    # Posizione del solo giunto in movimento.
    @property
    def position(self) -> np.ndarray:
        return self.positions[:, self.joint_index]

    # Durata del segmento in secondi.
    @property
    def duration(self) -> float:
        return float(self.time[-1] - self.time[0]) if self.time.size else 0.0

    # Spostamento compiuto dal giunto in movimento, con segno.
    @property
    def travel(self) -> float:
        q = self.position
        return float(q[-1] - q[0]) if q.size else 0.0

    # Indici (inizio, fine) del tratto in cui il giunto si muove davvero: il segmento
    # comprende anche la quiete di baseline prima del comando e l'assestamento dopo.
    @property
    def motion_bounds(self) -> tuple[int, int]:
        q = self.position
        n = int(q.size)
        if n < 2:
            return 0, n
        dq = np.abs(np.gradient(q))
        picco = _robust_peak(dq)  # 95esimo percentile: immune agli spike della quantizzazione
        if picco <= 0:
            return 0, n
        idx = np.flatnonzero(dq > AMAX_MOVING_RATIO * picco)
        return (int(idx[0]), int(idx[-1]) + 1) if idx.size else (0, n)


# Stime del Modulo 1 per un movimento: vmax, amax, tipo di profilo e segnali elaborati.
@dataclass
class JointMovementResult:
    vmax_est: Optional[float]
    amax_est: Optional[float]
    profile_type: str
    steady_state_window: Optional[tuple[int, int]]
    warnings: list[str] = field(default_factory=list)

    delta_q: float = np.nan
    duration: float = np.nan
    fs: float = np.nan
    cutoff_hz: float = np.nan
    joint_name: str = ""
    segment_index: int = -1
    source: str = ""
    steady_state_duration: float = np.nan
    reliable: bool = True
    amax_reliable: bool = True

    # Segnali elaborati: non finiscono nel CSV, servono ai grafici.
    time: Optional[np.ndarray] = field(default=None, repr=False)
    position_raw: Optional[np.ndarray] = field(default=None, repr=False)
    position_filtered: Optional[np.ndarray] = field(default=None, repr=False)
    velocity: Optional[np.ndarray] = field(default=None, repr=False)
    acceleration: Optional[np.ndarray] = field(default=None, repr=False)

    # Appiattisce le stime in un dizionario per il CSV, senza i segnali elaborati.
    def to_record(self) -> dict:
        skip = {
            "time", "position_raw", "position_filtered", "velocity",
            "acceleration", "warnings", "steady_state_window",
        }
        rec = {k: v for k, v in asdict(self).items() if k not in skip}
        ss = self.steady_state_window
        rec["steady_start_idx"] = ss[0] if ss is not None else np.nan
        rec["steady_stop_idx"] = ss[1] if ss is not None else np.nan
        rec["n_warnings"] = len(self.warnings)
        rec["warnings"] = " | ".join(self.warnings)
        return rec


# ---------------------------------------------------------------------------
# 1. Caricamento
# ---------------------------------------------------------------------------


# Misura quanto e' regolare il campionamento e individua i buchi fra un movimento e il
# successivo.
def check_sampling_uniformity(
    time, jitter_cv_threshold: float = JITTER_CV_THRESHOLD, gap_factor: float = 5.0
) -> dict:
    time = np.asarray(time, dtype=float).ravel()
    if time.size < 2:
        return {
            "n_samples": int(time.size), "dt_mean": np.nan, "dt_median": np.nan,
            "dt_std": np.nan, "cv": np.nan, "fs_est": np.nan, "n_gaps": 0,
            "gap_indices": np.empty(0, dtype=int), "is_uniform": False,
        }

    dt = np.diff(time)
    if np.any(dt <= 0):
        raise ValueError(
            "L'asse dei tempi non e' strettamente crescente: "
            f"{int(np.sum(dt <= 0))} intervalli non positivi."
        )

    dt_median = float(np.median(dt))
    dt_mean, dt_std = float(np.mean(dt)), float(np.std(dt))
    cv = dt_std / dt_mean if dt_mean > 0 else np.inf

    return {
        "n_samples": int(time.size),
        "dt_mean": dt_mean,
        "dt_median": dt_median,
        "dt_std": dt_std,
        "cv": float(cv),
        "fs_est": 1.0 / dt_median if dt_median > 0 else np.nan,
        "n_gaps": int(np.count_nonzero(dt > gap_factor * dt_median)),
        "gap_indices": np.flatnonzero(dt > gap_factor * dt_median),
        "is_uniform": bool(cv <= jitter_cv_threshold),
    }


# Carica una sessione grezza (.npy o .csv) in una tabella con tempo e sei giunti.
def load_position_data(path: str | Path, time_column: str = "timestamp") -> pd.DataFrame:
    path = Path(path)
    if path.is_dir():
        candidati = sorted(path.glob("*.npy")) or sorted(path.glob("*.csv"))
        if not candidati:
            raise FileNotFoundError(f"Nessun file .npy o .csv trovato in {path}")
        path = candidati[0]
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")

    if path.suffix.lower() == ".npy":
        data = np.load(path)
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError(f"{path}: attesa una matrice (N, 1+n_giunti), trovata {data.shape}.")
        colonne = [time_column] + [f"j{i + 1}" for i in range(data.shape[1] - 1)]
        df = pd.DataFrame(data, columns=colonne)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        df.columns = [str(c).strip() for c in df.columns]
        if time_column not in df.columns:
            raise ValueError(f"{path}: colonna '{time_column}' non trovata fra {list(df.columns)}.")
    else:
        raise ValueError(f"Estensione non supportata: '{path.suffix}' (attese .npy o .csv).")

    df = df.sort_values(time_column, kind="stable").reset_index(drop=True)
    df.attrs["source"] = str(path)
    return df


# ---------------------------------------------------------------------------
# 2. Segmentazione
# ---------------------------------------------------------------------------


# Divide una sessione nei singoli movimenti, usando le pause dell'acquisizione come
# separatori.
def segment_movements(
    time,
    positions,
    joint_names: Optional[list[str]] = None,
    gap_factor: float = 5.0,
    min_samples: int = 20,
    min_travel_rad: float = 0.05,
) -> list[MovementSegment]:
    time = np.asarray(time, dtype=float).ravel()
    positions = np.asarray(positions, dtype=float)
    if positions.ndim == 1:
        positions = positions[:, None]
    if positions.shape[0] != time.size:
        raise ValueError(
            f"time ({time.size}) e positions ({positions.shape[0]}) hanno lunghezze diverse."
        )
    if joint_names is None:
        joint_names = [f"j{i + 1}" for i in range(positions.shape[1])]
    if time.size < 2:
        return []

    dt = np.diff(time)
    # dove il passo salta: sono le pause fra un movimento e l'altro
    tagli = np.flatnonzero(dt > gap_factor * float(np.median(dt))) + 1
    bordi = np.concatenate(([0], tagli, [time.size]))  # estremi dei segmenti

    segmenti: list[MovementSegment] = []
    for start, stop in zip(bordi[:-1], bordi[1:]):
        start, stop = int(start), int(stop)
        if stop - start < min_samples:
            continue
        blocco = positions[start:stop]
        escursione = blocco.max(axis=0) - blocco.min(axis=0)  # quanto si e' mosso ogni giunto
        j = int(np.argmax(escursione))  # il giunto in movimento e' quello che si e' mosso di piu'
        if escursione[j] < min_travel_rad:
            continue
        segmenti.append(
            MovementSegment(
                index=len(segmenti),
                start=start,
                stop=stop,
                time=time[start:stop] - time[start],
                positions=blocco,
                joint_index=j,
                joint_name=str(joint_names[j]),
            )
        )
    return segmenti


# ---------------------------------------------------------------------------
# 3. Filtraggio e derivazione
# ---------------------------------------------------------------------------


# Filtra la posizione con un Butterworth passa-basso a fase nulla, che attenua senza
# sfasare.
def filter_position(position, fs: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    position = np.asarray(position, dtype=float).ravel()
    nyq = 0.5 * fs
    if not 0 < cutoff_hz < nyq:
        raise ValueError(f"cutoff_hz deve stare in (0, {nyq:g}) Hz, ricevuto {cutoff_hz}.")
    if position.size == 0:
        return position.copy()

    # cutoff normalizzato a Nyquist
    b, a = butter(order, cutoff_hz / nyq, btype="low", analog=False)

    padlen_default = 3 * max(len(a), len(b))  # padding che filtfilt usa di default
    if position.size <= padlen_default:
        padlen = max(0, position.size - 1)
        if padlen == 0:
            logger.warning("segnale di %d campioni: filtraggio saltato.", position.size)
            return position.copy()
        logger.warning(
            "segnale corto (%d campioni): padlen ridotto a %d, bordi meno affidabili.",
            position.size, padlen,
        )
        return filtfilt(b, a, position, padlen=padlen)

    return filtfilt(b, a, position)


# Deriva la posizione filtrata due volte, ottenendo velocita' e accelerazione.
def compute_derivatives(position_filtered, dt: float) -> tuple[np.ndarray, np.ndarray]:
    # differenze centrate
    velocity = np.gradient(np.asarray(position_filtered, dtype=float).ravel(), dt)
    return velocity, np.gradient(velocity, dt)


# ---------------------------------------------------------------------------
# 4. Fase a regime
# ---------------------------------------------------------------------------


# Picco robusto di un segnale: il 95esimo percentile invece del massimo, immune agli
# spike.
def _robust_peak(x: np.ndarray, percentile: float = 95.0) -> float:
    return float(np.percentile(np.abs(x), percentile)) if x.size else 0.0


# Richiude i buchi brevi in una maschera booleana, perche' un plateau non risulti
# spezzato.
def _bridge_short_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    if max_gap < 1 or mask.size == 0 or not mask.any():
        return mask
    out = mask.copy()
    idx = np.flatnonzero(mask)
    buchi = idx[1:] - idx[:-1] - 1  # quanti falsi separano due veri consecutivi
    for start, gap in zip(idx[:-1][buchi > 0], buchi[buchi > 0]):
        if gap <= max_gap:
            out[start + 1 : start + 1 + gap] = True
    return out


# Trova la sequenza contigua di valori veri piu' lunga e ne restituisce inizio, fine e
# lunghezza.
def _longest_true_run(mask: np.ndarray) -> tuple[int, int, int]:
    etichette, n = label(mask)  # numera i gruppi di veri contigui
    if n == 0:
        return -1, -1, 0
    fette = [f[0] for f in find_objects(etichette)]
    migliore = max(fette, key=lambda f: f.stop - f.start)  # la fetta piu' lunga
    return int(migliore.start), int(migliore.stop), int(migliore.stop - migliore.start)


# Individua la fase a velocita' costante con tre condizioni: accelerazione bassa,
# velocita' significativa, plateau stabile.
def detect_steady_state(
    velocity,
    acceleration,
    accel_threshold_ratio: float = 0.25,
    min_duration_ratio: float = 0.10,
    min_duration_samples_floor: int = MIN_DURATION_SAMPLES_FLOOR,
    max_gap_samples: int = 3,
    plateau_cv_max: float = 0.25,
) -> Optional[tuple[int, int]]:
    velocity = np.asarray(velocity, dtype=float).ravel()
    acceleration = np.asarray(acceleration, dtype=float).ravel()
    if velocity.size != acceleration.size:
        raise ValueError("velocity e acceleration devono avere la stessa lunghezza.")
    if velocity.size == 0:
        return None

    abs_v, abs_a = np.abs(velocity), np.abs(acceleration)
    v_peak, a_peak = _robust_peak(abs_v), _robust_peak(abs_a)
    if v_peak <= 0 or a_peak <= 0:
        return None

    accel_threshold = accel_threshold_ratio * a_peak
    min_velocity = MIN_VELOCITY_RATIO * v_peak

    n_moving = int(np.count_nonzero(abs_v > 0.5 * min_velocity))
    min_duration = max(
        2, max(min_duration_samples_floor, int(round(min_duration_ratio * n_moving)))
    )

    # Condizioni 1 e 2, campione per campione, poi il run consecutivo piu' lungo.
    mask = (abs_a < accel_threshold) & (abs_v >= min_velocity)  # le due condizioni insieme
    start, stop, length = _longest_true_run(_bridge_short_gaps(mask, int(max_gap_samples)))
    if length < min_duration:
        return None

    # Condizione 3: plateau vero.
    v_win = velocity[start:stop]
    if not (np.all(v_win > 0) or np.all(v_win < 0)):
        return None
    media = float(np.mean(v_win))
    cv = float(np.std(v_win) / abs(media)) if media != 0 else np.inf  # dispersione relativa
    if cv > plateau_cv_max:
        return None

    return start, stop


# ---------------------------------------------------------------------------
# 5-7. Stime e classificazione
# ---------------------------------------------------------------------------


# Stima la velocita' di crociera come media della velocita' sulla fase a regime.
def estimate_vmax(velocity, steady_state_indices) -> tuple[Optional[float], bool]:
    if steady_state_indices is None:
        return None, True
    start, stop = steady_state_indices
    finestra = np.asarray(velocity, dtype=float).ravel()[start:stop]
    if finestra.size == 0:
        return None, True
    return abs(float(np.mean(finestra))), False


# Stima l'accelerazione come mediana dei campioni sulle rampe, scartando gli spike col
# criterio della MAD.
def estimate_amax(
    acceleration,
    steady_state_indices,
    position=None,
    mad_k: float = MAD_K,
) -> Optional[float]:
    acceleration = np.asarray(acceleration, dtype=float).ravel()
    n = acceleration.size
    if n == 0:
        return None

    mask = np.ones(n, dtype=bool)
    if steady_state_indices is not None:
        start, stop = steady_state_indices
        mask[max(0, start) : min(n, stop)] = False  # esclude il plateau: restano le due rampe

    # Pseudo-velocita' (differenza di posizione per campione) confrontata con un
    # percentile di se stessa: il criterio e' indipendente da dt e dall'unita'.
    if position is not None:
        position = np.asarray(position, dtype=float).ravel()
        if position.size == n and n > 2:
            pseudo_v = np.abs(np.gradient(position))
            pv_peak = _robust_peak(pseudo_v)
            if pv_peak > 0:
                # campioni in cui si muove davvero
                moving = pseudo_v > AMAX_MOVING_RATIO * pv_peak
                if np.count_nonzero(mask & moving) >= 3:
                    mask &= moving

    a_transient = np.abs(acceleration[mask])
    if a_transient.size == 0 or not np.any(a_transient > 0):
        return None

    peak = _robust_peak(a_transient)
    if peak <= 0:
        return None

    # solo le rampe, non le code a zero
    attivi = a_transient[a_transient > TRANSIENT_ACTIVE_RATIO * peak]
    if attivi.size == 0:
        attivi = a_transient

    mediana = float(np.median(attivi))
    # MAD riscalata a deviazione standard
    mad = 1.4826 * float(np.median(np.abs(attivi - mediana)))
    tenuti = attivi[attivi <= mediana + mad_k * mad] if mad > 0 else attivi  # scarta gli spike
    if tenuti.size == 0:
        tenuti = attivi

    amax = float(np.median(tenuti))
    return amax if np.isfinite(amax) and amax > 0 else None


# Classifica il profilo osservato in trapezoidale o triangolare, gestendo le stime
# mancanti.
def classify_profile_type(
    delta_q: float, vmax: Optional[float], amax: Optional[float]
) -> str:
    if vmax is None:
        return "triangular"  # nessuna fase a velocita' costante osservata
    if amax is None or not np.isfinite(amax) or amax <= 0:
        return "unknown"
    if not np.isfinite(vmax) or vmax <= 0 or not np.isfinite(delta_q):
        return "unknown"
    return _classify(delta_q, vmax, amax)


# ---------------------------------------------------------------------------
# Orchestrazione
# ---------------------------------------------------------------------------


# Esegue l'intera pipeline del Modulo 1 su un movimento: filtro, derivate, regime e
# stime.
def analyze_joint_movement(
    time=None,
    position=None,
    fs: Optional[float] = None,
    dt: Optional[float] = None,
    cutoff_hz: float = 3.0,
    order: int = 4,
    config: Optional[PreprocessingConfig] = None,
    joint_name: str = "",
    segment_index: int = -1,
    source: str = "",
    keep_signals: bool = True,
) -> JointMovementResult:
    cfg = config or PreprocessingConfig()
    msgs: list[str] = []

    if position is None:
        raise ValueError("Il parametro 'position' e' obbligatorio.")
    position = np.asarray(position, dtype=float).ravel()

    if dt is None and fs is not None:
        dt = 1.0 / fs
    if dt is None:
        if time is None:
            raise ValueError("Serve almeno uno fra 'time', 'fs' e 'dt'.")
        t = np.asarray(time, dtype=float).ravel()
        if t.size < 2:
            raise ValueError("Asse dei tempi troppo corto per dedurre dt.")
        report = check_sampling_uniformity(t)
        dt = report["dt_median"]
        if not report["is_uniform"]:
            msgs.append(
                f"campionamento non uniforme: CV(dt) = {report['cv'] * 100:.2f}% "
                f"> {JITTER_CV_THRESHOLD * 100:.2f}%"
            )
    if fs is None:
        fs = 1.0 / dt

    time = (
        np.arange(position.size, dtype=float) * dt
        if time is None
        else np.asarray(time, dtype=float).ravel()
    )

    res = JointMovementResult(
        vmax_est=None,
        amax_est=None,
        profile_type="unknown",
        steady_state_window=None,
        warnings=msgs,
        delta_q=float(position[-1] - position[0]) if position.size else np.nan,
        duration=float(time[-1] - time[0]) if time.size else np.nan,
        fs=float(fs),
        cutoff_hz=float(cutoff_hz),
        joint_name=joint_name,
        segment_index=segment_index,
        source=source,
    )

    if position.size < 3:
        msgs.append("segmento troppo corto per essere analizzato")
        res.reliable = False
        return res

    if float(np.ptp(position)) == 0.0:
        msgs.append("posizione rigorosamente costante: nessun movimento")
        res.reliable = False
        res.profile_type = "no_motion"
        if keep_signals:
            res.time, res.position_raw = time, position
            res.position_filtered = position.copy()
            res.velocity = res.acceleration = np.zeros_like(position)
        return res

    # --- filtraggio e derivate ---------------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        position_filtered = filter_position(position, fs=fs, cutoff_hz=cutoff_hz, order=order)
    velocity, acceleration = compute_derivatives(position_filtered, dt)

    if keep_signals:
        res.time, res.position_raw = time, position
        res.position_filtered = position_filtered
        res.velocity, res.acceleration = velocity, acceleration

    if abs(res.delta_q) < cfg.min_travel_rad and float(np.ptp(position)) < cfg.min_travel_rad:
        msgs.append(
            f"spostamento trascurabile ({abs(res.delta_q):.4f} rad < "
            f"{cfg.min_travel_rad} rad): verosimilmente solo rumore"
        )
        res.reliable = False
        res.profile_type = "no_motion"
        return res

    # --- fase a regime ------------------------------------------------------
    # Il filtro smussa i gradini di accelerazione su circa fs/cutoff_hz
    # campioni: attorno al picco di un profilo *triangolare* si crea cosi' un
    # breve tratto ad accelerazione quasi nulla che non e' un vero plateau.
    duration_floor = max(
        MIN_DURATION_SAMPLES_FLOOR, int(np.ceil(FILTER_FLOOR_PERIODS * fs / cutoff_hz))
    )
    finestra = detect_steady_state(
        velocity,
        acceleration,
        accel_threshold_ratio=cfg.accel_threshold_ratio,
        min_duration_ratio=cfg.min_duration_ratio,
        min_duration_samples_floor=duration_floor,
        max_gap_samples=cfg.max_gap_samples,
        plateau_cv_max=cfg.plateau_cv_max,
    )
    res.steady_state_window = finestra
    if finestra is None:
        msgs.append("fase a regime non rilevata")
    else:
        res.steady_state_duration = float((finestra[1] - finestra[0]) * dt)

    # --- vmax, amax, classificazione ---------------------------------------
    vmax, triangolare = estimate_vmax(velocity, finestra)
    res.vmax_est = vmax
    if triangolare:
        msgs.append("nessun plateau di velocita': profilo probabilmente triangolare")

    amax = estimate_amax(acceleration, finestra, position=position_filtered)
    res.amax_est = amax
    if amax is None:
        msgs.append("amax non stimabile: nessun transitorio utilizzabile")

    res.profile_type = classify_profile_type(res.delta_q, vmax, amax)

    # Il filtro smussa i gradini su circa un periodo del cutoff: se la rampa
    # dura meno, il plateau di accelerazione non e' mai raggiunto nel segnale
    # filtrato e amax e' una sottostima, da usare come limite inferiore.
    if vmax is not None and amax is not None and amax > 0:
        t_ramp, t_filter = vmax / amax, AMAX_RAMP_PERIODS / cutoff_hz
        if t_ramp < t_filter:
            res.amax_reliable = False
            msgs.append(
                f"rampa di {t_ramp * 1e3:.0f} ms piu' breve dello smussamento del "
                f"filtro ({t_filter * 1e3:.0f} ms a {cutoff_hz:g} Hz): amax e' "
                "SOTTOSTIMATA, va usata come limite inferiore"
            )

    return res


# Applica l'analisi a tutti i movimenti di una sessione e ne raccoglie i risultati.
def analyze_session(
    path: str | Path,
    config: Optional[PreprocessingConfig] = None,
    cutoff_hz: Optional[float] = None,
    fs: Optional[float] = None,
    gap_factor: float = 5.0,
    min_samples: int = 20,
    keep_signals: bool = False,
    label: Optional[str] = None,
) -> tuple[list[JointMovementResult], pd.DataFrame]:
    cfg = config or PreprocessingConfig()
    df = load_position_data(path)
    source = df.attrs.get("source", str(path))
    if label is None:
        label = Path(source).parent.name

    time = df.iloc[:, 0].to_numpy()
    joint_names = list(df.columns[1:])
    segmenti = segment_movements(
        time,
        df[joint_names].to_numpy(),
        joint_names=joint_names,
        gap_factor=gap_factor,
        min_samples=min_samples,
        min_travel_rad=cfg.min_travel_rad,
    )

    risultati = [
        analyze_joint_movement(
            time=seg.time,
            position=seg.position,
            fs=fs,
            cutoff_hz=cutoff_hz if cutoff_hz is not None else cfg.cutoff_hz,
            order=cfg.filter_order,
            config=cfg,
            joint_name=seg.joint_name,
            segment_index=seg.index,
            source=source,
            keep_signals=keep_signals,
        )
        for seg in segmenti
    ]

    table = pd.DataFrame([r.to_record() for r in risultati])
    if not table.empty:
        table.insert(0, "session", label)
        table.insert(1, "direction", np.sign(table["delta_q"]).astype(int))
    return risultati, table


# Analizza tutte le sessioni del dataset, ricavando la velocita' comandata dal nome
# della cartella.
def analyze_dataset(
    root: str | Path,
    config: Optional[PreprocessingConfig] = None,
    cutoff_hz: Optional[float] = None,
    keep_signals: bool = False,
    verbose: bool = False,
) -> tuple[list[JointMovementResult], pd.DataFrame]:
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"{root} non e' una cartella.")

    tutti: list[JointMovementResult] = []
    tabelle: list[pd.DataFrame] = []

    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        try:
            risultati, table = analyze_session(
                folder, config=config, cutoff_hz=cutoff_hz,
                keep_signals=keep_signals, label=folder.name,
            )
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("sessione '%s' saltata: %s", folder.name, exc)
            continue
        if verbose:
            print(f"{folder.name:>8}: {len(risultati)} movimenti analizzati")
        tutti.extend(risultati)
        if not table.empty:
            tabelle.append(table)

    if not tabelle:
        return tutti, pd.DataFrame()

    table = pd.concat(tabelle, ignore_index=True)

    # Ricava la velocita' comandata dal nome della sessione.
    def _velocita_comandata(nome: str) -> float:
        try:
            return float(str(nome).split("_")[-1])
        except ValueError:
            return np.nan

    table.insert(1, "commanded_velocity", table["session"].map(_velocita_comandata))
    return tutti, table


# Salva la tabella delle stime in CSV, creando le cartelle mancanti.
def save_results(table: pd.DataFrame, path: str | Path, float_format: str = "%.6g") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, float_format=float_format)
    return path
