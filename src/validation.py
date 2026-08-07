#!/usr/bin/env python3
"""Modulo 4 — Validazione dei residui e confronto fra modelli.

Il Modulo 3 dice *quanto bene* il modello riproduce i dati; questo stabilisce se
sia la descrizione **giusta**. Ordina i profili candidati per AIC/BIC sullo
stesso movimento e misura i residui, tutto sulla finestra di moto: fuori da
quella il modello e' costante e il residuo e' identicamente nullo.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

try:  # pragma: no cover - dipende da come e' avviato l'interprete
    from src.fitting import load_movements_from_processed
    from src.trajectory_models import model_position, n_params, param_names, total_time
except ImportError:  # pragma: no cover
    from fitting import load_movements_from_processed
    from trajectory_models import model_position, n_params, param_names, total_time

__all__ = [
    "ValidationConfig",
    "ResidualsSummary",
    "motion_window",
    "compute_residuals_summary",
    "compute_aic_bic",
    "CANDIDATI_DEFAULT",
    "compare_models",
    "aggregate_model_comparison",
    "check_amax_consistency_across_speeds",
    "validate_all_movements",
    "summarize_by_joint",
    "save_validation_results",
    "plot_residuals_vs_time",
    "plot_residuals_histogram",
    "plot_residuals_qq",
    "plot_amax_across_speeds",
]

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Dove il colore codifica uno stato e' affiancato dalla forma del marcatore,
# cosi' l'informazione non passa mai dal solo colore.
COLORE_ACCENTO = "#ff7f0e"      # grandezza identificabile / in evidenza
COLORE_ATTENUATO = "#7f7f7f"    # grandezza non identificabile / sfondo
COLORE_SERIE = "#1a5a8a"        # serie singola (residui)
COLORE_RIFERIMENTO = "#1a1a1a"  # medie, soglie, linee guida

#: Profili messi a confronto da :func:`compare_models`.
CANDIDATI_DEFAULT = ("trapezoidal", "asymmetric_trapezoidal", "quintic")


# Soglie della validazione: quantizzazione dell'encoder, residuo massimo accettabile e
# criteri di identificabilita'.
@dataclass
class ValidationConfig:
    cv_threshold: float = 0.15
    min_ramp_samples: int = 5
    use_identifiable_only: bool = True
    encoder_quantum: float = 1.4e-3
    rmse_quanta_threshold: float = 4.0
    motion_margin_samples: int = 0


# Residui di un movimento con le metriche sulla finestra di moto e sul segmento intero.
@dataclass
class ResidualsSummary:
    residuals: np.ndarray = field(repr=False)
    motion_mask: np.ndarray = field(repr=False)
    rmse: float
    mean: float
    std: float
    max_abs: float
    n_samples: int
    rmse_full: float
    n_samples_full: int
    motion_fraction: float
    profile_type: str

    # Residui della sola finestra di moto.
    @property
    def motion_residuals(self) -> np.ndarray:
        return self.residuals[self.motion_mask]

    # Somma dei quadrati dei residui sulla finestra di moto.
    @property
    def rss(self) -> float:
        return float(np.sum(self.motion_residuals**2))


# Estrae i parametri nell'ordine richiesto dal profilo, da un dizionario o da una
# sequenza.
def _vettore_parametri(fitted_params, profile_type: str) -> np.ndarray:
    nomi = param_names(profile_type)
    if isinstance(fitted_params, Mapping):
        valori = []
        for nome in nomi:
            if nome in fitted_params:
                valori.append(float(fitted_params[nome]))
            elif f"{nome}_fit" in fitted_params:
                valori.append(float(fitted_params[f"{nome}_fit"]))
            else:
                raise KeyError(f"parametro '{nome}' assente per il profilo '{profile_type}'.")
        return np.array(valori, dtype=float)

    p = np.asarray(fitted_params, dtype=float).ravel()
    if p.size != len(nomi):
        raise ValueError(f"attesi {len(nomi)} parametri per '{profile_type}', ricevuti {p.size}.")
    return p


# ---------------------------------------------------------------------------
# Finestra di moto e residui
# ---------------------------------------------------------------------------


# Maschera i campioni in cui il movimento e' in corso, escludendo le fasi di quiete.
def motion_window(
    t_data, fitted_params, profile_type: str, q0: float, qf: float,
    margin_samples: int = 0,
) -> np.ndarray:
    t_data = np.asarray(t_data, dtype=float).ravel()
    if t_data.size == 0:
        return np.zeros(0, dtype=bool)

    p = _vettore_parametri(fitted_params, profile_type)
    t0 = float(p[param_names(profile_type).index("t0")])
    t_total = total_time(profile_type, q0, qf, p)
    if not np.isfinite(t_total) or t_total <= 0:
        return np.zeros(t_data.size, dtype=bool)

    passo = float(np.median(np.diff(t_data))) if t_data.size > 1 else 0.0
    margine = max(0, int(margin_samples)) * passo
    # solo i campioni in movimento
    return (t_data >= t0 - margine) & (t_data <= t0 + t_total + margine)


# Ricalcola il modello coi parametri identificati e misura lo scarto dai dati sulla
# finestra di moto.
def compute_residuals_summary(
    t_data, q_data, q0: float, qf: float, fitted_params, profile_type: str,
    margin_samples: int = 0,
    motion_mask=None,
    min_motion_samples: int = 10,
) -> ResidualsSummary:
    t_data = np.asarray(t_data, dtype=float).ravel()
    q_data = np.asarray(q_data, dtype=float).ravel()
    if t_data.size != q_data.size:
        raise ValueError(f"t_data ({t_data.size}) e q_data ({q_data.size}) di lunghezza diversa.")
    if t_data.size == 0:
        raise ValueError("nessun campione da analizzare.")
    if q0 == qf:
        raise ValueError("spostamento nullo: nessun profilo da confrontare.")

    p = _vettore_parametri(fitted_params, profile_type)
    residui = q_data - np.asarray(model_position(profile_type, t_data, q0, qf, p), dtype=float)

    if motion_mask is not None:
        maschera = np.asarray(motion_mask, dtype=bool).ravel()
        if maschera.size != residui.size:
            raise ValueError(f"motion_mask ha {maschera.size} elementi, i dati {residui.size}.")
    else:
        maschera = motion_window(t_data, fitted_params, profile_type, q0, qf, margin_samples)

    if int(np.count_nonzero(maschera)) < min_motion_samples:
        maschera = np.ones(residui.size, dtype=bool)

    in_moto = residui[maschera]
    return ResidualsSummary(
        residuals=residui,
        motion_mask=maschera,
        rmse=float(np.sqrt(np.mean(in_moto**2))),
        mean=float(np.mean(in_moto)),
        std=float(np.std(in_moto, ddof=1)) if in_moto.size > 1 else 0.0,
        max_abs=float(np.max(np.abs(in_moto))),
        n_samples=int(in_moto.size),
        rmse_full=float(np.sqrt(np.mean(residui**2))),
        n_samples_full=int(residui.size),
        motion_fraction=float(np.mean(maschera)),
        profile_type=profile_type,
    )


# Calcola AIC e BIC dai residui e dal numero di parametri, per confrontare modelli di
# complessita' diversa.
def compute_aic_bic(residuals, n_parametri: int) -> tuple[float, float]:
    residui = np.asarray(residuals, dtype=float).ravel()
    n = residui.size
    if n == 0:
        return np.nan, np.nan

    rss = float(np.sum(residui**2))
    if rss <= 0:
        # Fit esatto: la verosimiglianza diverge. Nel confronto e' l'esito
        # corretto, nessuna penalizzazione lo batte.
        return -np.inf, -np.inf

    log_ver = n * np.log(rss / n)  # verosimiglianza a meno di costanti additive
    return float(log_ver + 2 * n_parametri), float(log_ver + n_parametri * np.log(n))


# ---------------------------------------------------------------------------
# Confronto fra modelli
# ---------------------------------------------------------------------------


# Fitta piu' profili candidati sullo stesso movimento e li ordina per AIC/BIC,
# valutandoli tutti sulla stessa finestra.
def compare_models(
    t_data, q_data, q0: float, qf: float,
    vmax_est: float, amax_est: float,
    t0_est: Optional[float] = None,
    candidates: Sequence[str] = CANDIDATI_DEFAULT,
    reference: str = "trapezoidal",
    config: Optional[ValidationConfig] = None,
    fit_config=None,
) -> pd.DataFrame:
    try:  # import differito: fitting non dipende da validation
        from src.fitting import build_initial_guess, estimate_t0, fit_single_movement
    except ImportError:  # pragma: no cover
        from fitting import build_initial_guess, estimate_t0, fit_single_movement

    cfg = config or ValidationConfig()
    candidati = tuple(candidates)
    if reference not in candidati:
        raise ValueError(f"il riferimento '{reference}' deve essere fra i candidati {candidati}.")

    t_data = np.asarray(t_data, dtype=float).ravel()
    q_data = np.asarray(q_data, dtype=float).ravel()
    delta_q = float(qf) - float(q0)
    if t0_est is None:
        t0_est = estimate_t0(t_data, q_data)

    # Fitta un singolo profilo candidato sul movimento.
    def _fitta(nome: str):
        x0 = build_initial_guess(vmax_est, amax_est, t0_est, nome, delta_q=delta_q)
        return fit_single_movement(t_data, q_data, q0, qf, x0,
                                   profile_type=nome, config=fit_config)

    # 1. Il riferimento definisce la finestra comune.
    rif = _fitta(reference)
    if not rif.success:
        logger.warning("il riferimento '%s' non converge: confronto non eseguibile", reference)
        return pd.DataFrame()

    maschera = motion_window(t_data, rif.params, reference, q0, qf, cfg.motion_margin_samples)
    n_finestra = int(np.count_nonzero(maschera))
    if n_finestra < 10:
        maschera = np.ones(t_data.size, dtype=bool)
        n_finestra = int(t_data.size)

    # Estremi **analitici** della finestra, non il primo e l'ultimo campione che
    # ci cadono dentro: quelli stanno per costruzione strettamente all'interno, e
    # confrontarci il t0 di un candidato darebbe sempre esito negativo.
    passo = float(np.median(np.diff(t_data))) if t_data.size > 1 else 0.0
    margine = max(0, int(cfg.motion_margin_samples)) * passo
    t0_rif = float(rif.params["t0"])
    t_tot_rif = total_time(reference, q0, qf, _vettore_parametri(rif.params, reference))
    t_inizio, t_fine = t0_rif - margine, t0_rif + t_tot_rif + margine

    # 2. Tutti i candidati, valutati sulla stessa finestra.
    righe = []
    for nome in candidati:
        esito = rif if nome == reference else _fitta(nome)
        k = n_params(nome)
        riga = {"profile_type": nome, "n_params": k,
                "success": bool(esito.success), "n_finestra": n_finestra}

        if not esito.success:
            righe.append({**riga, "rmse": np.nan, "aic": np.nan, "bic": np.nan,
                          "nota": esito.message})
            continue

        sommario = compute_residuals_summary(
            t_data, q_data, q0, qf, esito.params, nome, motion_mask=maschera
        )
        aic, bic = compute_aic_bic(sommario.motion_residuals, k)
        t_total = total_time(nome, q0, qf, _vettore_parametri(esito.params, nome))
        t0 = float(esito.params["t0"])

        righe.append({
            **riga,
            "params": " ".join(f"{n}={v:.5g}" for n, v in esito.params.items()),
            "rmse": sommario.rmse,
            "rmse_quanti": sommario.rmse / cfg.encoder_quantum,
            "rss": sommario.rss,
            "aic": aic,
            "bic": bic,
            "t_total": t_total,
            "finestra_copre": bool(t0 >= t_inizio - 1e-9 and t0 + t_total <= t_fine + 1e-9),
            "ramp_samples": esito.ramp_samples,
            "nota": "",
        })

    tabella = pd.DataFrame(righe)
    if tabella.empty:
        return tabella

    base = tabella.loc[tabella.profile_type == reference]
    tabella["delta_aic"] = tabella["aic"] - (float(base.aic.iloc[0]) if len(base) else np.nan)
    tabella["delta_bic"] = tabella["bic"] - (float(base.bic.iloc[0]) if len(base) else np.nan)
    tabella["riferimento"] = reference
    return tabella


# Riassume il confronto su tutti i movimenti, contando le vittorie di ciascun candidato.
def aggregate_model_comparison(per_movimento: pd.DataFrame) -> pd.DataFrame:
    if per_movimento.empty:
        return pd.DataFrame()

    validi = per_movimento[per_movimento["success"].astype(bool)]
    vittorie: dict[str, int] = {}
    decisive: dict[str, int] = {}

    if "movimento" in per_movimento.columns and not validi.empty:
        for _, gruppo in validi.groupby("movimento"):
            if gruppo["aic"].isna().all():
                continue
            vincitore = gruppo.loc[gruppo["aic"].idxmin(), "profile_type"]
            vittorie[vincitore] = vittorie.get(vincitore, 0) + 1
            migliori = gruppo["aic"].nsmallest(2)
            if len(migliori) > 1 and (migliori.iloc[1] - migliori.iloc[0]) > 10:
                decisive[vincitore] = decisive.get(vincitore, 0) + 1

    righe = []
    for nome, gruppo in per_movimento.groupby("profile_type", sort=False):
        ok = gruppo[gruppo["success"].astype(bool)]
        righe.append({
            "profile_type": nome,
            "n_params": int(gruppo["n_params"].iloc[0]),
            "n_movimenti": len(gruppo),
            "n_convergenti": len(ok),
            "rmse_mediano": float(ok["rmse"].median()) if len(ok) else np.nan,
            "rmse_quanti_mediano": float(ok["rmse_quanti"].median()) if len(ok) else np.nan,
            "delta_aic_mediano": float(ok["delta_aic"].median()) if len(ok) else np.nan,
            "delta_bic_mediano": float(ok["delta_bic"].median()) if len(ok) else np.nan,
            "n_vittorie": vittorie.get(nome, 0),
            "n_vittorie_decisive": decisive.get(nome, 0),
            "n_finestra_incompleta": (
                int((~ok["finestra_copre"].astype(bool)).sum())
                if "finestra_copre" in ok and len(ok) else 0
            ),
        })

    return pd.DataFrame(righe).sort_values("n_vittorie", ascending=False, ignore_index=True)


# ---------------------------------------------------------------------------
# Consistenza fisica dei parametri
# ---------------------------------------------------------------------------


# Coefficiente di variazione di una serie di valori, nan se non calcolabile.
def _cv(valori) -> float:
    v = np.asarray(valori, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return np.nan
    media = float(np.mean(v))
    return float(np.std(v, ddof=1) / abs(media)) if media != 0 else np.inf


# Verifica che amax sia una costante del giunto e non dipenda dalla velocita' comandata.
def check_amax_consistency_across_speeds(
    fit_results: pd.DataFrame, joint_id: str, config: Optional[ValidationConfig] = None,
) -> dict:
    cfg = config or ValidationConfig()
    for colonna in ("joint_name", "commanded_velocity", "amax_fit"):
        if colonna not in fit_results.columns:
            raise ValueError(f"colonna '{colonna}' assente da fit_results.")

    dati = fit_results[fit_results["joint_name"] == joint_id]
    if dati.empty:
        raise ValueError(f"nessun movimento per il giunto '{joint_id}'.")
    if "success" in dati.columns:
        dati = dati[dati["success"].astype(bool)]

    if "amax_identifiable" in dati.columns:
        identificabili = dati["amax_identifiable"].astype(bool)
    elif "ramp_samples" in dati.columns:
        identificabili = dati["ramp_samples"] >= cfg.min_ramp_samples
    else:
        identificabili = pd.Series(True, index=dati.index)

    per_vel_raw = dati.groupby("commanded_velocity")["amax_fit"].mean()
    per_vel_filt = dati[identificabili].groupby("commanded_velocity")["amax_fit"].mean()

    cv_raw, cv_filt = _cv(per_vel_raw.to_numpy()), _cv(per_vel_filt.to_numpy())
    cv_giudicato = cv_filt if cfg.use_identifiable_only else cv_raw
    flagged = bool(np.isfinite(cv_giudicato) and cv_giudicato > cfg.cv_threshold)
    if flagged:
        logger.warning("giunto %s: amax non coerente fra le velocita' (CV = %.1f%%)",
                       joint_id, cv_giudicato * 100)

    return {
        "joint_id": joint_id,
        "cv_raw": cv_raw,
        "cv_filtered": cv_filt,
        "mean_raw": float(per_vel_raw.mean()) if per_vel_raw.size else np.nan,
        "mean_filtered": float(per_vel_filt.mean()) if per_vel_filt.size else np.nan,
        "std_filtered": float(per_vel_filt.std(ddof=1)) if per_vel_filt.size > 1 else np.nan,
        "per_velocity_raw": {float(k): float(v) for k, v in per_vel_raw.items()},
        "per_velocity_filtered": {float(k): float(v) for k, v in per_vel_filt.items()},
        "n_velocities_raw": int(per_vel_raw.size),
        "n_velocities_filtered": int(per_vel_filt.size),
        "n_movements_identifiable": int(identificabili.sum()),
        "n_movements_total": int(len(dati)),
        "flagged": flagged,
    }


# ---------------------------------------------------------------------------
# Validazione dell'intero dataset
# ---------------------------------------------------------------------------


# Applica la validazione a tutti i movimenti fittati e raccoglie residui e metriche in
# una tabella.
def validate_all_movements(
    fit_results_csv: str | Path,
    params_csv: Optional[str | Path] = None,
    raw_root: Optional[str | Path] = None,
    config: Optional[ValidationConfig] = None,
    output_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    cfg = config or ValidationConfig()
    fit = pd.read_csv(fit_results_csv)
    if params_csv is None:
        params_csv = Path(fit_results_csv).with_name("preprocessing_params.csv")

    movimenti = load_movements_from_processed(params_csv, raw_root=raw_root)
    per_chiave = {
        (m["session"], m["joint_name"], int(m["segment_index"])): m
        for m in movimenti.values()
    }

    righe = []
    for _, riga in fit.iterrows():
        chiave = (riga["session"], riga["joint_name"], int(riga["segment_index"]))
        base = {
            "session": riga["session"],
            "joint_name": riga["joint_name"],
            "segment_index": int(riga["segment_index"]),
            "commanded_velocity": riga.get("commanded_velocity", np.nan),
            "profile_type": riga.get("profile_type", "unknown"),
            "delta_q": riga.get("delta_q", np.nan),
        }

        mov = per_chiave.get(chiave)
        if mov is None or not bool(riga.get("success", False)):
            motivo = "fit non convergente" if mov is not None else "dati grezzi non ritrovati"
            righe.append({**base, "validato": False, "motivo": motivo})
            continue

        try:
            k = n_params(base["profile_type"])
            sommario = compute_residuals_summary(
                mov["t_data"], mov["q_data"], mov["q0"], mov["qf"],
                riga.to_dict(), base["profile_type"],
                margin_samples=cfg.motion_margin_samples,
            )
        except (ValueError, KeyError) as exc:
            logger.warning("movimento %s: validazione fallita - %s", chiave, exc)
            righe.append({**base, "validato": False, "motivo": str(exc)})
            continue

        in_moto = sommario.motion_residuals
        rmse_quanti = sommario.rmse / cfg.encoder_quantum
        aic, bic = compute_aic_bic(in_moto, k)

        righe.append({
            **base,
            "validato": True,
            "motivo": "",
            "n_samples": sommario.n_samples,
            "n_samples_completo": sommario.n_samples_full,
            "frazione_moto": sommario.motion_fraction,
            "n_params": k,
            "rmse": sommario.rmse,
            "rmse_quanti": rmse_quanti,
            "rmse_completo": sommario.rmse_full,
            "rmse_quanti_completo": sommario.rmse_full / cfg.encoder_quantum,
            "residuo_medio": sommario.mean,
            "residuo_std": sommario.std,
            "residuo_max_abs": sommario.max_abs,
            "aic": aic,
            "bic": bic,
            # Il criterio discriminante: residuo grande rispetto al suo
            # pavimento fisico.
            "fuori_norma": bool(rmse_quanti > cfg.rmse_quanta_threshold),
        })

    tabella = pd.DataFrame(righe)
    logger.info(
        "validazione completata: %d/%d movimenti, %d fuori norma",
        int(tabella.get("validato", pd.Series(dtype=bool)).sum()), len(tabella),
        int(tabella.get("fuori_norma", pd.Series(dtype=bool)).sum()),
    )
    if output_path is not None:
        save_validation_results(tabella, output_path)
    return tabella


# Aggrega la validazione per giunto, una riga ciascuno.
def summarize_by_joint(
    validation_results: pd.DataFrame, fit_results: pd.DataFrame,
    config: Optional[ValidationConfig] = None,
) -> pd.DataFrame:
    cfg = config or ValidationConfig()

    righe = []
    for giunto, gruppo in validation_results.groupby("joint_name"):
        validi = gruppo[gruppo["validato"]] if "validato" in gruppo else gruppo
        try:
            amax_rep = check_amax_consistency_across_speeds(fit_results, str(giunto), cfg)
        except ValueError as exc:
            logger.warning("giunto %s: consistenza amax non calcolabile - %s", giunto, exc)
            amax_rep = None

        righe.append({
            "joint_name": giunto,
            "n_movimenti": len(gruppo),
            "n_validati": len(validi),
            "n_fuori_norma": int(validi.get("fuori_norma", pd.Series(dtype=bool)).sum()),
            "rmse_mediano": float(validi["rmse"].median()) if len(validi) else np.nan,
            "rmse_quanti_mediano": float(validi["rmse_quanti"].median()) if len(validi) else np.nan,
            "amax_cv_grezzo": amax_rep["cv_raw"] if amax_rep else np.nan,
            "amax_cv_filtrato": amax_rep["cv_filtered"] if amax_rep else np.nan,
            "amax_medio": amax_rep["mean_filtered"] if amax_rep else np.nan,
            "amax_n_vel_utili": amax_rep["n_velocities_filtered"] if amax_rep else 0,
            "amax_coerente": (not amax_rep["flagged"]) if amax_rep else False,
        })

    return pd.DataFrame(righe)


# Salva la tabella di validazione in CSV, creando le cartelle mancanti.
def save_validation_results(table: pd.DataFrame, path: str | Path,
                            float_format: str = "%.6g") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, float_format=float_format)
    return path


# ---------------------------------------------------------------------------
# Grafici
# ---------------------------------------------------------------------------


# Salva la figura sul percorso indicato, creando le cartelle mancanti.
def _salva(fig, output_path: Optional[str | Path]) -> Optional[Path]:
    if output_path is None:
        return None
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return output_path


# Disegna l'andamento temporale dei residui, con banda di quantizzazione e finestra di
# analisi.
def plot_residuals_vs_time(
    t_data, residuals, title: str = "", config: Optional[ValidationConfig] = None,
    output_path: Optional[str | Path] = None, ax=None, mask=None,
):
    import matplotlib.pyplot as plt

    cfg = config or ValidationConfig()
    t = np.asarray(t_data, dtype=float).ravel()
    e = np.asarray(residuals, dtype=float).ravel()
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(8, 3.2))

    q = cfg.encoder_quantum * 1e3
    ax.axhspan(-q, q, color=COLORE_ATTENUATO, alpha=0.20, zorder=0,
               label="quantizzazione encoder")
    if mask is not None:
        m = np.asarray(mask, dtype=bool).ravel()
        if m.any():
            ax.axvspan(float(t[m].min()), float(t[m].max()), color=COLORE_ACCENTO,
                       alpha=0.12, zorder=0, label="finestra di analisi")
    ax.axhline(0, color=COLORE_RIFERIMENTO, lw=0.8, zorder=1)
    ax.plot(t, e * 1e3, lw=1.2, color=COLORE_SERIE, zorder=2)

    ax.set(xlabel="tempo [s]", ylabel="residuo [mrad]")
    if title:
        ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    _salva(fig, output_path)
    return ax


# Disegna la distribuzione dei residui con la gaussiana di pari media e deviazione.
def plot_residuals_histogram(
    residuals, title: str = "", bins: int = 40,
    output_path: Optional[str | Path] = None, ax=None, mask=None,
):
    import matplotlib.pyplot as plt

    e = np.asarray(residuals, dtype=float).ravel()
    if mask is not None:
        e = e[np.asarray(mask, dtype=bool).ravel()]
    e = e * 1e3
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(6, 3.4))

    ax.hist(e, bins=bins, density=True, color=COLORE_SERIE, alpha=0.75,
            edgecolor="white", linewidth=0.5, label="residui osservati")
    if e.size > 1 and np.std(e) > 0:
        x = np.linspace(e.min(), e.max(), 400)
        ax.plot(x, stats.norm.pdf(x, np.mean(e), np.std(e, ddof=1)), lw=2,
                color=COLORE_ACCENTO, label="gaussiana di pari media e $\\sigma$")

    ax.set(xlabel="residuo [mrad]", ylabel="densita'")
    if title:
        ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _salva(fig, output_path)
    return ax


# Diagramma quantile-quantile dei residui contro la normale: mostra *come* si discostano,
# non solo quanto.
def plot_residuals_qq(
    residuals, title: str = "", output_path: Optional[str | Path] = None,
    ax=None, mask=None,
):
    import matplotlib.pyplot as plt

    e = np.asarray(residuals, dtype=float).ravel()
    if mask is not None:
        e = e[np.asarray(mask, dtype=bool).ravel()]
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(5.2, 4.4))

    # Casi degeneri: si disegna un riquadro con la scritta invece di sollevare.
    if e.size < 3 or not np.all(np.isfinite(e)) or np.std(e) == 0:
        ax.text(0.5, 0.5, "residui non utilizzabili\nper il Q-Q plot",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)
        ax.set(xlabel="quantili teorici", ylabel="residui ordinati [mrad]")
        _salva(fig, output_path)
        return ax

    (teorici, ordinati), (pendenza, intercetta, _) = stats.probplot(e, dist="norm")
    ax.plot(teorici, ordinati * 1e3, ".", ms=4, alpha=0.7, color=COLORE_SERIE,
            label="residui osservati")
    ax.plot(teorici, (pendenza * teorici + intercetta) * 1e3, lw=1.6,
            color=COLORE_ACCENTO, label="normale di riferimento")

    ax.set(xlabel="quantili teorici (normale standard)", ylabel="residui ordinati [mrad]")
    if title:
        ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    _salva(fig, output_path)
    return ax


# Disegna amax contro la velocita' comandata, distinguendo i valori identificabili.
def plot_amax_across_speeds(
    fit_results: pd.DataFrame, joint_id: str, config: Optional[ValidationConfig] = None,
    output_path: Optional[str | Path] = None, ax=None,
):
    import matplotlib.pyplot as plt

    cfg = config or ValidationConfig()
    dati = fit_results[fit_results["joint_name"] == joint_id]
    if dati.empty:
        raise ValueError(f"nessun movimento per il giunto '{joint_id}'.")
    if "success" in dati.columns:
        dati = dati[dati["success"].astype(bool)]

    if "amax_identifiable" in dati.columns:
        ident = dati["amax_identifiable"].astype(bool)
    elif "ramp_samples" in dati.columns:
        ident = dati["ramp_samples"] >= cfg.min_ramp_samples
    else:
        ident = pd.Series(True, index=dati.index)

    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(5.5, 3.6))
    no, si = dati[~ident], dati[ident]

    if len(no):
        ax.scatter(no["commanded_velocity"], no["amax_fit"], s=42, marker="x",
                   linewidths=1.6, color=COLORE_ATTENUATO, zorder=2,
                   label=f"non identificabile (n={len(no)})")
    if len(si):
        ax.scatter(si["commanded_velocity"], si["amax_fit"], s=44, marker="o",
                   color=COLORE_ACCENTO, edgecolor="white", linewidth=0.8, zorder=3,
                   label=f"identificabile (n={len(si)})")
        media = float(si["amax_fit"].mean())
        ax.axhline(media, color=COLORE_RIFERIMENTO, ls=":", lw=1.4, zorder=1,
                   label=f"media = {media:.2f} rad/s$^2$")

    ax.set(xlabel="velocita' comandata [1-100]", ylabel="$a_{max}$ [rad/s$^2$]", yscale="log")
    ax.set_title(f"Giunto {joint_id}", fontsize=10)
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.3, which="both")
    _salva(fig, output_path)
    return ax
