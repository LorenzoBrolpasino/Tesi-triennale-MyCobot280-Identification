#!/usr/bin/env python3
"""Genera le figure e le tabelle aggiuntive per l'elaborato.

    python3 scripts/genera_figure_tesi.py

Consuma i moduli gia' esistenti applicandoli all'intero dataset: nessuna analisi
nuova. Legge i CSV di ``data/processed/``, che vanno prodotti prima con
``run_identification.py``, e scrive in ``results/figures/`` e ``results/tables/``.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # nessuna finestra: lo script gira anche senza display

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RADICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RADICE))

from src.fitting import (  # noqa: E402
    build_initial_guess, estimate_t0, fit_single_movement,
    load_movements_from_processed,
)
from src.trajectory_models import (  # noqa: E402
    asymmetric_position, classify_profile_type, n_params, profile_times,
    quintic_position, total_time, trapezoidal_position,
)
from src.validation import (  # noqa: E402
    CANDIDATI_DEFAULT, COLORE_ACCENTO, COLORE_ATTENUATO, COLORE_RIFERIMENTO,
    COLORE_SERIE, compute_aic_bic, compute_residuals_summary, motion_window,
    plot_amax_across_speeds, plot_residuals_histogram, plot_residuals_qq,
    plot_residuals_vs_time,
    summarize_by_joint, validate_all_movements,
)

DATI_GREZZI = RADICE / "data/raw/dati_tesi"
ELABORATI = RADICE / "data/processed"
FIGURE = RADICE / "results/figures"
TABELLE = RADICE / "results/tables"

QUANTO = 1.4e-3  # quantizzazione dell'encoder [rad]

# Parametri identificati, usati nelle figure teoriche.
VMAX_RIF, A_ACC_RIF, A_DEC_RIF, DELTA_Q_RIF = 1.3, 8.70, 7.52, 4.2

ETICHETTE = {
    "trapezoidal": "trapezoidale",
    "asymmetric_trapezoidal": "trapezoidale asimmetrico",
    "quintic": "quintico",
}

TIPI = {"trapezoidal": "trapezoidale", "triangular": "triangolare"}

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})


# Salva la figura in results/figures/, creando le cartelle mancanti.
def _salva_figura(fig, nome: str) -> Path:
    percorso = FIGURE / nome
    percorso.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(percorso, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return percorso


# Salva la tabella in results/tables/, creando le cartelle mancanti.
def _salva_tabella(tabella: pd.DataFrame, nome: str) -> Path:
    percorso = TABELLE / nome
    percorso.parent.mkdir(parents=True, exist_ok=True)
    tabella.to_csv(percorso, index=False, float_format="%.6g")
    return percorso


# ---------------------------------------------------------------------------
# 1. Confronto fra i tre profili teorici (Modulo 2)
# ---------------------------------------------------------------------------


# Sovrappone i tre profili candidati a parita' di spostamento e durata totale.
def figura_1_confronto_profili() -> Path:
    q0, qf = 0.0, DELTA_Q_RIF
    amax = 2.0 / (1.0 / A_ACC_RIF + 1.0 / A_DEC_RIF)  # media armonica di a_acc e a_dec
    durata = profile_times(q0, qf, VMAX_RIF, amax).t_total

    t = np.arange(-0.06, durata * 1.08, 2e-4)  # griglia fitta per derivare i profili
    curve = [
        ("trapezoidale", trapezoidal_position(t, q0, qf, VMAX_RIF, amax), COLORE_SERIE, "-"),
        ("trapezoidale asimmetrico",
         asymmetric_position(t, q0, qf, VMAX_RIF, A_ACC_RIF, A_DEC_RIF), COLORE_ACCENTO, "-"),
        ("quintico", quintic_position(t, q0, qf, durata), COLORE_ATTENUATO, "--"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(9, 8.5), sharex=True)
    for nome, q, colore, stile in curve:
        v = np.gradient(q, t)
        axes[0].plot(t, q, lw=1.8, color=colore, ls=stile, label=nome)
        axes[1].plot(t, v, lw=1.8, color=colore, ls=stile)
        axes[2].plot(t, np.gradient(v, t), lw=1.8, color=colore, ls=stile)

    axes[1].axhline(VMAX_RIF, color=COLORE_RIFERIMENTO, ls=":", lw=1.2,
                    label=f"$v_{{max}}$ = {VMAX_RIF} rad/s")
    axes[2].axhline(0, color=COLORE_RIFERIMENTO, lw=0.8)

    axes[0].set(ylabel="posizione [rad]")
    axes[1].set(ylabel="velocita' [rad/s]")
    axes[2].set(ylabel="accelerazione [rad/s$^2$]", xlabel="tempo [s]")
    axes[0].legend(fontsize=9, loc="lower right")
    axes[1].legend(fontsize=9, loc="upper right")
    for ax in axes:
        ax.grid(alpha=0.3)

    fig.suptitle(f"Tre profili a parita' di spostamento ({qf:g} rad) e durata "
                 f"({durata:.2f} s)", fontweight="bold")
    percorso = _salva_figura(fig, "02b_confronto_profili.png")
    print(f"      durata comune {durata:.4f} s, amax armonica {amax:.4f} rad/s^2")
    return percorso


# ---------------------------------------------------------------------------
# 2. Soglia di degenerazione trapezoidale/triangolare (Modulo 2)
# ---------------------------------------------------------------------------


# Mostra come il plateau sparisce quando lo spostamento scende sotto la soglia.
def figura_2_soglia_degenerazione() -> Path:
    vmax, amax = VMAX_RIF, 8.0
    dq_crit = vmax**2 / amax

    fig, ax = plt.subplots(figsize=(8, 4.6))
    for fattore, colore, stile in [(3.0, COLORE_SERIE, "-"),
                                   (1.0, COLORE_ACCENTO, "-"),
                                   (0.4, COLORE_ATTENUATO, "--")]:
        dq = fattore * dq_crit
        tipo = classify_profile_type(dq, vmax, amax)
        durata = profile_times(0.0, dq, vmax, amax).t_total
        t = np.arange(-0.03, durata * 1.08, 1e-4)
        v = np.gradient(trapezoidal_position(t, 0.0, dq, vmax, amax), t)
        ax.plot(t, v, lw=1.8, color=colore, ls=stile,
                label=f"$\\Delta q$ = {fattore:g}·$\\Delta q_{{crit}}$ = {dq:.3f} rad "
                      f"— {TIPI[tipo]}")

    ax.axhline(vmax, color=COLORE_RIFERIMENTO, ls=":", lw=1.2,
               label=f"$v_{{max}}$ = {vmax} rad/s")
    ax.set(xlabel="tempo [s]", ylabel="velocita' [rad/s]",
           title=f"Soglia di degenerazione: $\\Delta q_{{crit}} = v_{{max}}^2/a_{{max}}$ "
                 f"= {dq_crit:.3f} rad")
    ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    ax.grid(alpha=0.3)
    return _salva_figura(fig, "02c_soglia_degenerazione.png")


# ---------------------------------------------------------------------------
# 3. Distribuzione di delta-AIC fra asimmetrico e simmetrico (Modulo 4)
# ---------------------------------------------------------------------------


# Istogramma di delta-AIC fra asimmetrico e simmetrico, piu' la tabella dei quantili.
def figura_3_distribuzione_delta_aic(confronto: pd.DataFrame) -> tuple[Path, Path]:
    d = confronto[confronto.profile_type == "asymmetric_trapezoidal"].delta_aic.dropna()
    n = len(d)
    n_vince = int((d < 0).sum())
    n_decisivo = int((d < -10).sum())

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hist(d, bins=30, color=COLORE_SERIE, alpha=0.75, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color=COLORE_RIFERIMENTO, lw=1.5,
               label=f"0 — a sinistra vince l'asimmetrico ({n_vince}/{n})")
    ax.axvline(-10, color=COLORE_ACCENTO, ls="--", lw=1.8,
               label=f"-10 — evidenza decisiva ({n_decisivo}/{n})")
    ax.set(xlabel="$\\Delta$AIC = AIC(asimmetrico) - AIC(trapezoidale)",
           ylabel="numero di movimenti",
           title=f"L'asimmetrico vince su {n_vince} movimenti su {n}, "
                 f"{n_decisivo} in modo decisivo")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)
    fig_path = _salva_figura(fig, "04b_distribuzione_delta_aic.png")

    tabella = pd.DataFrame([{
        "n_movimenti": n,
        "delta_aic_min": d.min(),
        "delta_aic_q1": d.quantile(0.25),
        "delta_aic_mediano": d.median(),
        "delta_aic_q3": d.quantile(0.75),
        "delta_aic_max": d.max(),
        "n_asimmetrico_vince": n_vince,
        "perc_asimmetrico_vince": 100.0 * n_vince / n,
        "n_evidenza_decisiva": n_decisivo,
        "perc_evidenza_decisiva": 100.0 * n_decisivo / n,
    }])
    print(f"      asimmetrico vince {n_vince}/{n}, decisivo {n_decisivo}/{n}, "
          f"mediana {d.median():.2f}")
    return fig_path, _salva_tabella(tabella, "04b_statistiche_delta_aic.csv")


# ---------------------------------------------------------------------------
# 4. RMSE dei tre modelli a confronto (Modulo 4)
# ---------------------------------------------------------------------------


# Boxplot del residuo dei tre candidati sugli stessi movimenti, in scala log.
def figura_4_rmse_per_modello(confronto: pd.DataFrame) -> tuple[Path, Path]:
    ordine = ["trapezoidal", "asymmetric_trapezoidal", "quintic"]
    serie = [confronto[confronto.profile_type == p].rmse_quanti.dropna() for p in ordine]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    bp = ax.boxplot(serie, tick_labels=[ETICHETTE[p] for p in ordine],
                    widths=0.55, patch_artist=True, medianprops=dict(color=COLORE_RIFERIMENTO, lw=1.6))
    for corpo, colore in zip(bp["boxes"], [COLORE_SERIE, COLORE_ACCENTO, COLORE_ATTENUATO]):
        corpo.set(facecolor=colore, alpha=0.55, edgecolor=COLORE_RIFERIMENTO, linewidth=0.9)
    ax.axhline(1.0, color=COLORE_RIFERIMENTO, ls=":", lw=1.2,
               label="1 quanto di encoder (pavimento della misura)")

    ax.set(ylabel="RMSE [quanti di encoder]", yscale="log",
           title=f"Residuo dei tre candidati sugli stessi {len(serie[0])} movimenti")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3, which="both", axis="y")
    fig_path = _salva_figura(fig, "04c_rmse_per_modello.png")

    tabella = pd.DataFrame([{
        "profile_type": p,
        "modello": ETICHETTE[p],
        "n_movimenti": len(s),
        "media": s.mean(),
        "mediana": s.median(),
        "dev_standard": s.std(),
        "minimo": s.min(),
        "q1": s.quantile(0.25),
        "q3": s.quantile(0.75),
        "massimo": s.max(),
    } for p, s in zip(ordine, serie)])
    return fig_path, _salva_tabella(tabella, "04c_rmse_per_modello.csv")


# ---------------------------------------------------------------------------
# 5. Simmetria di amax fittata fra i due versi di marcia (Modulo 4)
# ---------------------------------------------------------------------------


# Accoppia i due versi di ogni giunto e sessione sulla colonna indicata, tenendo solo
# le coppie identificabili in entrambi i versi. Restituisce anche quante ne scarta.
def coppie_amax_per_verso(fit_results: pd.DataFrame,
                          colonna: str = "amax_fit") -> tuple[pd.DataFrame, int]:
    dati = fit_results
    if "success" in dati.columns:
        dati = dati[dati["success"].astype(bool)]

    righe, scartate = [], 0
    for (sessione, giunto), gruppo in dati.groupby(["session", "joint_name"]):
        positivo = gruppo[gruppo["delta_q"] > 0]
        negativo = gruppo[gruppo["delta_q"] < 0]
        if len(positivo) != 1 or len(negativo) != 1:
            continue  # non e' una coppia andata/ritorno: non entra nel confronto
        p, n = positivo.iloc[0], negativo.iloc[0]
        if not (bool(p["amax_identifiable"]) and bool(n["amax_identifiable"])):
            scartate += 1
            continue
        righe.append({
            "session": sessione,
            "joint_name": giunto,
            "commanded_velocity": p["commanded_velocity"],
            "amax_positivo": float(p[colonna]),
            "amax_negativo": float(n[colonna]),
        })

    coppie = pd.DataFrame(righe, columns=["session", "joint_name", "commanded_velocity",
                                          "amax_positivo", "amax_negativo"])
    if len(coppie):
        coppie["asimmetria_perc"] = (100.0 * (coppie.amax_negativo - coppie.amax_positivo).abs()
                                     / coppie.amax_positivo)
    else:
        coppie["asimmetria_perc"] = pd.Series(dtype=float)
    return coppie, scartate


# Disegna amax fittata dei due versi una contro l'altra, piu' la tabella per giunto.
def figura_5_simmetria_amax_fittato(fit: pd.DataFrame) -> tuple[Path, Path]:
    coppie, scartate = coppie_amax_per_verso(fit)
    velocita = sorted(int(v) for v in coppie.commanded_velocity.unique())

    fig, ax = plt.subplots(figsize=(5.6, 5))
    # j4 in evidenza: e' il giunto che le diagnostiche isolano. Colore e forma del
    # marcatore insieme, cosi' l'informazione non passa dal solo colore.
    altri, j4 = coppie[coppie.joint_name != "j4"], coppie[coppie.joint_name == "j4"]
    ax.scatter(altri.amax_positivo, altri.amax_negativo, s=55, alpha=0.8, marker="o",
               color=COLORE_SERIE, zorder=2, label=f"altri giunti (n={len(altri)})")
    if len(j4):
        ax.scatter(j4.amax_positivo, j4.amax_negativo, s=70, alpha=0.9, marker="^",
                   color=COLORE_ACCENTO, edgecolor="white", linewidth=0.8, zorder=3,
                   label=f"j4 (n={len(j4)})")
    lim = [float(coppie.amax_positivo.min()) * 0.96, float(coppie.amax_positivo.max()) * 1.04]
    ax.plot(lim, lim, "--", lw=1.2, color=COLORE_RIFERIMENTO, label="simmetria perfetta")
    ax.set(xlabel=r"$a_{max}$ fittata, verso positivo [rad/s$^2$]",
           ylabel=r"$a_{max}$ fittata, verso negativo [rad/s$^2$]",
           title=f"Simmetria di $a_{{max}}$ fra andata e ritorno\n"
                 f"{len(coppie)} coppie identificabili su {len(coppie) + scartate}, "
                 f"velocita' {', '.join(str(v) for v in velocita)}",
           xlim=lim, ylim=lim)
    ax.set_aspect("equal")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig_path = _salva_figura(fig, "04d_simmetria_amax_fittato.png")

    per_giunto = coppie.groupby("joint_name").agg(
        n_coppie=("asimmetria_perc", "size"),
        mediana_perc=("asimmetria_perc", "median"),
        massima_perc=("asimmetria_perc", "max"),
    ).round(4).reset_index()
    print(f"      {len(coppie)} coppie confrontate, {scartate} scartate "
          f"(amax non identificabile in almeno un verso)")
    return fig_path, _salva_tabella(per_giunto, "04d_simmetria_amax_fittato.csv")


# ---------------------------------------------------------------------------
# 6. Sensibilita' del confronto alla finestra di analisi (Modulo 4)
# ---------------------------------------------------------------------------


# Rivaluta i tre candidati sulla finestra di moto e sull'intero segmento.
def tabella_6_sensibilita_finestra() -> Path:
    movimenti = load_movements_from_processed(
        ELABORATI / "preprocessing_params.csv", raw_root=DATI_GREZZI
    )

    righe = []
    for chiave, m in movimenti.items():
        t, q, q0, qf = m["t_data"], m["q_data"], m["q0"], m["qf"]
        t0_iniziale = estimate_t0(t, q)

        esiti = {}
        for nome in CANDIDATI_DEFAULT:
            x0 = build_initial_guess(m["vmax_est"], m["amax_est"], t0_iniziale, nome,
                                     delta_q=qf - q0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                esiti[nome] = fit_single_movement(t, q, q0, qf, x0, profile_type=nome)
        if not all(e.success for e in esiti.values()):
            continue

        # Finestra comune, ricavata dal riferimento come fa compare_models.
        rif = esiti["trapezoidal"]
        finestra = motion_window(t, rif.params, "trapezoidal", q0, qf)
        intero = np.ones(np.asarray(t).size, dtype=bool)

        for nome, esito in esiti.items():
            for etichetta, maschera in [("finestra", finestra), ("segmento", intero)]:
                s = compute_residuals_summary(t, q, q0, qf, esito.params, nome,
                                              motion_mask=maschera)
                aic, _ = compute_aic_bic(s.motion_residuals, n_params(nome))
                righe.append({"movimento": chiave, "profile_type": nome,
                              "ambito": etichetta, "aic": aic,
                              "n_campioni": int(s.n_samples)})

    dettaglio = pd.DataFrame(righe)
    tabelle = []
    for ambito in ("finestra", "segmento"):
        sotto = dettaglio[dettaglio.ambito == ambito]
        aic = sotto.pivot(index="movimento", columns="profile_type", values="aic")
        vincitori = aic.idxmin(axis=1).value_counts()
        delta = aic["asymmetric_trapezoidal"] - aic["trapezoidal"]
        for nome in CANDIDATI_DEFAULT:
            tabelle.append({
                "ambito": "finestra di moto" if ambito == "finestra" else "segmento intero",
                "profile_type": nome,
                "modello": ETICHETTE[nome],
                "n_vittorie": int(vincitori.get(nome, 0)),
                "n_movimenti": int(len(aic)),
                "campioni_mediani": int(sotto[sotto.profile_type == nome].n_campioni.median()),
                "delta_aic_mediano": delta.median() if nome == "asymmetric_trapezoidal" else np.nan,
            })

    tabella = pd.DataFrame(tabelle)
    for _, r in tabella[tabella.profile_type == "asymmetric_trapezoidal"].iterrows():
        print(f"      {r['ambito']:18s} n={r['campioni_mediani']:3d}  "
              f"vittorie asimmetrico {r['n_vittorie']:3d}  "
              f"delta_aic mediano {r['delta_aic_mediano']:+.1f}")
    return _salva_tabella(tabella, "04e_sensibilita_finestra.csv")


# ---------------------------------------------------------------------------
# 7. Accuratezza del simulatore per giunto e per velocita' (Modulo 5)
# ---------------------------------------------------------------------------


# Aggrega l'errore del simulatore per giunto e per velocita' comandata.
def tabella_7_accuratezza_per_giunto(accuratezza: pd.DataFrame) -> Path:
    def _riassumi(gruppo, raggruppamento, chiave):
        r = gruppo.rmse_quanti.agg(["size", "mean", "median", "std", "min", "max"])
        r = r.rename(columns={"size": "n_movimenti", "mean": "media", "median": "mediana",
                              "std": "dev_standard", "min": "minimo", "max": "massimo"})
        r = r.reset_index().rename(columns={chiave: "valore"})
        r.insert(0, "raggruppamento", raggruppamento)
        r["valore"] = r["valore"].astype(str)
        return r

    per_giunto = _riassumi(accuratezza.groupby("joint_name"), "giunto", "joint_name")
    per_velocita = _riassumi(accuratezza.groupby("commanded_velocity"), "velocita",
                             "commanded_velocity")
    tabella = pd.concat([per_giunto, per_velocita], ignore_index=True)

    peggiore = per_giunto.loc[per_giunto.mediana.idxmax()]
    migliore = per_giunto.loc[per_giunto.mediana.idxmin()]
    print(f"      per giunto: da {migliore.valore} ({migliore.mediana:.2f} quanti) "
          f"a {peggiore.valore} ({peggiore.mediana:.2f} quanti)")
    return _salva_tabella(tabella, "05b_accuratezza_per_giunto.csv")


# ---------------------------------------------------------------------------
# 8-10. Residui dei tre candidati a confronto (Modulo 4)
# ---------------------------------------------------------------------------


# Rifitta i tre candidati su tutti i movimenti e ne conserva i residui e le finestre.
def _residui_per_modello() -> dict:
    movimenti = load_movements_from_processed(
        ELABORATI / "preprocessing_params.csv", raw_root=DATI_GREZZI
    )

    dati = {}
    for chiave, m in movimenti.items():
        t, q, q0, qf = m["t_data"], m["q_data"], m["q0"], m["qf"]
        t0_iniziale = estimate_t0(t, q)

        esiti = {}
        for nome in CANDIDATI_DEFAULT:
            x0 = build_initial_guess(m["vmax_est"], m["amax_est"], t0_iniziale, nome,
                                     delta_q=qf - q0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                esiti[nome] = fit_single_movement(t, q, q0, qf, x0, profile_type=nome)
        if not all(e.success for e in esiti.values()):
            continue

        # Finestra e asse dei tempi comuni, dal riferimento trapezoidale.
        rif = esiti["trapezoidal"]
        maschera = motion_window(t, rif.params, "trapezoidal", q0, qf)
        t0_rif = float(rif.params["t0"])
        durata_rif = total_time("trapezoidal", q0, qf,
                                [rif.params[n] for n in ("vmax", "amax", "t0")])

        sommari = {}
        for nome, esito in esiti.items():
            sommari[nome] = compute_residuals_summary(t, q, q0, qf, esito.params, nome,
                                                      motion_mask=maschera)
        dati[chiave] = {
            "movimento": m, "esiti": esiti, "sommari": sommari,
            "maschera": maschera, "t0_rif": t0_rif, "durata_rif": durata_rif,
            "rampa_rif": float(rif.params["vmax"] / rif.params["amax"]) / durata_rif,
        }
    return dati


# Sovrappone il residuo mediano per fase dei tre modelli, sulla stessa base temporale.
def figura_8_residuo_per_fase_confronto(dati: dict | None = None) -> Path:
    dati = dati if dati is not None else _residui_per_modello()
    griglia = np.linspace(0.0, 1.0, 200)

    profili, frazioni = {n: [] for n in CANDIDATI_DEFAULT}, []
    for voce in dati.values():
        t = voce["movimento"]["t_data"]
        fase = (t - voce["t0_rif"]) / voce["durata_rif"]
        dentro = (fase >= 0) & (fase <= 1)
        if int(dentro.sum()) < 10:
            continue
        frazioni.append(voce["rampa_rif"])
        for nome in CANDIDATI_DEFAULT:
            residui = np.abs(voce["sommari"][nome].residuals)
            profili[nome].append(np.interp(griglia, fase[dentro], residui[dentro]))

    mediane = {n: np.median(np.vstack(v), axis=0) * 1e3 for n, v in profili.items()}
    frazione_rampa = float(np.median(frazioni))
    n_mov = len(frazioni)

    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.axvspan(frazione_rampa, 1 - frazione_rampa, color="gold", alpha=0.18,
               label="fase di crociera")
    for nome, colore, stile in [("trapezoidal", COLORE_SERIE, "-"),
                                ("asymmetric_trapezoidal", COLORE_ACCENTO, "-"),
                                ("quintic", COLORE_ATTENUATO, "--")]:
        ax.plot(griglia, mediane[nome], lw=2, color=colore, ls=stile, label=ETICHETTE[nome])
    ax.axhline(QUANTO * 1e3, color=COLORE_RIFERIMENTO, ls=":", lw=1.4,
               label="quantizzazione encoder")

    ax.set(xlabel="tempo normalizzato sul movimento  (0 = partenza, 1 = arrivo)",
           ylabel="|residuo| [mrad]", yscale="log",
           title=f"Residuo mediano per fase, {n_mov} movimenti "
                 f"(asse dei tempi del fit trapezoidale)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    percorso = _salva_figura(fig, "04f_residuo_per_fase_confronto.png")

    _verifica_raccordi(griglia, mediane, frazione_rampa)
    return percorso


# Stampa il |residuo| mediano per zona e la riduzione dell'asimmetrico sul simmetrico.
def _verifica_raccordi(griglia, mediane, frazione_rampa) -> None:
    zone = {
        "salita": griglia <= frazione_rampa,
        "crociera": (griglia > frazione_rampa) & (griglia < 1 - frazione_rampa),
        "discesa": griglia >= 1 - frazione_rampa,
    }
    print(f"      |residuo| mediano [mrad] per zona (rampa = {100*frazione_rampa:.0f}% "
          f"del movimento):")
    riduzioni = {}
    for zona, m in zone.items():
        sim = float(np.median(mediane["trapezoidal"][m]))
        asim = float(np.median(mediane["asymmetric_trapezoidal"][m]))
        quint = float(np.median(mediane["quintic"][m]))
        riduzioni[zona] = 100.0 * (sim - asim) / sim
        print(f"        {zona:9s} simmetrico {sim:6.2f}   asimmetrico {asim:6.2f}   "
              f"quintico {quint:6.2f}   riduzione {riduzioni[zona]:+5.1f}%")

    salita, discesa = riduzioni["salita"], riduzioni["discesa"]
    if salita <= 0 or discesa <= 0:
        print("      NOTA: l'asimmetrico non abbassa il residuo su entrambi i raccordi: "
              "il dato non sostiene la lettura attesa.")
    elif abs(salita - discesa) < 5.0:
        print(f"      NOTA: la riduzione e' simile sui due raccordi "
              f"({salita:.1f}% contro {discesa:.1f}%): i dati non mostrano un vantaggio "
              f"asimmetrico fra salita e discesa.")
    else:
        piu, meno = ("salita", "discesa") if salita > discesa else ("discesa", "salita")
        print(f"      L'asimmetrico abbassa il residuo su entrambi i raccordi, di piu' "
              f"in {piu} ({max(salita, discesa):.1f}%) che in {meno} "
              f"({min(salita, discesa):.1f}%).")


# Griglia 3x3 dei residui di un modello: andamento, distribuzione e Q-Q plot sui tre
# casi migliore, tipico e peggiore.
def _griglia_residui(dati: dict, profilo: str, nome_file: str) -> tuple[Path, float]:
    rmse = {k: v["sommari"][profilo].rmse / QUANTO for k, v in dati.items()}
    serie = pd.Series(rmse)
    mediano = serie.index[(serie - serie.median()).abs().argmin()]
    scelti = [("migliore", serie.idxmin()), ("tipico", mediano),
              ("peggiore", serie.idxmax())]

    fig, axes = plt.subplots(3, 3, figsize=(14, 9))
    massimo = 0.0
    for col, (etichetta, chiave) in enumerate(scelti):
        voce = dati[chiave]
        s = voce["sommari"][profilo]
        sessione, giunto = voce["movimento"]["session"], voce["movimento"]["joint_name"]
        titolo = f"{etichetta}\n{giunto} · {sessione} · {rmse[chiave]:.2f} quanti"
        massimo = max(massimo, float(np.max(np.abs(s.residuals))) * 1e3)

        plot_residuals_vs_time(voce["movimento"]["t_data"], s.residuals, titolo,
                               ax=axes[0, col], mask=s.motion_mask)
        plot_residuals_histogram(s.residuals, "", ax=axes[1, col], mask=s.motion_mask)
        plot_residuals_qq(s.residuals, "", ax=axes[2, col], mask=s.motion_mask)

    # Andamento temporale sulla stessa scala nelle tre colonne, cosi' si legge il
    # salto di ampiezza fra i casi; distribuzione e Q-Q si adattano a ciascun caso,
    # altrimenti nei casi piccoli la forma sarebbe schiacciata su un picco.
    for col in range(3):
        axes[0, col].set_ylim(-massimo * 1.08, massimo * 1.08)

    fig.suptitle(f"Residui del profilo {ETICHETTE[profilo]}: andamento temporale, "
                 f"distribuzione e Q-Q plot (convenzione: reale - modello)",
                 fontweight="bold")
    return _salva_figura(fig, nome_file), massimo


# Residui del trapezoidale asimmetrico.
def figura_9_residui_asimmetrico(dati: dict | None = None) -> Path:
    dati = dati if dati is not None else _residui_per_modello()
    percorso, massimo = _griglia_residui(dati, "asymmetric_trapezoidal",
                                         "04g_residui_confronto_asimmetrico.png")
    print(f"      andamento temporale a ±{massimo:.1f} mrad, distribuzione e Q-Q "
          f"adattati a ciascun caso")
    return percorso


# Residui del quintico.
def figura_10_residui_quintico(dati: dict | None = None) -> Path:
    dati = dati if dati is not None else _residui_per_modello()
    percorso, massimo = _griglia_residui(dati, "quintic",
                                         "04h_residui_confronto_quintico.png")
    print(f"      andamento temporale a ±{massimo:.0f} mrad: oltre un ordine di "
          f"grandezza sopra 04g, le scale delle tre figure non sono confrontabili")
    return percorso


# ---------------------------------------------------------------------------
# 11. Il giunto j4 e' anomalo anche sul modello vincente? (Modulo 4)
# ---------------------------------------------------------------------------


# Rifa' le diagnostiche su j4 leggendo i fit asimmetrici gia' salvati, separatamente per
# a_acc e a_dec, e produce le due tabelle 04i.
def tabella_11_coerenza_j4_asimmetrico() -> tuple[Path, Path]:
    percorso_fit = ELABORATI / "fit_results_asymmetric.csv"
    fit = pd.read_csv(percorso_fit)

    # I residui vengono dal modello intero, non dal singolo parametro: si calcolano
    # una volta sola e valgono per entrambe le tabelle.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        validazione = validate_all_movements(
            percorso_fit, ELABORATI / "preprocessing_params.csv", raw_root=DATI_GREZZI
        )

    percorsi = []
    for colonna, etichetta, nome_file in [
        ("a_acc_fit", "a_acc", "04i_sommario_per_giunto_a_acc.csv"),
        ("a_dec_fit", "a_dec", "04i_sommario_per_giunto_a_dec.csv"),
    ]:
        sommario = summarize_by_joint(validazione, fit.assign(amax_fit=fit[colonna]))
        percorsi.append(_salva_tabella(sommario, nome_file))
        _verdetto_j4(sommario, etichetta)
    return tuple(percorsi)


# Risponde si'/no, per un parametro alla volta: j4 resta l'anomalia sull'asimmetrico?
def _verdetto_j4(sommario: pd.DataFrame, etichetta: str) -> None:
    j4 = sommario[sommario.joint_name == "j4"].iloc[0]
    altri = sommario[sommario.joint_name != "j4"]

    cv_j4 = float(j4.amax_cv_filtrato)
    cv_max_altri = float(altri.amax_cv_filtrato.max())
    incoerenti = sommario[~sommario.amax_coerente.astype(bool)].joint_name.tolist()
    fuori_j4 = int(j4.n_fuori_norma)
    fuori_altri = int(altri.n_fuori_norma.sum())

    print(f"      {etichetta}: CV filtrato j4 {100*cv_j4:.1f}%, massimo fra gli altri "
          f"{100*cv_max_altri:.1f}%; incoerenti {incoerenti or 'nessuno'}; "
          f"fuori norma j4 {fuori_j4} contro {fuori_altri}")

    resta = (incoerenti == ["j4"]) or (cv_j4 > 2 * cv_max_altri and fuori_j4 > fuori_altri)
    esito = "SI, resta l'anomalia" if resta else "NO, non e' piu' isolato"
    print(f"      {etichetta}: {esito}")


# ---------------------------------------------------------------------------
# 12-13. Le figure dell'asimmetrico, su a_acc e a_dec separati
# ---------------------------------------------------------------------------


# Legge i fit del trapezoidale asimmetrico.
def _fit_asimmetrico() -> pd.DataFrame:
    return pd.read_csv(ELABORATI / "fit_results_asymmetric.csv")


# Griglia 2x3 di un parametro contro la velocita' comandata, un pannello per giunto.
# Stessa struttura e stessa scala di 04_amax_per_giunto.png.
def _griglia_per_giunto(fit: pd.DataFrame, colonna: str, simbolo: str,
                        nome_file: str) -> Path:
    dati = fit.assign(amax_fit=fit[colonna])

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=True)
    for ax, giunto in zip(axes.flatten(), sorted(dati.joint_name.unique())):
        plot_amax_across_speeds(dati, giunto, ax=ax)
        ax.set_ylim(2, 200)
        ax.set_ylabel(f"${simbolo}$ [rad/s$^2$]")
    for ax in axes[0, :]:
        ax.set_xlabel("")
    for ax in axes[:, 1:].flatten():
        ax.set_ylabel("")

    fig.suptitle(f"${simbolo}$ alle quattro velocita' comandate, per giunto "
                 f"(modello asimmetrico)", fontweight="bold")
    return _salva_figura(fig, nome_file)


# Scatter dei due versi di marcia su un parametro, piu' la tabella per giunto.
# Stessa struttura di 04d_simmetria_amax_fittato.png.
def _simmetria_per_verso(fit: pd.DataFrame, colonna: str, simbolo: str,
                         nome_figura: str, nome_tabella: str) -> tuple[Path, Path]:
    coppie, scartate = coppie_amax_per_verso(fit, colonna=colonna)
    velocita = sorted(int(v) for v in coppie.commanded_velocity.unique())

    fig, ax = plt.subplots(figsize=(5.6, 5))
    # j4 in evidenza: e' il giunto che le diagnostiche isolano. Colore e forma del
    # marcatore insieme, cosi' l'informazione non passa dal solo colore.
    altri, j4 = coppie[coppie.joint_name != "j4"], coppie[coppie.joint_name == "j4"]
    ax.scatter(altri.amax_positivo, altri.amax_negativo, s=55, alpha=0.8, marker="o",
               color=COLORE_SERIE, zorder=2, label=f"altri giunti (n={len(altri)})")
    if len(j4):
        ax.scatter(j4.amax_positivo, j4.amax_negativo, s=70, alpha=0.9, marker="^",
                   color=COLORE_ACCENTO, edgecolor="white", linewidth=0.8, zorder=3,
                   label=f"j4 (n={len(j4)})")
    lim = [float(coppie.amax_positivo.min()) * 0.96, float(coppie.amax_positivo.max()) * 1.04]
    ax.plot(lim, lim, "--", lw=1.2, color=COLORE_RIFERIMENTO, label="simmetria perfetta")
    ax.set(xlabel=f"${simbolo}$ fittata, verso positivo [rad/s$^2$]",
           ylabel=f"${simbolo}$ fittata, verso negativo [rad/s$^2$]",
           title=f"Simmetria di ${simbolo}$ fra andata e ritorno\n"
                 f"{len(coppie)} coppie identificabili su {len(coppie) + scartate}, "
                 f"velocita' {', '.join(str(v) for v in velocita)}",
           xlim=lim, ylim=lim)
    ax.set_aspect("equal")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig_path = _salva_figura(fig, nome_figura)

    per_giunto = coppie.groupby("joint_name").agg(
        n_coppie=("asimmetria_perc", "size"),
        mediana_perc=("asimmetria_perc", "median"),
        massima_perc=("asimmetria_perc", "max"),
    ).round(4).reset_index()
    j4 = per_giunto[per_giunto.joint_name == "j4"]
    peggiore = per_giunto.loc[per_giunto.mediana_perc.idxmax()].joint_name
    print(f"      {simbolo}: {len(coppie)} coppie, {scartate} scartate; asimmetria "
          f"mediana peggiore su {peggiore}"
          + (f", j4 al {float(j4.iloc[0].mediana_perc):.2f}%" if len(j4) else ""))
    return fig_path, _salva_tabella(per_giunto, nome_tabella)


# Le due griglie per giunto, su a_acc e su a_dec.
def figura_12_accelerazioni_per_giunto(fit: pd.DataFrame | None = None) -> tuple[Path, Path]:
    fit = fit if fit is not None else _fit_asimmetrico()
    return (_griglia_per_giunto(fit, "a_acc_fit", "a_{acc}",
                                "04k_a_acc_per_giunto_asimmetrico.png"),
            _griglia_per_giunto(fit, "a_dec_fit", "a_{dec}",
                                "04k_a_dec_per_giunto_asimmetrico.png"))


# I due scatter di simmetria, su a_acc e su a_dec.
def figura_13_simmetria_accelerazioni(fit: pd.DataFrame | None = None) -> tuple[Path, ...]:
    fit = fit if fit is not None else _fit_asimmetrico()
    acc = _simmetria_per_verso(fit, "a_acc_fit", "a_{acc}",
                               "04l_simmetria_a_acc_asimmetrico.png",
                               "04j_simmetria_a_acc_asimmetrico.csv")
    dec = _simmetria_per_verso(fit, "a_dec_fit", "a_{dec}",
                               "04l_simmetria_a_dec_asimmetrico.png",
                               "04j_simmetria_a_dec_asimmetrico.csv")
    return acc + dec


# Genera in ordine tutte le figure e le tabelle aggiuntive.
def main() -> int:
    FIGURE.mkdir(parents=True, exist_ok=True)
    TABELLE.mkdir(parents=True, exist_ok=True)

    attesi = ["model_comparison.csv", "fit_results.csv", "simulator_accuracy.csv",
              "preprocessing_params.csv", "fit_results_asymmetric.csv"]
    mancanti = [n for n in attesi if not (ELABORATI / n).exists()]
    if mancanti:
        print(f"ERRORE: mancano {', '.join(mancanti)} in {ELABORATI}.\n"
              f"       Eseguire prima: python3 scripts/run_identification.py", file=sys.stderr)
        return 1

    confronto = pd.read_csv(ELABORATI / "model_comparison.csv")
    fit = pd.read_csv(ELABORATI / "fit_results.csv")
    accuratezza = pd.read_csv(ELABORATI / "simulator_accuracy.csv")

    print("[1/13] Modulo 2 - confronto fra i tre profili teorici")
    figura_1_confronto_profili()

    print("[2/13] Modulo 2 - soglia di degenerazione trapezoidale/triangolare")
    figura_2_soglia_degenerazione()

    print("[3/13] Modulo 4 - distribuzione di delta-AIC")
    figura_3_distribuzione_delta_aic(confronto)

    print("[4/13] Modulo 4 - RMSE dei tre modelli a confronto")
    figura_4_rmse_per_modello(confronto)

    print("[5/13] Modulo 4 - simmetria di amax fittata fra i due versi")
    figura_5_simmetria_amax_fittato(fit)

    print("[6/13] Modulo 4 - sensibilita' alla finestra di analisi")
    tabella_6_sensibilita_finestra()

    print("[7/13] Modulo 5 - accuratezza del simulatore per giunto")
    tabella_7_accuratezza_per_giunto(accuratezza)

    # Calcolati una volta sola e passati alle tre figure che li usano.
    residui = _residui_per_modello()

    print("[8/13] Modulo 4 - residuo per fase, i tre modelli a confronto")
    figura_8_residuo_per_fase_confronto(residui)

    print("[9/13] Modulo 4 - residui del trapezoidale asimmetrico")
    figura_9_residui_asimmetrico(residui)

    print("[10/13] Modulo 4 - residui del quintico")
    figura_10_residui_quintico(residui)

    print("[11/13] Modulo 4 - coerenza per giunto su a_acc e a_dec")
    tabella_11_coerenza_j4_asimmetrico()

    fit_asim = _fit_asimmetrico()

    print("[12/13] Modulo 4 - a_acc e a_dec per giunto, modello asimmetrico")
    figura_12_accelerazioni_per_giunto(fit_asim)

    print("[13/13] Modulo 4 - simmetria fra i versi su a_acc e a_dec")
    figura_13_simmetria_accelerazioni(fit_asim)

    print("Fatto.")
    print(f"      figure in   {FIGURE.relative_to(RADICE)}/")
    print(f"      tabelle in  {TABELLE.relative_to(RADICE)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
