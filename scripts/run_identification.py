#!/usr/bin/env python3
"""Esegue l'intera identificazione da capo e rigenera tutti i risultati.

    python3 scripts/run_identification.py

Cinque passi in ordine: preprocessing, fitting, confronto fra modelli,
validazione dei residui, accuratezza del simulatore. Produce i CSV in
``data/processed/`` e le tabelle in ``results/tables/``. Si lancia dalla
cartella del progetto.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RADICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RADICE))

from src.fitting import (  # noqa: E402
    build_initial_guess, estimate_t0, fit_all_movements, fit_single_movement,
    load_movements_from_processed, save_fit_results,
)
from src.preprocessing import analyze_dataset, save_results  # noqa: E402
from src.simulator import SimulatorParams, validate_simulator  # noqa: E402
from src.validation import (  # noqa: E402
    aggregate_model_comparison, compare_models, summarize_by_joint,
    validate_all_movements,
)

DATI_GREZZI = RADICE / "data/raw/dati_tesi"
ELABORATI = RADICE / "data/processed"
TABELLE = RADICE / "results/tables"


QUANTO = 1.4e-3  # quantizzazione dell'encoder [rad]


# Regressione lineare di una grandezza sulla velocita' comandata: coefficiente e R^2.
def _regressione(dati: pd.DataFrame, colonna: str, ascissa=None):
    x = dati["commanded_velocity"] if ascissa is None else ascissa
    coef = np.polyfit(x, dati[colonna], 1)
    residuo = dati[colonna] - np.polyval(coef, x)
    return coef, 1.0 - np.var(residuo) / np.var(dati[colonna])


# Riepilogo dell'identificazione con i tre modelli insieme, in formato lungo: i tre
# hanno parametri diversi, e una tabella larga avrebbe celle vuote.
def riepilogo_tre_modelli(fit_per_modello: dict) -> Path:
    tr, asi, qui = (fit_per_modello["trapezoidal"],
                    fit_per_modello["asymmetric_trapezoidal"],
                    fit_per_modello["quintic"])
    righe = []

    def _residuo(nome, d):
        return {"modello": nome, "grandezza": "residuo del fit",
                "relazione col comando": "-",
                "valore": f"{d.rmse.median()*1e3:.2f} mrad (mediano)",
                "qualita'": f"{d.rmse.median()/QUANTO:.2f} quanti di encoder", "n": len(d)}

    # --- trapezoidale -------------------------------------------------------
    c_tr, r2_tr = _regressione(tr, "vmax_fit")
    id_tr = tr[tr.amax_identifiable]
    righe += [
        {"modello": "trapezoidale", "grandezza": "vmax", "relazione col comando": "lineare",
         "valore": f"{c_tr[0]:.5f} * cmd {c_tr[1]:+.5f} rad/s",
         "qualita'": f"R2 = {r2_tr:.5f}", "n": len(tr)},
        {"modello": "trapezoidale", "grandezza": "amax", "relazione col comando": "costante",
         "valore": f"{id_tr.amax_fit.median():.3f} rad/s^2 (mediana)",
         "qualita'": f"dev.std {id_tr.amax_fit.std():.3f} "
                     f"({100*id_tr.amax_fit.std()/id_tr.amax_fit.mean():.1f}%)",
         "n": len(id_tr)},
        _residuo("trapezoidale", tr),
    ]

    # --- asimmetrico --------------------------------------------------------
    c_as, r2_as = _regressione(asi, "vmax_fit")
    # Un solo sottoinsieme per entrambe le accelerazioni, quello con la rampa di salita
    # risolta: cosi' il rapporto e' calcolato su un'unica popolazione.
    ident = asi[asi.amax_identifiable]
    m_acc, m_dec = ident.a_acc_fit.median(), ident.a_dec_fit.median()
    righe += [
        {"modello": "asimmetrico", "grandezza": "vmax", "relazione col comando": "lineare",
         "valore": f"{c_as[0]:.5f} * cmd {c_as[1]:+.5f} rad/s",
         "qualita'": f"R2 = {r2_as:.5f}", "n": len(asi)},
        {"modello": "asimmetrico", "grandezza": "a_acc", "relazione col comando": "costante",
         "valore": f"{m_acc:.3f} rad/s^2 (mediana)",
         "qualita'": f"dev.std {ident.a_acc_fit.std():.3f} "
                     f"({100*ident.a_acc_fit.std()/ident.a_acc_fit.mean():.1f}%)",
         "n": len(ident)},
        {"modello": "asimmetrico", "grandezza": "a_dec", "relazione col comando": "costante",
         "valore": f"{m_dec:.3f} rad/s^2 (mediana)",
         "qualita'": f"dev.std {ident.a_dec_fit.std():.3f} "
                     f"({100*ident.a_dec_fit.std()/ident.a_dec_fit.mean():.1f}%)",
         "n": len(ident)},
        {"modello": "asimmetrico", "grandezza": "rapporto a_acc/a_dec",
         "relazione col comando": "costante",
         "valore": f"{m_acc:.3f} / {m_dec:.3f} = {m_acc/m_dec:.4f}",
         "qualita'": f"entrambe sui movimenti con la rampa di salita risolta",
         "n": len(ident)},
        _residuo("asimmetrico", asi),
    ]

    # --- quintico -----------------------------------------------------------
    _, r2_lin = _regressione(qui, "t_move_fit")
    c_inv, r2_inv = _regressione(qui, "t_move_fit", ascissa=1.0 / qui.commanded_velocity)
    righe += [
        {"modello": "quintico", "grandezza": "t_move",
         "relazione col comando": "iperbolica (lineare in 1/cmd)",
         "valore": f"{c_inv[0]:.3f} / cmd {c_inv[1]:+.3f} s",
         "qualita'": f"R2 = {r2_inv:.5f} contro {r2_lin:.5f} in scala lineare",
         "n": len(qui)},
        _residuo("quintico", qui),
    ]

    tabella = pd.DataFrame(righe)
    tabella.to_csv(TABELLE / "03b_riepilogo_identificazione_tre_modelli.csv",
                   index=False, float_format="%.6g")

    scarto_vmax = 100 * abs(c_as[0] - c_tr[0]) / c_tr[0]
    print(f"      vmax: coefficiente {c_tr[0]:.6f} (trapezoidale) contro {c_as[0]:.6f} "
          f"(asimmetrico), scarto {scarto_vmax:.3f}%")
    if scarto_vmax > 1.0:
        print("      ATTENZIONE: vmax e' lo stesso parametro nei due modelli ma i due "
              "fit divergono di oltre l'1%.")
    scarto = max(abs(m_acc - SimulatorParams.a_acc) / SimulatorParams.a_acc,
                 abs(m_dec - SimulatorParams.a_dec) / SimulatorParams.a_dec)
    print(f"      a_acc {m_acc:.3f} e a_dec {m_dec:.3f} su {len(ident)} movimenti, "
          f"contro {SimulatorParams.a_acc} e {SimulatorParams.a_dec} di SimulatorParams "
          f"(scarto {100*scarto:.1f}%)")
    if scarto > 0.01:
        print("      ATTENZIONE: SimulatorParams non e' riproducibile dai fit: "
              "i parametri del simulatore andrebbero aggiornati.")
    return TABELLE / "03b_riepilogo_identificazione_tre_modelli.csv"


# Esegue in ordine i cinque passi dell'identificazione e salva CSV e tabelle
# riassuntive.
def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    ELABORATI.mkdir(parents=True, exist_ok=True)
    TABELLE.mkdir(parents=True, exist_ok=True)

    if not DATI_GREZZI.exists():
        print(f"ERRORE: dati grezzi non trovati in {DATI_GREZZI}", file=sys.stderr)
        return 1

    # --- 1. Preprocessing ---------------------------------------------------
    print("[1/6] Modulo 1 - preprocessing di tutte le sessioni")
    _, params = analyze_dataset(DATI_GREZZI, verbose=True)
    save_results(params, ELABORATI / "preprocessing_params.csv")
    print(f"      {len(params)} movimenti analizzati")

    riassunto = params.groupby("commanded_velocity").agg(
        n=("vmax_est", "size"),
        vmax=("vmax_est", "mean"), vmax_std=("vmax_est", "std"),
        amax=("amax_est", "mean"), amax_std=("amax_est", "std"),
        regime_s=("steady_state_duration", "mean"),
    ).reset_index()
    riassunto.to_csv(TABELLE / "01_riassunto_per_velocita.csv", index=False, float_format="%.6g")

    # --- 2. Fitting ---------------------------------------------------------
    print("[2/6] Modulo 3 - fitting sulla posizione grezza")
    movimenti = load_movements_from_processed(
        ELABORATI / "preprocessing_params.csv", raw_root=DATI_GREZZI
    )
    fit_res, fit = fit_all_movements(movimenti, output_path=ELABORATI / "fit_results.csv")
    print(f"      convergenza {int(fit.success.sum())}/{len(fit)}, "
          f"RMSE mediano {fit.rmse.median():.5f} rad")

    identificabili = fit[fit.amax_identifiable]
    fit.groupby("commanded_velocity").agg(
        n=("vmax_fit", "size"),
        vmax=("vmax_fit", "mean"), vmax_std=("vmax_fit", "std"),
        amax=("amax_fit", "mean"), amax_std=("amax_fit", "std"),
        t0=("t0_fit", "mean"),
        # Mediana: a comando 25 la rampa copre pochi campioni e la
        # distribuzione ha una coda che sposterebbe la media.
        rampa_campioni=("ramp_samples", "median"),
    ).reset_index().to_csv(TABELLE / "03_parametri_identificati.csv",
                           index=False, float_format="%.6g")
    pd.DataFrame([{
        "n_movimenti": len(fit),
        "n_convergenti": int(fit.success.sum()),
        "rmse_mediano": fit.rmse.median(),
        "n_amax_identificabili": len(identificabili),
        "amax_medio": identificabili.amax_fit.mean(),
        "amax_std": identificabili.amax_fit.std(),
    }]).to_csv(TABELLE / "03_riepilogo_identificazione.csv", index=False, float_format="%.6g")
    print(f"      amax su {len(identificabili)} identificabili: "
          f"{identificabili.amax_fit.mean():.2f} +/- {identificabili.amax_fit.std():.2f} rad/s^2")

    # Gli altri due candidati, salvati nello stesso formato del trapezoidale.
    fit_per_modello = {"trapezoidal": fit}
    for nome, uscita in [("asymmetric_trapezoidal", "fit_results_asymmetric.csv"),
                         ("quintic", "fit_results_quintic.csv")]:
        esiti = []
        for mov in movimenti.values():
            t, q, q0, qf = mov["t_data"], mov["q_data"], mov["q0"], mov["qf"]
            x0 = build_initial_guess(mov["vmax_est"], mov["amax_est"],
                                     estimate_t0(t, q), nome, delta_q=qf - q0)
            esiti.append(fit_single_movement(
                t, q, q0, qf, x0, profile_type=nome,
                joint_name=mov["joint_name"], segment_index=mov["segment_index"],
                session=mov["session"], source=mov["source"],
                commanded_velocity=mov["commanded_velocity"],
            ))
        tabella = pd.DataFrame([e.to_record() for e in esiti])
        fit_per_modello[nome] = tabella
        save_fit_results(tabella, ELABORATI / uscita)
        print(f"      {nome}: convergenza {int(tabella.success.sum())}/{len(tabella)}, "
              f"RMSE mediano {tabella.rmse.median():.5f} rad -> {uscita}")

    riepilogo_tre_modelli(fit_per_modello)

    # --- 3. Confronto fra modelli ------------------------------------------
    print("[3/6] Modulo 4 - confronto fra modelli candidati")
    righe = []
    for chiave, mov in movimenti.items():
        tabella = compare_models(
            mov["t_data"], mov["q_data"], mov["q0"], mov["qf"],
            mov["vmax_est"], mov["amax_est"],
        )
        if tabella.empty:
            continue
        tabella = tabella.copy()
        tabella["movimento"] = chiave
        tabella["joint_name"] = mov["joint_name"]
        tabella["commanded_velocity"] = mov["commanded_velocity"]
        righe.append(tabella)

    per_movimento = pd.concat(righe, ignore_index=True)
    per_movimento.to_csv(ELABORATI / "model_comparison.csv", index=False, float_format="%.6g")
    confronto = aggregate_model_comparison(per_movimento)
    confronto.to_csv(TABELLE / "04_confronto_modelli.csv", index=False, float_format="%.6g")
    vincitore = confronto.iloc[0]
    print(f"      vince '{vincitore.profile_type}' su {int(vincitore.n_vittorie)}/"
          f"{int(vincitore.n_movimenti)} movimenti")
    print(f"      confronto riga per riga: {len(per_movimento)} righe "
          f"({per_movimento.movimento.nunique()} movimenti x "
          f"{per_movimento.profile_type.nunique()} modelli) in model_comparison.csv")

    # --- 4. Validazione dei residui ----------------------------------------
    print("[4/6] Modulo 4 - validazione dei residui")
    validazione = validate_all_movements(
        ELABORATI / "fit_results.csv", ELABORATI / "preprocessing_params.csv",
        raw_root=DATI_GREZZI, output_path=ELABORATI / "validation_results.csv",
    )
    sommario = summarize_by_joint(validazione, fit)
    sommario.to_csv(TABELLE / "04_sommario_per_giunto.csv", index=False, float_format="%.6g")
    print(f"      RMSE mediano {validazione.rmse_quanti.median():.2f} quanti di encoder, "
          f"{int(validazione.fuori_norma.sum())} movimenti fuori norma")

    # --- 5. Accuratezza del simulatore -------------------------------------
    print("[5/6] Modulo 5 - accuratezza del simulatore")
    accuratezza = validate_simulator(DATI_GREZZI)
    accuratezza.to_csv(ELABORATI / "simulator_accuracy.csv", index=False, float_format="%.6g")
    accuratezza.groupby("commanded_velocity").rmse_quanti.agg(
        ["median", "mean", "max", "count"]
    ).reset_index().to_csv(TABELLE / "05_accuratezza_simulatore.csv",
                           index=False, float_format="%.6g")
    print(f"      errore mediano {accuratezza.rmse_quanti.median():.2f} quanti "
          f"= {np.rad2deg(accuratezza.rmse.median()):.3f} gradi")

    # --- 6. Fatto -----------------------------------------------------------
    print("[6/6] Fatto.")
    print(f"      CSV in       {ELABORATI.relative_to(RADICE)}/")
    print(f"      tabelle in   {TABELLE.relative_to(RADICE)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
