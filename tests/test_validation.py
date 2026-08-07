"""Test della validazione e del confronto fra modelli (Modulo 4)."""

import numpy as np
import pytest

from src.trajectory_models import generate_trapezoidal_profile
from src.validation import (
    aggregate_model_comparison, compare_models, compute_aic_bic,
    compute_residuals_summary, motion_window,
)

FS = 25.0
VMAX, AMAX, T0 = 1.5, 6.0, 1.0


@pytest.fixture
def ground_truth():
    t, q, info = generate_trapezoidal_profile(
        q0=0.0, delta_q=4.0, vmax=VMAX, amax=AMAX, fs=FS, t_pre=T0, t_post=1.0
    )
    return t, q, info


class TestFinestraDiMoto:
    def test_esclude_la_quiete(self, ground_truth):
        t, _, info = ground_truth
        m = motion_window(t, {"vmax": VMAX, "amax": AMAX, "t0": T0}, "trapezoidal", 0.0, 4.0)
        assert m.sum() < t.size          # scarta qualcosa
        assert not m[0] and not m[-1]    # e proprio le code
        assert t[m].min() >= T0 - 1e-9
        assert t[m].max() <= T0 + info["t_move"] + 1e-9

    def test_accetta_il_formato_del_csv(self, ground_truth):
        t, _, _ = ground_truth
        con_suffisso = motion_window(
            t, {"vmax_fit": VMAX, "amax_fit": AMAX, "t0_fit": T0}, "trapezoidal", 0.0, 4.0
        )
        senza = motion_window(t, [VMAX, AMAX, T0], "trapezoidal", 0.0, 4.0)
        assert np.array_equal(con_suffisso, senza)

    def test_parametro_mancante(self, ground_truth):
        t, _, _ = ground_truth
        with pytest.raises(KeyError, match="amax"):
            motion_window(t, {"vmax": VMAX, "t0": T0}, "trapezoidal", 0.0, 4.0)


class TestResidui:
    def test_modello_esatto_da_residuo_nullo(self, ground_truth):
        t, q, _ = ground_truth
        s = compute_residuals_summary(t, q, 0.0, 4.0, [VMAX, AMAX, T0], "trapezoidal")
        assert s.rmse < 1e-12
        assert s.n_samples < s.n_samples_full

    def test_la_finestra_cambia_l_rmse(self, ground_truth):
        # Fuori dal moto il residuo e' identicamente nullo: includerlo diluisce.
        t, q, _ = ground_truth
        q_rumore = q + np.random.default_rng(0).normal(0, 1e-3, q.size) * (
            motion_window(t, [VMAX, AMAX, T0], "trapezoidal", 0.0, 4.0)
        )
        s = compute_residuals_summary(t, q_rumore, 0.0, 4.0, [VMAX, AMAX, T0], "trapezoidal")
        assert s.rmse > s.rmse_full
        assert 0 < s.motion_fraction < 1

    def test_spostamento_nullo(self, ground_truth):
        t, q, _ = ground_truth
        with pytest.raises(ValueError, match="nullo"):
            compute_residuals_summary(t, q, 1.0, 1.0, [VMAX, AMAX, T0], "trapezoidal")


class TestAic:
    def test_premia_il_residuo_minore_a_parita_di_parametri(self):
        rng = np.random.default_rng(0)
        piccolo = rng.normal(0, 1e-3, 100)
        grande = rng.normal(0, 1e-2, 100)
        assert compute_aic_bic(piccolo, 3)[0] < compute_aic_bic(grande, 3)[0]

    def test_penalizza_i_parametri_a_parita_di_residuo(self):
        r = np.random.default_rng(0).normal(0, 1e-3, 100)
        assert compute_aic_bic(r, 4)[0] > compute_aic_bic(r, 3)[0]
        # BIC penalizza di piu' di AIC quando n > e^2.
        assert compute_aic_bic(r, 4)[1] - compute_aic_bic(r, 3)[1] > 2.0

    def test_fit_esatto(self):
        assert compute_aic_bic(np.zeros(50), 3) == (-np.inf, -np.inf)


class TestConfrontoFraModelli:
    def test_vince_il_modello_che_ha_generato_i_dati(self, ground_truth):
        t, q, _ = ground_truth
        tab = compare_models(t, q, 0.0, 4.0, VMAX, AMAX)
        assert not tab.empty
        assert tab.success.all()
        vincitore = tab.loc[tab.aic.idxmin(), "profile_type"]
        assert vincitore in ("trapezoidal", "asymmetric_trapezoidal")
        # Il quintico non ha tratto a velocita' costante: deve perdere nettamente.
        quintico = tab.loc[tab.profile_type == "quintic", "delta_aic"].iloc[0]
        assert quintico > 10

    def test_finestra_comune_a_tutti(self, ground_truth):
        t, q, _ = ground_truth
        tab = compare_models(t, q, 0.0, 4.0, VMAX, AMAX)
        # E' cio' che rende gli AIC confrontabili: omettendo le costanti
        # additive, valgono solo a parita' di campioni.
        assert tab.n_finestra.nunique() == 1

    def test_riferimento_fuori_dai_candidati(self, ground_truth):
        t, q, _ = ground_truth
        with pytest.raises(ValueError, match="riferimento"):
            compare_models(t, q, 0.0, 4.0, VMAX, AMAX, reference="quintic",
                           candidates=("trapezoidal",))

    def test_aggregazione_conta_le_vittorie(self, ground_truth):
        import pandas as pd

        t, q, _ = ground_truth
        tabelle = []
        for k in range(3):
            tab = compare_models(t, q, 0.0, 4.0, VMAX, AMAX).copy()
            tab["movimento"] = f"mov{k}"
            tabelle.append(tab)

        agg = aggregate_model_comparison(pd.concat(tabelle, ignore_index=True))
        assert agg.n_vittorie.sum() == 3
        assert agg.iloc[0].n_vittorie >= agg.iloc[-1].n_vittorie  # ordinata
