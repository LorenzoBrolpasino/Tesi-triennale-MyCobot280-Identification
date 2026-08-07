"""Test dell'identificazione ai minimi quadrati (Modulo 3)."""

import numpy as np
import pytest

from src.fitting import (
    FitConfig, build_initial_guess, estimate_t0, fit_all_movements, fit_single_movement,
)
from src.trajectory_models import generate_trapezoidal_profile, trapezoidal_position

FS = 25.0
VMAX, AMAX, T0 = 1.5, 6.0, 1.0


@pytest.fixture
def ground_truth():
    """Trapezoidale a parametri noti, con la quiete iniziale dei dati reali."""
    t, q, info = generate_trapezoidal_profile(
        q0=0.0, delta_q=4.0, vmax=VMAX, amax=AMAX, fs=FS, t_pre=T0, t_post=1.0
    )
    return t, q, info


class TestRecuperoDeiParametri:
    def test_da_guess_sbagliato_del_30_percento(self, ground_truth):
        t, q, _ = ground_truth
        x0 = build_initial_guess(VMAX * 1.3, AMAX * 0.7, T0 + 0.1)
        res = fit_single_movement(t, q, q[0], q[-1], x0)

        assert res.success
        assert res.vmax_fit == pytest.approx(VMAX, rel=1e-3)
        assert res.amax_fit == pytest.approx(AMAX, rel=1e-2)
        assert res.t0_fit == pytest.approx(T0, abs=1e-3)
        assert res.rmse < 1e-6

    def test_con_rumore_di_misura(self, ground_truth):
        t, _, _ = ground_truth
        _, q, _ = generate_trapezoidal_profile(
            0.0, 4.0, VMAX, AMAX, FS, t_pre=T0, t_post=1.0, noise_std=1.4e-3, seed=0
        )
        res = fit_single_movement(t, q, q[0], 4.0, build_initial_guess(VMAX, AMAX, T0))
        assert res.success
        assert res.vmax_fit == pytest.approx(VMAX, rel=0.02)

    def test_il_jacobiano_analitico_costa_meno(self, ground_truth, monkeypatch):
        """A parita' di risultato, valutare il modello meno volte."""
        import src.fitting as modulo

        t, q, _ = ground_truth
        x0 = build_initial_guess(VMAX * 1.2, AMAX * 0.8, T0)
        originale = modulo.model_position

        def conta(config):
            chiamate = []
            monkeypatch.setattr(
                modulo, "model_position",
                lambda *a, **k: (chiamate.append(1), originale(*a, **k))[1],
            )
            res = fit_single_movement(t, q, q[0], q[-1], x0, config=config)
            monkeypatch.undo()
            return len(chiamate), res

        n_con, con = conta(FitConfig())
        n_senza, senza = conta(FitConfig(use_analytic_jacobian=False))

        assert n_con < n_senza
        assert con.vmax_fit == pytest.approx(senza.vmax_fit, rel=1e-4)


class TestProiezioneSulRamo:
    def test_guess_nel_ramo_degenere_viene_riportato(self, ground_truth):
        # Con vmax^2/amax > |dq| il profilo e' triangolare nel punto di partenza,
        # dove d/dvmax e' identicamente nulla: senza proiezione l'ottimizzatore
        # non ha direzione di discesa e si ferma su un minimo molto peggiore.
        t, q, _ = ground_truth
        dq = 4.0
        x0 = np.array([np.sqrt(AMAX * dq) * 1.5, AMAX, T0])  # vmax^2/amax > dq
        assert x0[0] ** 2 / x0[1] > dq

        con = fit_single_movement(t, q, q[0], q[-1], x0)
        senza = fit_single_movement(t, q, q[0], q[-1], x0,
                                    config=FitConfig(branch_margin=-1.0))
        assert con.rmse < senza.rmse
        assert con.vmax_fit == pytest.approx(VMAX, rel=1e-2)


class TestIdentificabilita:
    def test_ramp_samples_coerente(self, ground_truth):
        t, q, _ = ground_truth
        res = fit_single_movement(t, q, q[0], q[-1], build_initial_guess(VMAX, AMAX, T0))
        # La rampa dura vmax/amax = 0.25 s = 6.25 campioni a 25 Hz.
        assert res.ramp_samples == pytest.approx(VMAX / AMAX * FS, rel=1e-2)
        assert res.amax_identifiable

    def test_rampa_troppo_corta_non_identificabile(self):
        # amax alta -> rampa breve -> il residuo diventa insensibile ad amax.
        t, q, _ = generate_trapezoidal_profile(0.0, 4.0, 1.5, 60.0, FS, t_pre=T0)
        res = fit_single_movement(t, q, q[0], q[-1], build_initial_guess(1.5, 60.0, T0))
        assert res.ramp_samples < 5
        assert not res.amax_identifiable


class TestRobustezza:
    """Nessun ingresso deve produrre un'eccezione: un batch non si interrompe."""

    @pytest.mark.parametrize(
        "t,q,q0,qf,x0",
        [
            (np.arange(50) / FS, np.zeros(30), 0.0, 1.0, [1.5, 6.0, 0.0]),   # lunghezze diverse
            (np.arange(2) / FS, np.zeros(2), 0.0, 1.0, [1.5, 6.0, 0.0]),     # troppo corti
            (np.arange(50) / FS, np.zeros(50), 1.0, 1.0, [1.5, 6.0, 0.0]),   # dq nullo
            (np.arange(50) / FS, np.full(50, np.nan), 0.0, 1.0, [1.5, 6.0, 0.0]),  # NaN
            (np.arange(50) / FS, np.zeros(50), 0.0, 1.0, [1.5, 6.0]),        # guess corto
        ],
    )
    def test_fallisce_senza_sollevare(self, t, q, q0, qf, x0):
        res = fit_single_movement(t, q, q0, qf, x0)
        assert not res.success
        assert res.message

    def test_profilo_sconosciuto(self):
        res = fit_single_movement(np.arange(50) / FS, np.zeros(50), 0.0, 1.0,
                                  [1.0], profile_type="bezier")
        assert not res.success


class TestOnset:
    def test_trova_l_inizio_del_moto(self, ground_truth):
        t, q, _ = ground_truth
        # Grossolano per costruzione: quantizza a un campione (40 ms).
        assert estimate_t0(t, q) == pytest.approx(T0, abs=3 / FS)

    def test_segnale_fermo(self):
        t = np.arange(50) / FS
        assert estimate_t0(t, np.zeros(50)) == pytest.approx(t[0])


class TestGuessIniziale:
    @pytest.mark.parametrize(
        "profilo,attesi", [("trapezoidal", 3), ("triangular", 2),
                           ("asymmetric_trapezoidal", 4), ("quintic", 2)],
    )
    def test_lunghezza_per_profilo(self, profilo, attesi):
        x0 = build_initial_guess(VMAX, AMAX, T0, profilo, delta_q=4.0)
        assert x0.size == attesi
        assert np.all(np.isfinite(x0))

    def test_asimmetrico_parte_dal_simmetrico(self):
        x0 = build_initial_guess(VMAX, AMAX, T0, "asymmetric_trapezoidal", delta_q=4.0)
        assert x0[1] == x0[2] == AMAX

    @pytest.mark.parametrize("vmax,amax", [(0.0, 6.0), (1.5, -1.0), (np.nan, 6.0)])
    def test_stime_non_valide(self, vmax, amax):
        with pytest.raises(ValueError):
            build_initial_guess(vmax, amax, T0)


class TestBatch:
    def test_un_fallimento_non_ferma_il_lotto(self, ground_truth):
        t, q, _ = ground_truth
        dati = {
            "buono": {"t_data": t, "q_data": q, "q0": q[0], "qf": q[-1],
                      "vmax_est": VMAX, "amax_est": AMAX, "joint_name": "j1"},
            "rotto": {"t_data": t, "q_data": q, "q0": 1.0, "qf": 1.0,
                      "vmax_est": VMAX, "amax_est": AMAX, "joint_name": "j2"},
            "senza_stime": {"t_data": t, "q_data": q, "q0": q[0], "qf": q[-1],
                            "joint_name": "j3"},
        }
        risultati, tabella = fit_all_movements(dati)
        assert len(risultati) == 3
        assert risultati[0].success and not risultati[1].success and not risultati[2].success
        assert len(tabella) == 3
        assert "vmax_fit" in tabella.columns
