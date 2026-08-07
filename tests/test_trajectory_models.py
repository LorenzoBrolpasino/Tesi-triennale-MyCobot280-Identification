"""Test dei profili teorici (Modulo 2)."""

import numpy as np
import pytest

from src.trajectory_models import (
    asymmetric_position, asymmetric_times, classify_profile_type, model_position,
    profile_times, quintic_position, ramp_time, total_time,
    trapezoidal_position, trapezoidal_position_jacobian, trapezoidal_velocity,
)


class TestTrapezoidale:
    def test_raggiunge_il_bersaglio(self):
        q0, qf, vmax, amax = 0.3, 2.8, 1.5, 6.0
        tm = profile_times(q0, qf, vmax, amax)
        assert trapezoidal_position(np.array([tm.t_total]), q0, qf, vmax, amax) == pytest.approx(qf)

    def test_parte_dalla_posizione_iniziale(self):
        t = np.linspace(-1.0, 0.0, 20)
        q = trapezoidal_position(t, 0.3, 2.8, 1.5, 6.0)
        assert np.allclose(q, 0.3)

    def test_clippato_fuori_dal_moto(self):
        q0, qf = 0.3, 2.8
        tm = profile_times(q0, qf, 1.5, 6.0)
        q = trapezoidal_position(np.array([-5.0, tm.t_total + 5.0]), q0, qf, 1.5, 6.0)
        assert q[0] == pytest.approx(q0)
        assert q[1] == pytest.approx(qf)

    def test_spostamento_negativo(self):
        q0, qf, vmax, amax = 1.0, -2.0, 1.2, 5.0
        tm = profile_times(q0, qf, vmax, amax)
        t = np.linspace(0, tm.t_total, 200)
        q = trapezoidal_position(t, q0, qf, vmax, amax)
        assert q[-1] == pytest.approx(qf)
        assert np.all(np.diff(q) <= 1e-12)  # monotono decrescente

    def test_velocita_di_crociera_raggiunta(self):
        vmax, amax = 1.5, 6.0
        tm = profile_times(0.0, 4.0, vmax, amax)
        t = np.array([tm.t_acc + tm.t_flat / 2])
        assert trapezoidal_velocity(t, 0.0, 4.0, vmax, amax) == pytest.approx(vmax)

    def test_posizione_continua_ai_raccordi(self):
        q0, qf, vmax, amax = 0.0, 4.0, 1.5, 6.0
        tm = profile_times(q0, qf, vmax, amax)
        eps = 1e-9
        for t_switch in (tm.t_acc, tm.t_acc + tm.t_flat, tm.t_total):
            prima = trapezoidal_position(np.array([t_switch - eps]), q0, qf, vmax, amax)
            dopo = trapezoidal_position(np.array([t_switch + eps]), q0, qf, vmax, amax)
            assert prima == pytest.approx(dopo, abs=1e-7)

    def test_velocita_e_derivata_della_posizione(self):
        q0, qf, vmax, amax = 0.0, 4.0, 1.5, 6.0
        t = np.linspace(0, profile_times(q0, qf, vmax, amax).t_total, 4001)
        v_num = np.gradient(trapezoidal_position(t, q0, qf, vmax, amax), t)
        v_an = trapezoidal_velocity(t, q0, qf, vmax, amax)
        # I raccordi sono spigoli: si confronta lontano da quelli.
        interno = np.abs(v_num - v_an) < 1e6
        assert np.median(np.abs(v_num - v_an)[interno]) < 1e-6


class TestDegenerazioneTriangolare:
    def test_sotto_soglia_e_triangolare(self):
        vmax, amax = 2.0, 4.0
        dq_crit = vmax**2 / amax  # 1.0
        assert profile_times(0.0, 0.5, vmax, amax).profile_type == "triangular"
        assert profile_times(0.0, 2.0, vmax, amax).profile_type == "trapezoidal"
        assert classify_profile_type(dq_crit, vmax, amax) == "trapezoidal"
        assert classify_profile_type(dq_crit * 0.999, vmax, amax) == "triangular"

    def test_velocita_di_picco_del_triangolare(self):
        vmax, amax, dq = 2.0, 4.0, 0.5
        tm = profile_times(0.0, dq, vmax, amax)
        assert tm.v_peak == pytest.approx(np.sqrt(amax * dq))
        assert tm.v_peak < vmax
        assert tm.t_flat == 0.0

    def test_il_triangolare_raggiunge_il_bersaglio(self):
        amax, qf = 4.0, 0.5
        p = [amax, 0.0]
        q = model_position("triangular", np.array([total_time("triangular", 0.0, qf, p)]),
                           0.0, qf, p)
        assert q == pytest.approx(qf)


class TestJacobiano:
    def test_contro_differenze_finite(self):
        q0, qf, vmax, amax, t0 = 0.0, 4.0, 1.5, 6.0, 0.7
        tm = profile_times(q0, qf, vmax, amax)
        # Campioni lontani dai raccordi: li' il Jacobiano e' discontinuo.
        t = t0 + np.array([0.10, 0.20, 0.50, 0.90, 1.40, 1.90, 2.30, 2.60])
        assert np.all(t - t0 < tm.t_total)

        analitico = trapezoidal_position_jacobian(t, q0, qf, vmax, amax, t0)
        h = 1e-7
        numerico = np.empty_like(analitico)
        for k, p in enumerate((vmax, amax, t0)):
            su = [vmax, amax, t0]; giu = [vmax, amax, t0]
            su[k], giu[k] = p + h, p - h
            numerico[:, k] = (
                trapezoidal_position(t, q0, qf, su[0], su[1], su[2])
                - trapezoidal_position(t, q0, qf, giu[0], giu[1], giu[2])
            ) / (2 * h)

        assert np.allclose(analitico, numerico, atol=1e-5)

    def test_colonna_vmax_nulla_nel_triangolare(self):
        # Sotto la soglia il profilo non dipende da vmax: e' la ragione per cui
        # il Modulo 3 proietta il guess sul ramo trapezoidale.
        jac = trapezoidal_position_jacobian(np.linspace(0, 1, 50), 0.0, 0.5, 2.0, 4.0)
        assert np.allclose(jac[:, 0], 0.0)


class TestAsimmetrico:
    def test_con_rampe_uguali_coincide_col_simmetrico(self):
        q0, qf, vmax, a = 0.0, 4.0, 1.5, 6.0
        t = np.linspace(-0.5, 4.0, 500)
        assert np.allclose(
            asymmetric_position(t, q0, qf, vmax, a, a),
            trapezoidal_position(t, q0, qf, vmax, a),
        )

    def test_raggiunge_il_bersaglio_con_rampe_diverse(self):
        q0, qf, vmax, a_acc, a_dec = 0.0, 4.19, 2.64, 8.70, 7.52
        tm = asymmetric_times(q0, qf, vmax, a_acc, a_dec)
        assert tm.t_acc < tm.t_dec  # accelera piu' forte di quanto deceleri
        q = asymmetric_position(np.array([tm.t_total]), q0, qf, vmax, a_acc, a_dec)
        assert q == pytest.approx(qf)

    def test_degenerazione_triangolare_asimmetrica(self):
        vmax, a_acc, a_dec = 3.0, 8.0, 6.0
        d_crit = 0.5 * vmax**2 * (1 / a_acc + 1 / a_dec)
        tm = asymmetric_times(0.0, d_crit * 0.5, vmax, a_acc, a_dec)
        assert tm.profile_type == "triangular"
        assert tm.v_peak < vmax


class TestQuintico:
    def test_estremi_e_derivate_nulle(self):
        q0, qf, T = 0.5, 3.5, 2.0
        t = np.linspace(0, T, 2001)
        q = quintic_position(t, q0, qf, T)
        assert q[0] == pytest.approx(q0)
        assert q[-1] == pytest.approx(qf)
        v = np.gradient(q, t)
        assert abs(v[0]) < 1e-3 and abs(v[-1]) < 1e-3

    def test_niente_tratto_a_velocita_costante(self):
        # E' la ragione per cui perde il confronto del Modulo 4.
        t = np.linspace(0, 2.0, 1001)
        v = np.gradient(quintic_position(t, 0.0, 3.0, 2.0), t)
        centro = v[(t > 0.5) & (t < 1.5)]
        assert centro.std() / centro.mean() > 0.05


class TestInterfacciaComune:
    @pytest.mark.parametrize(
        "profilo,params",
        [("trapezoidal", [1.5, 6.0, 0.0]), ("triangular", [6.0, 0.0]),
         ("asymmetric_trapezoidal", [1.5, 6.0, 5.0, 0.0]), ("quintic", [2.0, 0.0])],
    )
    def test_model_position_raggiunge_il_bersaglio(self, profilo, params):
        q0, qf = 0.0, 4.0
        T = total_time(profilo, q0, qf, params)
        q = model_position(profilo, np.array([T]), q0, qf, params)
        assert q == pytest.approx(qf, abs=1e-9)

    def test_numero_di_parametri_sbagliato(self):
        with pytest.raises(ValueError, match="richiede"):
            model_position("trapezoidal", np.array([0.0]), 0.0, 1.0, [1.5, 6.0])

    def test_profilo_sconosciuto(self):
        with pytest.raises(ValueError, match="sconosciuto"):
            model_position("bezier", np.array([0.0]), 0.0, 1.0, [1.0])

    def test_ramp_time_coerente(self):
        assert ramp_time("trapezoidal", 0.0, 4.0, [1.5, 6.0, 0.0]) == pytest.approx(0.25)
        assert np.isnan(ramp_time("quintic", 0.0, 4.0, [2.0, 0.0]))


class TestVincoliDiIngresso:
    @pytest.mark.parametrize("vmax,amax", [(0.0, 6.0), (-1.0, 6.0), (1.5, 0.0), (np.inf, 6.0)])
    def test_parametri_non_validi(self, vmax, amax):
        with pytest.raises(ValueError):
            profile_times(0.0, 1.0, vmax, amax)

    def test_spostamento_nullo(self):
        tm = profile_times(1.0, 1.0, 1.5, 6.0)
        assert tm.profile_type == "degenerate"
        assert tm.t_total == 0.0
        assert np.allclose(trapezoidal_position(np.linspace(0, 1, 10), 1.0, 1.0, 1.5, 6.0), 1.0)
