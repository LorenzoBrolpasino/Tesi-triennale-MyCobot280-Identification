"""Test della pipeline di preprocessing (Modulo 1)."""

import numpy as np
import pytest

from src.preprocessing import (
    MovementSegment, analyze_joint_movement, check_sampling_uniformity, classify_profile_type,
    compute_derivatives, detect_steady_state, estimate_amax, estimate_vmax,
    filter_position, segment_movements,
)
from src.trajectory_models import generate_trapezoidal_profile

FS = 25.0


@pytest.fixture
def profilo_sintetico():
    """Trapezoidale a parametri noti, campionato come i dati reali."""
    return generate_trapezoidal_profile(
        q0=0.0, delta_q=4.0, vmax=1.5, amax=6.0, fs=FS, t_pre=1.0, t_post=1.0
    )


class TestSegmentazione:
    def test_separa_sui_buchi_temporali(self):
        # Tre movimenti separati da pause di 2 s: sono i buchi a delimitarli.
        pezzi_t, pezzi_q, t_corrente = [], [], 0.0
        for k in range(3):
            t = t_corrente + np.arange(50) / FS
            pezzi_t.append(t)
            pezzi_q.append(np.linspace(0, 1.0 * (k + 1), 50))
            t_corrente = t[-1] + 2.0

        seg = segment_movements(np.concatenate(pezzi_t),
                                np.concatenate(pezzi_q)[:, None])
        assert len(seg) == 3
        assert [s.travel for s in seg] == pytest.approx([1.0, 2.0, 3.0])

    def test_scarta_i_segmenti_immobili(self):
        t = np.arange(100) / FS
        q = np.zeros((100, 1))
        assert segment_movements(t, q) == []

    def test_individua_il_giunto_in_moto(self):
        t = np.arange(60) / FS
        Q = np.zeros((60, 6))
        Q[:, 3] = np.linspace(0, 2.0, 60)  # solo j4 si muove
        seg = segment_movements(t, Q)
        assert len(seg) == 1
        assert seg[0].joint_index == 3
        assert seg[0].joint_name == "j4"

    def test_lunghezze_incoerenti(self):
        with pytest.raises(ValueError, match="lunghezze diverse"):
            segment_movements(np.arange(10) / FS, np.zeros((5, 1)))

    def test_motion_bounds_esclude_la_quiete(self):
        # Quiete 40 campioni, rampa 40, quiete 40: il moto sta fra 40 e 80.
        q = np.concatenate([np.zeros(40), np.linspace(0, 2.0, 40), np.full(40, 2.0)])
        t = np.arange(q.size) / FS
        seg = segment_movements(t, q[:, None])
        assert len(seg) == 1

        inizio, fine = seg[0].motion_bounds
        assert inizio == pytest.approx(40, abs=3)
        assert fine == pytest.approx(80, abs=3)
        # La finestra di moto e' una frazione stretta del segmento.
        assert (fine - inizio) < 0.5 * q.size

    def test_motion_bounds_su_giunto_fermo(self):
        # Senza movimento non c'e' un tratto da isolare: si restituisce tutto.
        seg = MovementSegment(index=0, start=0, stop=30, time=np.arange(30) / FS,
                              positions=np.zeros((30, 6)), joint_index=0, joint_name="j1")
        assert seg.motion_bounds == (0, 30)


class TestCampionamento:
    def test_passo_uniforme(self):
        rep = check_sampling_uniformity(np.arange(100) / FS)
        assert rep["is_uniform"]
        assert rep["fs_est"] == pytest.approx(FS)
        assert rep["n_gaps"] == 0

    def test_riconosce_i_buchi(self):
        t = np.concatenate([np.arange(50) / FS, 5.0 + np.arange(50) / FS])
        rep = check_sampling_uniformity(t)
        assert rep["n_gaps"] == 1
        assert not rep["is_uniform"]

    def test_tempo_non_crescente(self):
        with pytest.raises(ValueError, match="crescente"):
            check_sampling_uniformity(np.array([0.0, 0.1, 0.05, 0.2]))


class TestFiltroEDerivate:
    def test_non_introduce_sfasamento(self):
        # E' la ragione per cui si usa filtfilt e non lfilter: il picco di una
        # sinusoide deve restare dov'era.
        t = np.arange(500) / FS
        x = np.sin(2 * np.pi * 0.5 * t)
        y = filter_position(x, fs=FS, cutoff_hz=3.0)
        assert abs(int(np.argmax(y[:100])) - int(np.argmax(x[:100]))) <= 1

    def test_attenua_il_rumore(self):
        t = np.arange(500) / FS
        pulito = np.sin(2 * np.pi * 0.3 * t)
        rumoroso = pulito + np.random.default_rng(0).normal(0, 0.05, t.size)
        filtrato = filter_position(rumoroso, fs=FS, cutoff_hz=3.0)
        assert np.std(filtrato - pulito) < np.std(rumoroso - pulito)

    def test_cutoff_oltre_nyquist(self):
        with pytest.raises(ValueError, match="Nyquist|cutoff_hz"):
            filter_position(np.zeros(100), fs=FS, cutoff_hz=20.0)

    def test_derivate_di_una_rampa(self):
        dt = 1 / FS
        q = 2.5 * np.arange(100) * dt
        v, a = compute_derivatives(q, dt)
        assert np.allclose(v[2:-2], 2.5)
        assert np.allclose(a[2:-2], 0.0, atol=1e-9)


class TestStime:
    def test_vmax_sul_profilo_sintetico(self, profilo_sintetico):
        t, q, info = profilo_sintetico
        res = analyze_joint_movement(time=t, position=q, cutoff_hz=3.0)
        assert res.vmax_est == pytest.approx(info["vmax_nominal"], rel=0.01)
        assert res.profile_type == "trapezoidal"

    def test_nessun_regime_nel_triangolare(self):
        # Spostamento sotto la soglia: la crociera non esiste.
        t, q, _ = generate_trapezoidal_profile(0.0, 0.2, vmax=1.5, amax=6.0, fs=FS)
        res = analyze_joint_movement(time=t, position=q, cutoff_hz=3.0)
        assert res.steady_state_window is None
        assert res.vmax_est is None
        assert res.profile_type == "triangular"

    def test_regime_su_velocita_costante(self):
        v = np.concatenate([np.linspace(0, 1.5, 20), np.full(60, 1.5),
                            np.linspace(1.5, 0, 20)])
        a = np.gradient(v, 1 / FS)
        finestra = detect_steady_state(v, a)
        assert finestra is not None
        inizio, fine = finestra
        assert fine - inizio >= 40
        assert estimate_vmax(v, finestra)[0] == pytest.approx(1.5, rel=1e-3)

    def test_amax_robusta_agli_spike(self):
        dt = 1 / FS
        a = np.concatenate([np.full(20, 6.0), np.zeros(40), np.full(20, -6.0)])
        a[5] = 200.0  # spike da doppia derivazione di un segnale quantizzato
        stima = estimate_amax(a, (20, 60))
        assert stima == pytest.approx(6.0, rel=0.05)

    def test_amax_non_stimabile(self):
        assert estimate_amax(np.zeros(50), None) is None


class TestClassificazione:
    def test_soglia(self):
        vmax, amax = 1.5, 6.0
        dq_crit = vmax**2 / amax
        assert classify_profile_type(dq_crit * 1.01, vmax, amax) == "trapezoidal"
        assert classify_profile_type(dq_crit * 0.99, vmax, amax) == "triangular"
        assert classify_profile_type(-dq_crit * 1.01, vmax, amax) == "trapezoidal"

    def test_stime_mancanti(self):
        assert classify_profile_type(4.0, None, 6.0) == "triangular"
        assert classify_profile_type(4.0, 1.5, None) == "unknown"
        assert classify_profile_type(4.0, 1.5, -1.0) == "unknown"


class TestAnalisiCompleta:
    def test_segnale_costante(self):
        res = analyze_joint_movement(position=np.ones(100), fs=FS)
        assert not res.reliable
        assert res.profile_type == "no_motion"

    def test_segnale_troppo_corto(self):
        res = analyze_joint_movement(position=np.array([0.0, 1.0]), fs=FS)
        assert not res.reliable

    def test_segnala_amax_sottostimata(self, profilo_sintetico):
        # Rampa di 0.25 s contro uno smussamento del filtro di 0.33 s a 3 Hz.
        t, q, _ = profilo_sintetico
        res = analyze_joint_movement(time=t, position=q, cutoff_hz=3.0)
        assert not res.amax_reliable
        assert any("SOTTOSTIMATA" in w for w in res.warnings)

    def test_serve_almeno_un_riferimento_temporale(self):
        with pytest.raises(ValueError, match="time.*fs.*dt|Serve almeno"):
            analyze_joint_movement(position=np.zeros(50))
