"""Test del simulatore e della cinematica diretta (Moduli 5 e 6)."""

import numpy as np
import pytest

from src.simulator import N_JOINTS, MyCobot280Simulator, SimulatorParams
from src.trajectory_models import asymmetric_times


class TestComandoSingolo:
    def test_raggiunge_il_bersaglio(self):
        sim = MyCobot280Simulator()
        q_ref = np.deg2rad([120.0, -60.0, 30.0, 0.0, 45.0, -90.0])
        t, Q = sim.send_q_ref(q_ref, vel=100)

        assert Q.shape == (t.size, N_JOINTS)
        assert np.allclose(Q[0], 0.0)
        assert np.allclose(Q[-1], q_ref)
        assert np.allclose(sim.read_q(), q_ref)

    def test_durata_coerente_col_profilo(self):
        sim = MyCobot280Simulator()
        par = sim.params
        q_ref = np.zeros(N_JOINTS)
        q_ref[0] = np.deg2rad(240.0)  # il giunto piu' lento detta il tempo

        atteso = asymmetric_times(0.0, q_ref[0], par.vmax(100), par.a_acc, par.a_dec).t_total
        assert sim.move_duration(q_ref, vel=100) == pytest.approx(atteso)

    def test_la_velocita_comandata_scala_la_crociera(self):
        sim = MyCobot280Simulator()
        q_ref = np.zeros(N_JOINTS)
        q_ref[0] = np.deg2rad(240.0)

        lento = sim.move_duration(q_ref, vel=25)
        veloce = sim.move_duration(q_ref, vel=100)
        # Solo la crociera scala col comando: le rampe restano, quindi il
        # rapporto e' minore di 4.
        assert 2.5 < lento / veloce < 4.0

    def test_giunti_fermi_restano_fermi(self):
        sim = MyCobot280Simulator()
        q_ref = np.zeros(N_JOINTS)
        q_ref[2] = 1.0
        _, Q = sim.send_q_ref(q_ref, vel=50)
        for j in (0, 1, 3, 4, 5):
            assert np.allclose(Q[:, j], 0.0)

    def test_movimento_monotono(self):
        sim = MyCobot280Simulator()
        q_ref = np.zeros(N_JOINTS)
        q_ref[0] = -2.0
        _, Q = sim.send_q_ref(q_ref, vel=100)
        assert np.all(np.diff(Q[:, 0]) <= 1e-12)

    def test_dimensione_sbagliata(self):
        with pytest.raises(ValueError, match="6"):
            MyCobot280Simulator().send_q_ref(np.zeros(3))


class TestSequenza:
    def test_continua_e_senza_duplicati(self):
        sim = MyCobot280Simulator()
        comandi = [
            (np.deg2rad([120, 0, 0, 0, 0, 0]), 100),
            (np.deg2rad([-120, 0, 0, 0, 0, 0]), 100),
            (np.zeros(6), 50),
        ]
        t, Q = sim.simulate_sequence(comandi)

        assert np.all(np.diff(t) > 0)                 # tempo monotono
        assert np.abs(np.diff(Q, axis=0)).max() < 0.2  # nessun salto
        assert np.allclose(Q[-1], 0.0)                 # torna a casa

    def test_ogni_comando_riparte_da_dove_finisce_il_precedente(self):
        sim = MyCobot280Simulator()
        primo = np.deg2rad([90, 0, 0, 0, 0, 0])
        t1, Q1 = sim.send_q_ref(primo, vel=100)
        t2, Q2 = sim.send_q_ref(np.zeros(6), vel=100)
        assert np.allclose(Q2[0], primo)
        assert t2[0] == pytest.approx(t1[-1])

    def test_reset(self):
        sim = MyCobot280Simulator()
        sim.send_q_ref(np.deg2rad([90, 0, 0, 0, 0, 0]), vel=100)
        sim.reset()
        assert np.allclose(sim.read_q(), 0.0)
        assert sim.t == 0.0


class TestParametri:
    def test_vmax_lineare_nel_comando(self):
        par = SimulatorParams()
        assert par.vmax(100) == pytest.approx(par.vmax_per_unit * 100)
        assert par.vmax(50) == pytest.approx(par.vmax(100) / 2)

    def test_comando_clippato(self):
        par = SimulatorParams()
        assert par.vmax(500) == par.vmax(100)
        assert par.vmax(0) == par.vmax(1)

    def test_accelera_piu_di_quanto_decelera(self):
        # E' il risultato che giustifica il profilo asimmetrico.
        par = SimulatorParams()
        assert par.a_acc > par.a_dec
        assert par.a_acc / par.a_dec == pytest.approx(1.16, abs=0.02)


class TestCinematicaDiretta:
    """Il visualizzatore dipende da urdf_parser_py, che puo' non esserci."""

    def test_rotazione_della_base(self):
        pytest.importorskip("urdf_parser_py")
        from src.viewer import URDF_DEFAULT, forward_kinematics, parse_urdf

        if not URDF_DEFAULT.exists():
            pytest.skip("URDF non disponibile")

        joints, _ = parse_urdf()
        assert len(joints) == N_JOINTS

        a_riposo = forward_kinematics(np.zeros(N_JOINTS), joints)[-1]
        ruotato = forward_kinematics(np.deg2rad([90, 0, 0, 0, 0, 0]), joints)[-1]

        # Ruotare j1 di 90 gradi manda (x, y) in (-y, x) e lascia z invariata.
        assert ruotato[0, 3] == pytest.approx(-a_riposo[1, 3], abs=1e-9)
        assert ruotato[1, 3] == pytest.approx(a_riposo[0, 3], abs=1e-9)
        assert ruotato[2, 3] == pytest.approx(a_riposo[2, 3], abs=1e-9)

    def test_matrici_di_rotazione_valide(self):
        pytest.importorskip("urdf_parser_py")
        from src.viewer import URDF_DEFAULT, forward_kinematics, parse_urdf

        if not URDF_DEFAULT.exists():
            pytest.skip("URDF non disponibile")

        joints, _ = parse_urdf()
        for H in forward_kinematics(np.random.default_rng(0).uniform(-2, 2, N_JOINTS), joints):
            R = H[:3, :3]
            assert np.allclose(R @ R.T, np.eye(3))
            assert float(np.linalg.det(R)) == pytest.approx(1.0)
