#!/usr/bin/env python3
"""Modulo 6 — Visualizzazione 3D della traiettoria.

Legge l'URDF, calcola la cinematica diretta e disegna le mesh in MeshCat, che
apre una vista nel browser. Contiene solo cio' che serve a guardare una
traiettoria: niente collisioni, niente Jacobiano. ``meshcat`` e
``urdf_parser_py`` sono dipendenze opzionali e servono soltanto qui.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np

__all__ = [
    "rpy_to_rot",
    "xyz_rpy_to_homogeneous",
    "angle_axis_to_homogeneous",
    "parse_urdf",
    "forward_kinematics",
    "RobotViewer",
    "URDF_DEFAULT",
]

#: URDF del MyCobot280-M5, incluso nel progetto sotto ``robot_description/``.
#: Il modello e' di Elephant Robotics, distribuito con licenza BSD 2-Clause
#: (cfr. ``robot_description/LICENSE``).
URDF_DEFAULT = Path(__file__).resolve().parents[1] / (
    "robot_description/urdf/mycobot_280_m5/mycobot_280_m5_capsules.urdf"
)


# ---------------------------------------------------------------------------
# Geometria
# ---------------------------------------------------------------------------


# Converte una terna roll-pitch-yaw nella matrice di rotazione corrispondente.
def rpy_to_rot(rpy) -> np.ndarray:
    r, p, y = (float(v) for v in rpy)
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])


# Compone traslazione e rotazione in una singola trasformazione omogenea 4x4.
def xyz_rpy_to_homogeneous(xyz, rpy) -> np.ndarray:
    H = np.identity(4)
    H[:3, :3] = rpy_to_rot(rpy)  # blocco di rotazione
    H[:3, 3] = np.asarray(xyz, dtype=float)  # blocco di traslazione
    return H


# Trasformazione omogenea di una rotazione attorno a un asse, con la formula di
# Rodrigues.
def angle_axis_to_homogeneous(angle: float, axis) -> np.ndarray:
    v = np.asarray(axis, dtype=float)
    v = v / np.linalg.norm(v)  # l'asse dev'essere unitario
    ca, sa = np.cos(angle), np.sin(angle)
    vvT = np.outer(v, v)  # proiettore sull'asse di rotazione
    # matrice del prodotto vettoriale
    skew = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])

    H = np.identity(4)
    H[:3, :3] = vvT + (np.identity(3) - vvT) * ca + skew * sa  # formula di Rodrigues
    return H


# ---------------------------------------------------------------------------
# Modello
# ---------------------------------------------------------------------------


# Legge dall'URDF la catena cinematica: giunti con asse di rotazione e link associati.
def parse_urdf(urdf_path: str | Path = URDF_DEFAULT) -> tuple[list, list]:
    from urdf_parser_py.urdf import URDF

    urdf_path = Path(urdf_path)
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF non trovato: {urdf_path}")

    modello = URDF.from_xml_string(urdf_path.read_text())

    joints, collegati = [], set()
    for j in modello.joints:
        if j.axis is not None and np.linalg.norm(j.axis) > 0:
            joints.append(j)
            collegati.update((j.parent, j.child))

    links = [l for l in modello.links if l.name in collegati]
    return joints, links


# Calcola la posa di ogni link componendo le trasformazioni dei giunti lungo la catena.
def forward_kinematics(q, joints) -> list[np.ndarray]:
    q = np.asarray(q, dtype=float).ravel()
    if q.size < len(joints):
        raise ValueError(f"attesi {len(joints)} valori di giunto, ricevuti {q.size}.")

    H = np.identity(4)
    pose = []
    for i, giunto in enumerate(joints):
        # posa fissa del giunto
        H = H @ xyz_rpy_to_homogeneous(giunto.origin.xyz, giunto.origin.rpy)
        H = H @ angle_axis_to_homogeneous(q[i], giunto.axis)  # rotazione comandata del giunto
        pose.append(H.copy())
    return pose


# ---------------------------------------------------------------------------
# Visualizzatore
# ---------------------------------------------------------------------------


# Visualizzatore 3D del robot in MeshCat, per guardare una traiettoria nel browser.
class RobotViewer:

    # Legge l'URDF e prepara il visualizzatore, senza ancora aprirlo.
    def __init__(self, urdf_path: str | Path = URDF_DEFAULT):
        self.urdf_path = Path(urdf_path)
        self.mesh_root = self.urdf_path.parents[2]
        self.joints, self.links = parse_urdf(self.urdf_path)
        self.viz = None

    # Apre la vista nel browser e vi carica le mesh dei link.
    def open(self, open_browser: bool = True):
        import meshcat
        import meshcat.geometry as geom

        self.viz = meshcat.Visualizer()
        if open_browser:
            self.viz.open()

        for link in self.links:
            if link.visual is None or link.visual.geometry is None:
                continue
            nome_file = getattr(link.visual.geometry, "filename", None)
            if nome_file is None:
                continue

            # I percorsi URDF sono nella forma "package://<pkg>/<path>".
            relativo = nome_file.split("://", 1)[-1].split("/", 1)[-1]
            mesh = self.mesh_root / relativo
            if not mesh.exists():
                print(f"ATTENZIONE: mesh non trovata per il link {link.name}: {mesh}")
                continue

            estensione = mesh.suffix.lower()
            if estensione == ".stl":
                g = geom.StlMeshGeometry.from_file(str(mesh))
            elif estensione == ".dae":
                g = geom.DaeMeshGeometry.from_file(str(mesh))
            else:
                print(f"ATTENZIONE: formato mesh non gestito ({estensione}) per {link.name}")
                continue

            self.viz[link.name].set_object(g, geom.MeshPhongMaterial())

        return self.viz

    # Porta il robot nella configurazione indicata, aggiornando la posa di ogni link.
    def set_configuration(self, q) -> None:
        if self.viz is None:
            raise RuntimeError("il visualizzatore non e' aperto: chiamare open() prima.")

        pose = forward_kinematics(q, self.joints)
        for i, giunto in enumerate(self.joints):
            # Il link mosso dal giunto i e' il suo figlio: e' il primo link
            # della lista a valere come base, quindi lo scorrimento e' sfalsato.
            link = self.links[i + 1] if i + 1 < len(self.links) else None
            if link is None or link.visual is None:
                continue
            H = pose[i]
            if link.visual.origin is not None:
                H = H @ xyz_rpy_to_homogeneous(
                    link.visual.origin.xyz, link.visual.origin.rpy
                )
            self.viz[link.name].set_transform(H)

    # Riproduce una traiettoria nel visualizzatore, in tempo reale o a velocita'
    # alterata.
    def play(self, t, Q, realtime: bool = True, speed: float = 1.0) -> None:
        t = np.asarray(t, dtype=float).ravel()
        Q = np.atleast_2d(np.asarray(Q, dtype=float))
        if Q.shape[0] != t.size:
            raise ValueError(f"t ha {t.size} istanti, Q ne ha {Q.shape[0]}.")

        inizio = time.perf_counter()
        for k in range(t.size):
            self.set_configuration(Q[k])
            if realtime:
                atteso = (t[k] - t[0]) / max(speed, 1e-9)
                ritardo = atteso - (time.perf_counter() - inizio)
                if ritardo > 0:
                    time.sleep(ritardo)
