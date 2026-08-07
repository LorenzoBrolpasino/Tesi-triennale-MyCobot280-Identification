#!/usr/bin/env python3
"""Produce il video del robot simulato.

    python3 scripts/make_video.py            # sequenza dell'acquisizione, vel 100
    python3 scripts/make_video.py --vel 50

Apre una vista MeshCat nel browser e riproduce la traiettoria in tempo reale;
MeshCat non registra, quindi si cattura lo schermo. La sequenza e' la stessa
dell'acquisizione reale, cosi' i due video sono confrontabili.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

RADICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RADICE))

from src.simulator import MyCobot280Simulator  # noqa: E402
from src.viewer import URDF_DEFAULT, RobotViewer  # noqa: E402

N_JOINTS = 6
AMPIEZZA = np.radians(120.0)  # come SAFE_LIMITS dell'acquisizione


# Ricostruisce la sequenza di comandi usata per i dati reali, cosi' i due video sono
# confrontabili.
def sequenza_acquisizione(vel: float) -> list[tuple]:
    comandi = []
    for j in range(N_JOINTS):
        for angolo in (-AMPIEZZA, AMPIEZZA, 0.0):
            q = np.zeros(N_JOINTS)
            q[j] = angolo
            comandi.append((q, vel))
    return comandi


# Simula la sequenza dell'acquisizione e la riproduce nel visualizzatore 3D.
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vel", type=float, default=100,
                   help="velocita' comandata 1-100 (default: 100)")
    p.add_argument("--speed", type=float, default=1.0,
                   help="fattore di riproduzione; 1.0 = tempo reale (default: 1.0)")
    p.add_argument("--loop", action="store_true", help="ripete finche' non si interrompe")
    args = p.parse_args()

    sim = MyCobot280Simulator()
    t, Q = sim.simulate_sequence(sequenza_acquisizione(args.vel))
    print(f"simulazione - velocita' comandata {args.vel:g}: "
          f"{len(t)} fotogrammi, {t[-1]:.1f} s")

    if not URDF_DEFAULT.exists():
        print(f"ERRORE: URDF non trovato: {URDF_DEFAULT}", file=sys.stderr)
        return 1

    viewer = RobotViewer(URDF_DEFAULT)
    viewer.open()
    print("Vista MeshCat aperta nel browser. Avvia la cattura dello schermo, "
          "poi premi INVIO per far partire il movimento.")
    input()

    try:
        while True:
            viewer.play(t, Q, realtime=True, speed=args.speed)
            if not args.loop:
                break
    except KeyboardInterrupt:
        print("\nInterrotto.")

    print("Riproduzione conclusa. Ferma la cattura dello schermo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
