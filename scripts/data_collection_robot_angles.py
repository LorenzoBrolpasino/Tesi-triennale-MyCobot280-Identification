#!/usr/bin/env python3
"""Acquisizione dei dati di giunto dal MyCobot280.

Comanda a ciascun giunto un movimento ampio (-120 -> +120 gradi e ritorno) e ne
registra la posizione a 25 Hz con un ciclo temporizzato su ``perf_counter``.
Angoli inviati e letti in radianti. Funziona sul robot reale o sul simulatore,
secondo ``SIMULATION_MODE``. Salva i campioni in ``.npy`` e ``.csv``.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from mycobot280_wrapper import MyCobot280Wrapper

# 1. MODALITÀ APPLICAZIONE (SIMULAZIONE VS REALE)
SIMULATION_MODE = False  # Impostare a False per usare il robot reale
REAL_ROBOT_TYPE = "M5"  # "M5" per la versione montata sui veicoli, "pi" per quella sui tavoli

# Frequenza e periodo di campionamento stabili (tra 20 Hz e 30 Hz)
SAMPLING_FREQUENCY = 25.0  # Hz
SAMPLING_PERIOD = 1.0 / SAMPLING_FREQUENCY  # secondi (0.04s)

# Velocità massima per stimare la velocità di regime permanente (1-100)
MOVEMENT_VELOCITY = 25

# File di salvataggio nella stessa cartella del codice
CSV_OUTPUT_FILE = "mycobot280_angle_data.csv"
NPY_OUTPUT_FILE = "mycobot280_angle_data.npy"

# Limiti di sicurezza in RADIANTI per ciascun giunto (j1, j2, j3, j4, j5, j6)
# I movimenti sono centrati rispetto alla pos. zero per evitare auto-collisioni e impatti
SAFE_LIMITS = [
    [-np.radians(120.0), np.radians(120.0)],  # Giunto 1 (Rotazione Base)
    [-np.radians(120.0), np.radians(120.0)],  # Giunto 2
    [-np.radians(120.0), np.radians(120.0)],  # Giunto 3
    [-np.radians(120.0), np.radians(120.0)],  # Giunto 4
    [-np.radians(120.0), np.radians(120.0)],  # Giunto 5
    [-np.radians(120.0), np.radians(120.0)],  # Giunto 6
]

# Configurazione iniziale di home (tutti zeri in radianti)
q_home = np.zeros(6)

# Strutture dati per raccogliere i dati
timestamps = []
all_angles = []
global_start_time = None

# Inizializzazione del robot tramite wrapper
robot_type = "sim" if SIMULATION_MODE else REAL_ROBOT_TYPE
q_home_rad = q_home

print(f"Inizializzazione di MyCobot280Wrapper in modalità: {robot_type.upper()}")
robot = MyCobot280Wrapper(robot_type=robot_type, dt=SAMPLING_PERIOD, q_init=q_home_rad)


# Invia al robot i target angolari dei sei giunti, in radianti.
def send_angles(target_rad, vel=100, wait_to_reach=False):
    robot.send_q_ref(target_rad, vel=vel, wait_to_reach=wait_to_reach)


# Legge la posizione corrente dei sei giunti, in radianti.
def read_angles():
    return robot.read_q()


# Comanda un movimento e ne registra la traiettoria a 25 Hz, con una quiete iniziale che
# fa da linea di base.
def record_transition(target_rad, duration, initial_wait=1.5):
    global global_start_time
    
    # Legge la postura di partenza (in radianti)
    q_start_rad = robot.read_q()
    
    # Loop temporizzato per la durata della transizione + il tempo di attesa iniziale
    start_loop = time.perf_counter()
    next_sample_time = start_loop
    
    command_sent = False
    
    while (time.perf_counter() - start_loop) < (duration + initial_wait):
        elapsed = time.perf_counter() - start_loop
        
        # Invia il comando di movimento solo dopo il tempo di attesa iniziale per registrare la linea di base
        if elapsed >= initial_wait and not command_sent:
            print(f"  -> Comando movimento verso: {np.round(target_rad, 3)} (rad)")
            send_angles(target_rad, vel=MOVEMENT_VELOCITY, wait_to_reach=False)
            command_sent = True
            
        # Avanza il simulatore se siamo in simulazione
        if SIMULATION_MODE:
            # Mantiene la postura di partenza se il comando non è ancora stato inviato
            current_target = target_rad if command_sent else q_start_rad
            robot.simulate(current_target, MOVEMENT_VELOCITY)
            
        current_time = time.perf_counter()
        if global_start_time is None:
            global_start_time = current_time
            
        # Lettura giunti (in radianti)
        q_rad = read_angles()
        
        # Salvataggio dati
        timestamps.append(current_time - global_start_time)
        all_angles.append(q_rad.copy())
        
        # Calcolo del tempo di attesa per mantenere stabile la frequenza
        next_sample_time += SAMPLING_PERIOD
        sleep_time = next_sample_time - time.perf_counter()
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            # Recupera se c'è stato del ritardo
            next_sample_time = time.perf_counter()


# Salva i campioni raccolti in formato .npy e .csv.
def save_data():
    if len(timestamps) == 0:
        print("Nessun dato raccolto da salvare.")
        return
        
    t_arr = np.array(timestamps)
    q_arr = np.array(all_angles)
    
    # Creazione della matrice dei dati: [timestamp, j1, j2, j3, j4, j5, j6]
    data_matrix = np.column_stack((t_arr, q_arr))
    
    print("\n" + "="*60)
    print("SALVATAGGIO DATI IN CORSO...")
    print("="*60)
    
    # 1. Salvataggio in formato .npy
    np.save(NPY_OUTPUT_FILE, data_matrix)
    print(f"A. Salvato file NumPy: {NPY_OUTPUT_FILE} con shape {data_matrix.shape}")
    
    # 2. Salvataggio in formato .csv con intestazione richiesta
    header = "timestamp,j1,j2,j3,j4,j5,j6"
    np.savetxt(CSV_OUTPUT_FILE, data_matrix, delimiter=",", header=header, comments="")
    print(f"B. Salvato file CSV: {CSV_OUTPUT_FILE}")
    print(f"   Dati totali: {len(t_arr)} campioni ({t_arr[-1]:.2f} secondi)")


# Disegna le sei traiettorie acquisite, come verifica rapida dell'acquisizione.
def plot_data():
    if len(timestamps) == 0:
        return
        
    t_arr = np.array(timestamps)
    q_arr = np.array(all_angles)
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 8))
    fig.suptitle("Traiettorie dei Giunti MyCobot280 (Radianti)", fontsize=14, fontweight="bold")
    
    axes = axes.flatten()
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    
    for i in range(6):
        ax = axes[i]
        ax.plot(t_arr, q_arr[:, i], label=f"Giunto {i+1}", color=colors[i], linewidth=1.5)
        ax.set_title(f"Giunto {i+1}")
        ax.set_xlabel("Tempo (s)")
        ax.set_ylabel("Angolo (rad)")
        ax.grid(True)
        ax.legend()
        
    plt.tight_layout()
    plot_filename = "mycobot280_trajectories.png"
    plt.savefig(plot_filename, dpi=150)
    print(f"Grafico salvato in: {plot_filename}")
    try:
        plt.show()
    except Exception as e:
        print(f"Impossibile visualizzare il grafico (modalità headless): {e}")


# Esegue l'acquisizione completa: per ogni giunto un movimento di andata e ritorno.
def main():
    # Accensione motori
    print("Attivazione dei motori...")
    robot.turn_motors_on()
    time.sleep(0.5)
    
    # Spostamento iniziale in posizione di home (sincrono)
    print("\nPosizionamento iniziale in HOME [0, 0, 0, 0, 0, 0]...")
    send_angles(q_home, vel=100, wait_to_reach=False)
    time.sleep(1.0)
    
    # Durata della registrazione di ciascun movimento (min -> max, max -> min)
    T_record = 8  # s
    
    try:
        for joint_idx in range(6):
            print("\n" + "="*60)
            print(f"AVVIO TEST AMPIO MOVIMENTO PER GIUNTO {joint_idx + 1}")
            print("="*60)
            
            q_min, q_max = SAFE_LIMITS[joint_idx]
            
            # Postura iniziale con giunto i a q_min
            q_start = np.zeros(6)
            q_start[joint_idx] = q_min
            
            # Postura finale con giunto i a q_max
            q_end = np.zeros(6)
            q_end[joint_idx] = q_max
            
            # 1. Posizionamento iniziale prima della registrazione (sincrono)
            print(f"\nSpostamento sincrono al punto di partenza ({q_min:.3f} rad)...")
            send_angles(q_start, vel=100, wait_to_reach=True)
            time.sleep(0.5)  # Stabilizzazione
            
            # 2. Registrazione transizione: min -> max
            record_transition(q_end, duration=T_record)
            time.sleep(0.5)
            
            # 3. Registrazione transizione: max -> min
            record_transition(q_start, duration=T_record)
            time.sleep(0.5)
            
            # 4. Ritorno in HOME (sincrono)
            print("\nRitorno sincrono in HOME...")
            send_angles(q_home, vel=100, wait_to_reach=True)
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\n\nAcquisizione interrotta dall'utente.")
    except Exception as e:
        print(f"\n\nErrore durante il ciclo di acquisizione: {e}")
    finally:
        # Spegnimento motori
        print("\nDisattivazione motori...")
        robot.turn_motors_off()
        print("Motori spenti.")
        
    # Salvataggio e plot finali
    save_data()
    plot_data()


if __name__ == "__main__":
    main()
