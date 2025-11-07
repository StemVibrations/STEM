import json
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft, rfftfreq

# Path naar JSON
json_dyn_path = r"C:\Users\ritfeldis\STEM\Schalkwijk_stem_version_1.2.40\json_output_N_solver_v1.2.4o_stage_2.json"

# JSON inlezen
with open(json_dyn_path, 'r') as f:
    data_dyn = json.load(f)

# Tijdvector en sampling
time_dyn = np.array(data_dyn['TIME'])
Fs = 1.0 / np.mean(np.diff(time_dyn))  # sampling frequency
dt = 1 / Fs

# Selecteer eerste 25 nodes en sorteer op Z-coördinaat
node_keys = [key for key in data_dyn if key.startswith("NODE_")][:25]
node_keys.sort(key=lambda k: data_dyn[k]['COORDINATES'][2])  # sorteren op Z

# === Eén figuur, twee subplots ===
fig, (ax_time, ax_fft) = plt.subplots(2, 1, figsize=(14, 10))

# --- Subplot 1: velocity over time ---
for node_key in node_keys:
    try:
        vel_dyn = np.array(data_dyn[node_key]['VELOCITY_Y'])
        N = len(vel_dyn)
        time_vector = np.arange(N) * dt
        window = np.hamming(N)
        coords = data_dyn[node_key]['COORDINATES']

        ax_time.plot(time_vector, vel_dyn * window, linestyle="-",
                     label=f"{node_key} (X={coords[0]:.1f}, Y={coords[1]:.2f}, Z={coords[2]:.1f})")

    except Exception as e:
        print(f"Error at {node_key}: {e}")

ax_time.set_title("Timesignal model")
ax_time.set_xlabel("Time [s]")
ax_time.set_ylabel("Vy [m/s]")
ax_time.grid(True)
ax_time.set_ylim([-0.0015, 0.0015])
ax_time.legend(fontsize='small', loc='upper right')

# --- Subplot 2: amplitude spectrum ---
for node_key in node_keys:
    try:
        vel_dyn = np.array(data_dyn[node_key]['VELOCITY_Y'])
        N = len(vel_dyn)
        window = np.hamming(N)
        coords = data_dyn[node_key]['COORDINATES']

        fft_vals_win = rfft(vel_dyn * window)
        freqs = rfftfreq(N, d=dt)
        amp_win = 2.0/N * np.abs(fft_vals_win)

        ax_fft.plot(freqs, amp_win, linestyle="-",
                    label=f"{node_key} (X={coords[0]:.1f}, Y={coords[1]:.2f}, Y={coords[2]:.1f})")

    except Exception as e:
        print(f"Error at {node_key}: {e}")

ax_fft.set_title("Amplitude Spectrum – All nodes (Hamming)")
ax_fft.set_xlabel("Frequency [Hz]")
ax_fft.set_ylabel("Amplitude [m/s]")
ax_fft.grid(True, linestyle='--', linewidth=0.5)
ax_fft.set_xlim(0, 100)
ax_fft.set_ylim([0, 0.00025])
ax_fft.legend(fontsize='small', loc='upper right')

plt.tight_layout()
plt.show()





