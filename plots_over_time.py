import json
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft, rfftfreq
import os

json_dyn_path = r"C:\Users\ritfeldis\Documents\Python\STEM\stiff_files_dir\json_output_stage_2.json"
json_quasi_path = r"C:\Users\ritfeldis\Documents\Python\STEM\stiff_files_dir\json_output.json"

with open(json_dyn_path, 'r') as f:
    data_dyn = json.load(f)
with open(json_quasi_path, 'r') as f:
    data_quasi = json.load(f)

time_dyn = np.array(data_dyn['TIME'])
time_quasi = np.array(data_quasi['TIME'])
Fs = 1.0 / np.mean(np.diff(time_dyn))
dt = 1 / Fs
end_time = 0.002  # nieuwe quasi-eindtijd
quasi_mask = time_quasi <= end_time

node_keys = [key for key in data_dyn if key.startswith("NODE_") and key in data_quasi]
node_coords = {key: data_dyn[key]['COORDINATES'] for key in node_keys}
node_keys_sorted = sorted(node_keys, key=lambda k: data_dyn[k]['COORDINATES'][0])  # verste knoop laatst

kleuren = ['tab:blue', 'tab:orange', 'tab:green']
node_kleuren = {}
for i, key in enumerate(node_keys_sorted):
    node_kleuren[key] = kleuren[i] if i < len(kleuren) else None

output_dir = r"C:\Users\ritfeldis\Documents\Python\STEM\output_node_plots"
os.makedirs(output_dir, exist_ok=True)

fig_all, (ax_time_all, ax_amp_all) = plt.subplots(2, 1, figsize=(16, 10), sharex=False)

for node_key in node_keys_sorted:
    try:
        vel_quasi = np.array(data_quasi[node_key]['VELOCITY_Y'])[quasi_mask]
        vel_dyn = np.array(data_dyn[node_key]['VELOCITY_Y'])
        velocity_combined = np.concatenate([vel_quasi, vel_dyn])
        N = len(velocity_combined)
        time_combined = np.arange(N) * dt
        coords = tuple(round(c, 2) for c in data_dyn[node_key]['COORDINATES'])
        freqs = rfftfreq(N, d=dt)
        amp_spectrum = np.abs(rfft(velocity_combined)) / N
        kleur = node_kleuren[node_key]

        fig, (ax_time, ax_amp) = plt.subplots(2, 1, figsize=(14, 10), sharex=False)
        ax_time.plot(time_combined, velocity_combined, label=f"{node_key} {coords}", color=kleur)
        ax_time.axvline(x=end_time, color='gray', linestyle='--', label="Einde quasi-static")
        ax_time.set_title(f"Velocity Y over Time – {node_key}")
        ax_time.set_xlabel("Time [s]")
        ax_time.set_ylabel("Velocity Y")
        ax_time.grid(True)
        ax_time.legend()

        ax_amp.plot(freqs, amp_spectrum, label=f"{node_key} {coords}", color=kleur)
        ax_amp.set_title(f"Amplitude Spectrum – {node_key}")
        ax_amp.set_xlabel("Frequency [Hz]")
        ax_amp.set_ylabel("Amplitude [unit]")
        ax_amp.grid(True, linestyle='--', linewidth=0.5)
        ticks = np.arange(0, freqs.max() + 100, 100)
        ax_amp.set_xticks(ticks)
        ax_amp.legend(fontsize='small')
        plt.tight_layout()

        fig_path = os.path.join(output_dir, f"{node_key}_plots.png")
        plt.savefig(fig_path)
        plt.close()
        print(f"{node_key}: individuele plot opgeslagen")

        ax_time_all.plot(time_combined, velocity_combined, label=f"{node_key} {coords}", color=kleur)
        ax_amp_all.plot(freqs, amp_spectrum, label=f"{node_key} {coords}", color=kleur)

    except Exception as e:
        print(f"Fout bij {node_key}: {e}")

ax_time_all.axvline(x=end_time, color='gray', linestyle='--', label="Einde quasi-static")
ax_time_all.set_title("Velocity Y over Time – Alle nodes")
ax_time_all.set_xlabel("Time [s]")
ax_time_all.set_ylabel("Velocity Y")
ax_time_all.grid(True)
ax_time_all.legend(fontsize='small', loc='upper right')

ax_amp_all.set_title("Amplitude Spectrum – Alle nodes")
ax_amp_all.set_xlabel("Frequency [Hz]")
ax_amp_all.set_ylabel("Amplitude [unit]")
ax_amp_all.grid(True, linestyle='--', linewidth=0.5)
max_freq = max(line.get_xdata().max() for line in ax_amp_all.lines)
ax_amp_all.set_xticks(np.arange(0, max_freq + 100, 100))
ax_amp_all.legend(fontsize='small', loc='upper right')

plt.tight_layout()
plt.show()


