import json
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft, rfftfreq
import os

# Pad naar de JSON-bestanden
json_dyn_path = r"C:\Users\ritfeldis\Documents\remote\STEM\Holten_dir_length\json_output_stage_2.json"
json_quasi_path = r"C:\Users\ritfeldis\Documents\remote\STEM\Holten_dir_length\json_output.json"

# Inlezen van data
with open(json_dyn_path, 'r') as f:
    data_dyn = json.load(f)
with open(json_quasi_path, 'r') as f:
    data_quasi = json.load(f)

# Tijdstap en samplingfrequentie
time_dyn = np.array(data_dyn['TIME'])
time_quasi = np.array(data_quasi['TIME'])
Fs = 1.0 / np.mean(np.diff(time_dyn))
dt = 1 / Fs
end_time = 0.002  # nieuwe quasi-eindtijd
quasi_mask = time_quasi <= end_time

# Node keys ophalen en sorteren
node_keys = sorted([key for key in data_dyn if key.startswith("NODE_") and key in data_quasi])

# # Colormap voor onbeperkt aantal kleuren
# cmap = plt.get_cmap('tab20', len(node_keys))
# node_kleuren = {key: cmap(i) for i, key in enumerate(node_keys)}

# Outputmap
output_dir = r"C:\Users\ritfeldis\Documents\Python\STEM\output_node_plots"
os.makedirs(output_dir, exist_ok=True)

# Figuren voor gecombineerde plots
fig_all, (ax_time_all, ax_amp_all) = plt.subplots(2, 1, figsize=(16, 10), sharex=False)

# Doorloop alle nodes
for i, node_key in enumerate(node_keys):
    try:
        vel_quasi = np.array(data_quasi[node_key]['VELOCITY_Y'])[quasi_mask]
        vel_dyn = np.array(data_dyn[node_key]['VELOCITY_Y'])
        velocity_combined = np.concatenate([vel_quasi, vel_dyn])
        N = len(velocity_combined)
        time_combined = np.arange(N) * dt
        freqs = rfftfreq(N, d=dt)
        amp_spectrum = np.abs(rfft(velocity_combined)) / N
        # kleur = node_kleuren[node_key]

        # Individuele plots
        fig, (ax_time, ax_amp) = plt.subplots(2, 1, figsize=(14, 10), sharex=False)
        ax_time.plot(time_combined, velocity_combined, label=node_key)
        ax_time.axvline(x=end_time, color='gray', linestyle='--', label="Einde quasi-static")
        ax_time.set_title(f"Velocity Y over Time – {node_key}")
        ax_time.set_xlabel("Time [s]")
        ax_time.set_ylabel("Velocity Y")
        ax_time.grid(True)
        ax_time.legend()

        ax_amp.plot(freqs, amp_spectrum, label=node_key)
        ax_amp.set_title(f"Amplitude Spectrum – {node_key}")
        ax_amp.set_xlabel("Frequency [Hz]")
        ax_amp.set_ylabel("Amplitude [unit]")
        ax_amp.grid(True, linestyle='--', linewidth=0.5)
        ticks = np.arange(0, freqs.max() + 100, 100)
        ax_amp.set_xticks(ticks)
        ax_amp.legend(fontsize='small')
        plt.tight_layout()

        # Opslaan van figuur
        fig_path = os.path.join(output_dir, f"{node_key}_plots.png")
        plt.savefig(fig_path)
        plt.close()
        print(f"{node_key}: individuele plot opgeslagen")

        # Toevoegen aan gecombineerde figuren
        ax_time_all.plot(time_combined, velocity_combined, label=node_key)
        ax_amp_all.plot(freqs, amp_spectrum, label=node_key)

    except Exception as e:
        print(f"Fout bij {node_key}: {e}")

# Afwerking gecombineerde figuren
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


