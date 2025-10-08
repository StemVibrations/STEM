import pickle
import numpy as np
import matplotlib.pyplot as plt

# ===== Data laden =====
with open("D:/VIRM.pkl", "rb") as f:
    data = pickle.load(f)

# ===== Meetpunten en tijdstip selecteren =====
meetpunten = [
    'Meetjournal_MP8_Holten_zuid_4m_C',
    'Meetjournal_MP2_Holten_zuid_25m_C'
]
tijdstip_fft = '2024-09-10 14:16:06'

# ===== Vaste limieten instellen =====
y_lim_time = [-0.006, 0.006]  # snelheid [m/s]
y_lim_fft = [0, 0.00025]      # amplitude FFT [m/s]
x_lim_fft = [1, 500]          # frequentie [Hz]

# ===== Figuur met 2 subplots =====
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# ===== Loop over meetpunten =====
for meetpunt in meetpunten:
    time = data[meetpunt][tijdstip_fft]['data']['TIME']
    trace_z = np.array(data[meetpunt][tijdstip_fft]['data']['TRACE_Z']) / 1000.0  # naar m/s

    n = len(trace_z)
    dt = time[1] - time[0]

    # ===== Hamming window =====
    window = np.hamming(n)
    trace_win = trace_z * window
    U = np.sum(window) / n
    fft_vals_win = np.fft.fft(trace_win)
    fft_freq = np.fft.fftfreq(n, dt)
    pos_mask = fft_freq > 0
    fft_freq = fft_freq[pos_mask]
    fft_ampl_win = np.abs(fft_vals_win[pos_mask]) * 2 / (n * U)

    # Tijdsignaal
    ax1.plot(time, trace_win, label=meetpunt)

    # FFT-spectrum
    ax2.plot(fft_freq, fft_ampl_win, label=meetpunt)

# ===== Opmaak tijdsignaal =====
ax1.set_xlabel("Tijd [s]")
ax1.set_ylabel("Vz [m/s]")
ax1.set_title(f"Tijdsignaal (met Hamming) - {tijdstip_fft}")
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.set_ylim(y_lim_time)
ax1.legend()

# ===== Opmaak FFT =====
ax2.set_xlabel("Frequentie [Hz]")
ax2.set_ylabel("Amplitude [m/s]")
ax2.set_title("FFT-spectrum (met Hamming)")
ax2.grid(True, linestyle="--", alpha=0.6, which="both")
ax2.set_ylim(y_lim_fft)
ax2.set_xscale("log")
ax2.set_xlim(x_lim_fft)
ax2.legend()

plt.tight_layout()
plt.show()







