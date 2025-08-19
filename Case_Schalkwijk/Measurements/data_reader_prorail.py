from typing import Dict

import matplotlib.pyplot as plt
import scipy

import pandas as pd
from pathlib import Path

import numpy as np
import datetime

from matplotlib import colors
from tqdm import tqdm

from mappers import ricardo_train_mapper


class Sensor:

    def __init__(self, traces: np.ndarray, fs: float, label: str, n_samples: int):
        self.traces = traces
        self.fs = fs
        self.label = label
        self.n_samples = n_samples

        self.time = np.arange(self.n_samples) * 1 / self.fs
        self.f_xx = None
        self.p_xx = None

    def welch(self, **kwargs):
        f_xx, p_xx = scipy.signal.welch(x=self.traces.T, fs=self.fs, **kwargs)
        self.f_xx = f_xx
        self.p_xx = p_xx


def readxmr(fname, aantalbits=24):
    H = {}

    with open(fname, 'rb') as fid:
        fid.seek(8)
        H['nsamp'] = int.from_bytes(fid.read(4), 'little')

        fid.seek(4)
        H['filetype'] = int.from_bytes(fid.read(1), 'little')

        fid.seek(54)
        H['fs'] = int.from_bytes(fid.read(2), 'little')
        H['nchan'] = int.from_bytes(fid.read(1), 'little')

        fid.seek(33)
        dum = int.from_bytes(fid.read(2), 'little')
        H['ntr'] = readdate(fid) + (dum / H['fs']) / (60 * 60 * 24)

        fid.seek(58)
        H['firm'] = int.from_bytes(fid.read(1), 'little')

        fid.seek(63)
        H['bit'] = int.from_bytes(fid.read(1), 'little')

        fid.seek(83)
        H['LSBx'] = int.from_bytes(fid.read(2), 'little') * 10 ** int.from_bytes(fid.read(1), 'little', signed=True)
        H['LSBy'] = int.from_bytes(fid.read(2), 'little') * 10 ** int.from_bytes(fid.read(1), 'little', signed=True)
        H['LSBz'] = int.from_bytes(fid.read(2), 'little') * 10 ** int.from_bytes(fid.read(1), 'little', signed=True)

        fid.seek(92)
        H['EUx'] = fid.read(5).decode('utf-8')
        H['EUy'] = fid.read(5).decode('utf-8')
        H['EUz'] = fid.read(5).decode('utf-8')

        fid.seek(107)
        H['namex'] = fid.read(3).decode('utf-8')
        H['namey'] = fid.read(3).decode('utf-8')
        H['namez'] = fid.read(3).decode('utf-8')

        fid.seek(116)
        H['trigX'] = int.from_bytes(fid.read(3), 'little') * H['LSBx']
        H['trigY'] = int.from_bytes(fid.read(3), 'little') * H['LSBy']
        H['trigZ'] = int.from_bytes(fid.read(3), 'little') * H['LSBz']

        fid.seek(143)
        H['nst'] = H['ntr'] - int.from_bytes(fid.read(1), 'little') / (60 * 60 * 24)

        fid.seek(145)
        H['comment'] = fid.read(30).decode('utf-8')

        # try:
        fid.seek(145)
        H['comment'] = fid.read(30).decode('utf-8')

        fid.seek(256)
        TRACE = np.array([read_int24(fid) for _ in range(H['nchan'] * H['nsamp'])])
        TRACE = TRACE.reshape((H['nsamp'], H['nchan'])) * np.array([H['LSBx'], H['LSBy'], H['LSBz']])

        return H, TRACE

        # except  Exception as e:
        #     print(e)
        #     return H


def readdate(fid):
    secs = bcd2str(int.from_bytes(fid.read(1), 'little'))
    mins = bcd2str(int.from_bytes(fid.read(1), 'little'))
    hour = bcd2str(int.from_bytes(fid.read(1), 'little'))
    days = bcd2str(int.from_bytes(fid.read(1), 'little'))
    mont = bcd2str(int.from_bytes(fid.read(1), 'little'))
    year = bcd2str(int.from_bytes(fid.read(1), 'little'))
    date_str = f'{days}-{mont}-20{year} {hour}:{mins}:{secs}'
    return datetime.datetime.strptime(date_str, '%d-%m-%Y %H:%M:%S').timestamp()


def bcd2str(bcd):
    bin_str = format(bcd, '08b')
    val1 = int(bin_str[:4], 2)
    val2 = int(bin_str[4:], 2)
    return f'{val1}{val2}'


def read_int24(fid):
    data = fid.read(3)
    value = int.from_bytes(data, byteorder='little', signed=True)
    return value


def extract_path_info(dataframe_passage: pd.Series, sensor_label: str):
    path_string = dataframe_passage["Tijdsignaal " + sensor_label]
    if isinstance(path_string, float):
        return None, None, None, None
    split_string = path_string.split("\\")
    _, _, _, _, year, month, day, filename = split_string
    return year, month, day, filename


def extract_ricardo_info(ricardo_data: pd.DataFrame):
    ricardo_speeds = []
    ricardo_vehicles = []
    for column in ricardo_data.columns[1:]:
        part_a, part_b = column.split("_")
        ricardo_vehicles.append(part_a)
        ricardo_speeds.append(int(part_b.split("kph")[0]))

    ricardo_speeds = list(set(ricardo_speeds))
    ricardo_vehicles = list(set(ricardo_vehicles))
    return ricardo_speeds, ricardo_vehicles


def plot_ricardo_data(ricardo_data, group_by="speed", subset1: None | list = None, subset2: None | list = None):
    ricardo_speeds, ricardo_vehicles = extract_ricardo_info(ricardo_data)

    if group_by == "speed":
        main_iterable = ricardo_speeds
        secondary_iterable = ricardo_vehicles
    else:
        main_iterable = ricardo_vehicles
        secondary_iterable = ricardo_speeds

    for iter1 in sorted(main_iterable):

        if subset1:
            if iter1 not in subset1:
                continue

        plt.figure(figsize=(8, 4))
        for iter2 in sorted(secondary_iterable):

            if subset2:
                if iter2 not in subset1:
                    continue
            try:
                if group_by == "speed":
                    yy = ricardo_data[f"{iter2}_{iter1}kph"].values
                else:
                    yy = ricardo_data[f"{iter1}_{iter2}kph"].values

            except KeyError as e:
                print(e)
                continue

            plt.loglog(ricardo_data["Frequency_Hz"], yy, label=iter2)

        if group_by == "speed":
            plt.title("Train speed: " + str(iter1) + " kmph")
        else:
            plt.title("Train type: " + str(iter1))

        plt.xlim([1, 100])
        plt.ylim([5, 3000])
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("|Q(f)| [N]")
        plt.grid(which="both", axis="both", alpha=0.2, c="k")
        plt.legend(bbox_to_anchor=(1.05, 0.5), loc="center left")
        plt.tight_layout()


def read_meetboek(path_meetboek: str):
    meetboek_passages = pd.read_excel(path_meetboek, sheet_name="passages")

    # adjust labels of the GT

    GT_group = ["GT Erts/kolen",
                "GT Ketel",
                "GT Container",
                "GT Bont"
                ]

    meetboek_passages.loc[meetboek_passages["Treintype"].isin(GT_group), "Treintype"] = "GT"

    return meetboek_passages


def filter_passages(meetboek: pd.DataFrame, train_type: str = None, min_speed: float = 0, max_speed: float = np.inf):
    """Create a  subset of passages the meetboek for the given train type and within certain passage speeds.

    Args:
        meetboek (pd.DataFrame): DataFrame containing the passages and the corresponding path to the file per sensor.
        train_type (str): train type of interest.
        min_speed (float): minimum passage speed (km/h).
        max_speed (float): maximum passage speed (km/h).

    Returns:
        subset (pd.DataFrame): subset of passages the meetboek for the given train type and within certain passage speeds
    """
    subset = meetboek.copy()
    if train_type is not None:
        subset = subset[subset["Treintype"] == train_type]

    if train_type is not None:
        subset = subset[(subset["Rijsnelheid [km/u]"] <= max_speed) & (subset["Rijsnelheid [km/u]"] >= min_speed)]

    return subset


def get_sensor_data_from_passages(location_data: str, subset_passages: pd.DataFrame, sensor: str) -> Dict[str, Sensor]:
    """Reads the sensor data of the subset of passages extracted from the meetboek for the given sensor.

    Args:
        subset_passages (pd.DataFrame): filtered meetboek for the required passages (can also be the full meetboek).
        sensor (str): sensor label

    Returns:
        sensor_objects (dict): dictionary with row index and corresponding sensor object for the specified sensor
    """

    sensor_objects: Dict[str, Sensor] = {}

    for index, passage_row in tqdm(subset_passages.iterrows()):
        year, month, day, filename = extract_path_info(passage_row, sensor)
        path = Path(location_data) / str(sensor) / str(sensor) / "events" / str(year) / str(month) / str(day) / filename

        print(f"\n[INFO] Index: {index}")
        print(f"[INFO] Bestandspad: {path}")
        print(f"[INFO] Bestandsnaam: {filename}")

        header, traces = readxmr(path)
        sensor_objects[str(index)] = Sensor(
            label=sensor, fs=header['fs'], traces=traces, n_samples=header['nsamp']
        )

    return sensor_objects


def get_ricardo_columns(ricardo_database: pd.DataFrame, train_type: str = None, min_speed: float = 0,
                        max_speed: float = np.inf):
    """Create a  subset of passages the meetboek for the given train type and within certain passage speeds.

    Args:
        ricardo_database (pd.DataFrame): DataFrame containing the PSD from Ricardo.
        train_type (str): train type of interest.
        min_speed (float): minimum passage speed (km/h).
        max_speed (float): maximum passage speed (km/h).

    Returns:
        subset (pd.DataFrame): subset of PSD related to the current analysis
    """

    ricardo_speeds, ricardo_vehicles = extract_ricardo_info(ricardo_data=ricardo_database)

    subset_speeds = np.array(ricardo_speeds)
    subset_speeds = subset_speeds[np.logical_and(subset_speeds >= min_speed, subset_speeds <= max_speed)]

    train_types_of_interest = ricardo_train_mapper[train_type]

    all_columns = []

    for speed in subset_speeds:
        for train_ricardo in train_types_of_interest:
            column = f"{train_ricardo}_{speed:.0f}kph"
            if column in ricardo_database.columns:
                all_columns.append(column)

    freq = ricardo_database["Frequency_Hz"]

    subset_dataset = ricardo_database[all_columns].values
    return freq, all_columns, subset_dataset


if __name__ == "__main__":

    location_data = r"C:\Users\ritfeldis\OneDrive - TNO\Team - ProRail - Analyse Trillingspectra\Work\prorail data\VoorProRail\VoorProRail\RBX files"
    path_meetboek = r"C:\Users\ritfeldis\OneDrive - TNO\Team - ProRail - Analyse Trillingspectra\Work\prorail data\VoorProRail\VoorProRail\Meetjournaal trillingsonderzoek Schalkwijk definitief 20190107 DM adjusted.xlsx"
    path_ricardo_data = r"Export_PSD_Schalkwijk.csv"

    meetboek_passages = pd.read_excel(path_meetboek, sheet_name="passages")

    ricardo_data = pd.read_csv(path_ricardo_data)
    print([col for col in ricardo_data.columns if "falns" in col.lower()])
    ricardo_speeds, ricardo_vehicles = extract_ricardo_info(ricardo_data)

    # plot_ricardo_data(ricardo_data, group_by="speed")
    # plt.show()

    subset_trains = ["VIRMmBvk12Leeg", "SLT4baks", "Traxx", "FalnsY25MaxBasis"]
    subset_speeds_others = ["100", "120", "140"]

    subset_speeds_falns = ["80"]

    ricardo_speeds, ricardo_vehicles = extract_ricardo_info(ricardo_data)

    fig, axs = plt.subplots(len(subset_trains), 1, figsize=(7, 8), sharex="all", sharey="all")

    for ix, iter1 in enumerate(subset_trains):

        subset_speeds = subset_speeds_others
        if "Falns" in iter1:
            subset_speeds = subset_speeds_falns

        # plt.figure(figsize=(8, 4))
        for iter2 in sorted(subset_speeds):

            # if subset_speeds:
            #     if iter2 not in subset_speeds:
            #         continue
            try:
                yy = ricardo_data[f"{iter1}_{iter2}kph"].values

            except KeyError as e:
                print(e)
                continue

            fig.axes[ix].loglog(ricardo_data["Frequency_Hz"], yy, label=iter2 + " km/h")

        fig.axes[ix].set_title(iter1)
        fig.axes[ix].set_xlim([1, 100])
        fig.axes[ix].set_ylim([30, 3000])
        fig.axes[ix].set_xlabel("Frequency [Hz]")
        fig.axes[ix].set_ylabel("|Q(f)| [N]")
        fig.axes[ix].grid(which="both", axis="both", alpha=0.2, c="k")
        fig.axes[ix].legend(bbox_to_anchor=(1.05, 0.5), loc="center left")
    fig.tight_layout()

    fig, axs = plt.subplots(len(subset_speeds_falns + subset_speeds_others), 1, figsize=(7, 8), sharex="all",
                            sharey="all")

    for ix, iter1 in enumerate(subset_speeds_falns + subset_speeds_others):

        # plt.figure(figsize=(8, 4))
        for iter2 in subset_trains:

            # if subset_speeds:
            #     if iter2 not in subset_speeds:
            #         continue
            try:
                yy = ricardo_data[f"{iter2}_{iter1}kph"].values

            except KeyError as e:
                print(e)
                continue

            fig.axes[ix].loglog(ricardo_data["Frequency_Hz"], yy, label=iter2)

        fig.axes[ix].set_title(iter1 + " km/h")
        fig.axes[ix].set_xlim([1, 100])
        fig.axes[ix].set_ylim([30, 3000])
        fig.axes[ix].set_xlabel("Frequency [Hz]")
        fig.axes[ix].set_ylabel("|Q(f)| [N]")
        fig.axes[ix].grid(which="both", axis="both", alpha=0.2, c="k")
        fig.axes[ix].legend(bbox_to_anchor=(1.05, 0.5), loc="center left")
    fig.tight_layout()

    plt.show()

    sensors = ["MT30", "MT31", "MT32"]

    passage_id = 126
    passage_row = meetboek_passages.loc[
        meetboek_passages[meetboek_passages["Passage ID"] == passage_id].index.values[0]
    ]
    # passage_row = meetboek_passages.loc[
    #         meetboek_passages["Rijsnelheid [km/u]"].argmax()
    # ]

    train_label, passage_date, train_speed = passage_row[["Treintype", "Datum", "Rijsnelheid [km/u]"]]

    sensor_objects = []

    for sensor in sensors:
        year, month, day, filename = extract_path_info(passage_row, sensor)
        path = Path(location_data) / str(sensor) / str(sensor) / "events" / str(year) / str(month) / str(day) / filename
        header, traces = readxmr(path)
        sensor_objects.append(Sensor(label=sensor, fs=header['fs'], traces=traces, n_samples=header['nsamp']))

    # files = list(folder.glob("*.XMR"))
    # mt30\events\2018\10\11\18284092.XMR

    # path = files[0]
    # try obspy to read

    # convert the matlab file into python

    fig, axs = plt.subplots(3, 2, sharey="col", sharex="col")
    fig2, axs2 = plt.subplots(3, 1, figsize=(6, 6), sharex="col", sharey="col")
    nfft = 2 ** 14

    for ix, ss in enumerate(sensor_objects):
        axs[ix, 0].plot(ss.time, ss.traces[:, -1])
        # f_xx, p_xx = scipy.signal.periodogram(x=ss.traces.T, fs=ss.fs, scaling="density", nfft=2**14,)
        f_xx, p_xx = scipy.signal.welch(x=ss.traces[:, -1].T, fs=ss.fs, scaling="density", nfft=2 ** 14,
                                        nperseg=2 ** 12)
        axs[ix, 1].plot(f_xx, p_xx.T)

        # f_sg, t_sg, S_xx = scipy.signal.spectrogram(x=ss.traces[:, -1],
        #                                             fs=ss.fs,
        #                                             nperseg=512,
        #                                             noverlap=128,
        #                                             window="hamming",
        #                                             scaling="density",
        #                                             nfft=nfft
        #                                             )

        f_sg, t_sg, S_xx = scipy.signal.stft(x=ss.traces[:, -1],
                                             fs=ss.fs,
                                             nperseg=512,
                                             noverlap=128,
                                             scaling="psd",
                                             nfft=int(nfft / 2)
                                             )

        mask_f = f_sg <= 100

        abs_Sxx = np.abs(S_xx[mask_f, :])
        # norm=colors.LogNorm(vmin=abs_Sxx.min(), vmax=abs_Sxx.max())
        norm = colors.LogNorm(vmin=1e-7, vmax=abs_Sxx.max())

        im = axs2[ix].pcolormesh(t_sg, f_sg, np.abs(S_xx), norm=norm, shading='gouraud')
        axs2[ix].set_ylabel('Frequency [Hz]')
        axs2[ix].set_xlabel('Time [sec]')
        axs2[ix].set_ylim([1, 100])
        # axs2[ix].set_yscale("log")

    fig.suptitle(
        f"Passage {passage_row['Passage ID']} | {passage_row.Treintype} |{passage_row['Rijsnelheid [km/u]']:.0f} km/h \n {passage_row.Datum}")
    fig2.suptitle(
        f"Passage {passage_row['Passage ID']} | {passage_row.Treintype} |{passage_row['Rijsnelheid [km/u]']:.0f} km/h \n {passage_row.Datum}")

    fig.tight_layout()
    fig2.tight_layout()

    fig2.subplots_adjust(right=0.7)
    cbar_ax = fig2.add_axes([0.75, 0.15, 0.035, 0.7])
    fig2.colorbar(im, cax=cbar_ax)
    cbar_ax.set_ylabel("$V_Z$ [mm/s/HZ]")
    # fig2.tight_layout()

    plt.show()