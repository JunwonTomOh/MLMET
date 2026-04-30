#!/usr/bin/env python3
import h5py
import numpy as np
# import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from utils import *
import os


# def make_turnon(prj_name, model, turnon_list = ["TTToSemileptonic_238k_input", ], background = "SingleNeutrino_300k_input"):
#     Xr_bkg_puppi, Y_bkg = prepare_inputs(background)

#     px_puppi_bkg = -np.sum(Xr_bkg_puppi[1][:, :, 0], axis=1)
#     py_puppi_bkg = -np.sum(Xr_bkg_puppi[1][:, :, 1], axis=1)
#     puppi_bkg_pt = np.hypot(px_puppi_bkg, py_puppi_bkg)
#     ml_bkg_pt = predict_ml_pt(model, Xr_bkg_puppi)
    
#     bkg_list = [puppi_bkg_pt, ml_bkg_pt]
    
#     for signal in turnon_list:
#         Xr_sig, Yr_sig = prepare_inputs(signal)

#         true_sig_pt, _ = to_ptphi(Yr_sig)

#         px_puppi_sig = -np.sum(Xr_sig[1][:, :, 0], axis=1)  # shape [N]
#         py_puppi_sig = -np.sum(Xr_sig[1][:, :, 1], axis=1)  # shape [N]
#         puppi_sig_pt = np.hypot(px_puppi_sig, py_puppi_sig)

#         ml_sig_pt = predict_ml_pt(model, Xr_sig)

#         sig_list = [[true_sig_pt, puppi_sig_pt], [true_sig_pt, ml_sig_pt]]

#         save_turnon(bkg_list, sig_list, prj_name, signal)

def to_ptphi(xy):
    px, py = xy[:, 0], xy[:, 1]
    pt  = np.sqrt(px**2 + py**2)
    phi = np.arctan2(py, px)
    return pt, phi


def safe_percentile(arr, q):
    arr = arr[np.isfinite(arr)]
    return np.percentile(arr, q) if arr.size else np.nan


def predict_ml_pt(model, X):
    y = model.predict(X, verbose=0)  # expected [N,2]
    px, py = y[:, 0], y[:, 1]
    return np.sqrt(px**2 + py**2)


def save_turnon(bkg_list, sig_list, prj_name, signal_event):
    from scipy.stats import binned_statistic
    import matplotlib.pyplot as plt
    
    output_name = f"{prj_name}TurnOn_{signal_event}.png"
    trigger_rate_hz = 30e3     # 30 kHz
    l1_clock_hz     = 40e6     # 40 MHz
    percentile = 100 * (1.0 - trigger_rate_hz / l1_clock_hz)  # ~99.9925

    Case_list = ["puppi", "model"]

    if len(bkg_list) != len(sig_list):
        print("*** Number of Cases doesn't Match ***")
        return
    else:
        plt.figure(figsize=(7,7))

        for i in range(len(bkg_list)):
            centers, eff, thr, signal_pass = draw_turnon(bkg_list[i], sig_list[i][0], sig_list[i][1])
            plt.plot(centers, eff, ".-", lw=2, label=f"{Case_list[i]} (thr={thr:.1f} GeV)\n{signal_pass:.3f}")
            plt.axvline(thr)

        plt.xlabel("True MET [GeV]")
        plt.ylabel("Efficiency")
        plt.ylim(0, 1.05)
        plt.axhline(0.95)
        plt.grid(True, alpha=0.3)
        plt.title(f"Turn-on @ {trigger_rate_hz/1e3:.1f} kHz")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_name, dpi=200)
        plt.close()
        print(f"Saved → {output_name}")


def draw_turnon(bkg, true_sig, pred_sig):
    trigger_rate_hz = 30e3     # 30 kHz
    l1_clock_hz     = 40e6     # 40 MHz
    percentile = 100 * (1.0 - trigger_rate_hz / l1_clock_hz)  # ~99.9925

    nbins = 20
    truth_pt_bins = np.linspace(0, 500, nbins + 1)

    thr = safe_percentile(bkg, percentile)
    eff, _, _ = binned_statistic(true_sig, (pred_sig > thr).astype(float), statistic = "mean", bins = truth_pt_bins)
    centers = 0.5*(truth_pt_bins[1:] + truth_pt_bins[:-1])

    # signal_pass = np.mean(pred_sig > thr) * l1_clock_hz / np.shape(pred_sig)[0]
    signal_pass = np.sum(pred_sig > thr) / np.shape(pred_sig)[0]

    return centers, eff, thr, signal_pass

    
    
    