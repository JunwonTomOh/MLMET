import os
import sys
import shutil
from pathlib import Path
import argparse
import yaml
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    ModelCheckpoint,
    EarlyStopping,
    CSVLogger,
)
from sklearn.model_selection import train_test_split
from glob import glob

from utils import *


def load_h5_inputs(input_cfg):
    X_list = []
    Y_list = []
    Z_list = []

    for item in input_cfg:
        # old style support:
        # - path/to/file.h5
        if isinstance(item, str):
            path = item
            max_events = -1

        # new style:
        # - path: path/to/file.h5
        #   max_events: 50000
        elif isinstance(item, dict):
            path = item["path"]
            max_events = item.get("max_events", -1)

        else:
            raise TypeError(f"Invalid input config item: {item}")

        files = []
        if path.endswith(".h5"):
            files.append(path)
        else:
            files += glob(os.path.join(path, "*.h5"))

        if len(files) == 0:
            raise RuntimeError(f"No h5 files found from input: {path}")
        print("=" * 80)
        print(f"Loading input: {path}")
        print(f"  files: {len(files)}")
        print(f"  max_events: {max_events}")

        X, Y, Z, REF = read_input(files)

        if max_events is not None and max_events > 0:
            X = X[:max_events]
            Y = Y[:max_events]
            Z = Z[:max_events]

        print(f"  loaded events: {len(Y)}")
        print("=" * 80)

        X_list.append(X)
        Y_list.append(Y)
        Z_list.append(Z)

    X_all = np.concatenate(X_list, axis=0)
    Y_all = np.concatenate(Y_list, axis=0)
    Z_all = np.concatenate(Z_list, axis=0)

    return X_all, Y_all, Z_all


def setup_gpu(cfg):
    gpu_cfg = cfg.get("gpu", {})
    use_gpu = gpu_cfg.get("use_gpu", False)

    if not use_gpu:
        # This works only if TensorFlow has not initialized GPUs yet.
        try:
            tf.config.set_visible_devices([], "GPU")
            print("GPU disabled. Using CPU.")
        except RuntimeError as e:
            print("Could not disable GPU because TensorFlow is already initialized.")
            print(e)
        return

    gpu_index = gpu_cfg.get("gpu_index", 0)
    memory_growth = gpu_cfg.get("memory_growth", True)

    print("Built with CUDA:", tf.test.is_built_with_cuda())
    print("Physical GPUs:", tf.config.list_physical_devices("GPU"))

    gpus = tf.config.list_physical_devices("GPU")

    if len(gpus) == 0:
        print("No GPU found. Using CPU.")
        return

    if gpu_index >= len(gpus):
        raise RuntimeError(
            f"Requested gpu_index={gpu_index}, but only {len(gpus)} GPU(s) found."
        )

    tf.config.set_visible_devices(gpus[gpu_index], "GPU")

    if memory_growth:
        tf.config.experimental.set_memory_growth(gpus[gpu_index], True)

    print(f"Using GPU index: {gpu_index}")

def quantize_weight_np(w, total_bits, integer_bits):
    frac_bits = total_bits - integer_bits
    scale = 2 ** frac_bits

    qmin = -(2 ** (total_bits - 1))
    qmax = 2 ** (total_bits - 1) - 1

    q = np.round(w * scale)
    q = np.clip(q, qmin, qmax)

    w_q = q / scale

    return w_q.astype(np.float32), q.astype(np.int32)


def save_model_weights_txt(model, path_out, cfg):
    weight_dir = os.path.join(path_out, "weights_txt")
    quant_dir = os.path.join(path_out, "weights_txt_quantized")

    os.makedirs(weight_dir, exist_ok=True)
    os.makedirs(quant_dir, exist_ok=True)

    quant_cfg = cfg.get("quantization", {})
    gamma_cfg = quant_cfg.get("gamma", {})
    beta_cfg = quant_cfg.get("beta", {})

    qmap = {
        "gamma": {
            "total_bits": gamma_cfg.get("total_bits", 9),
            "integer_bits": gamma_cfg.get("integer_bits", 2),
        },
        "beta": {
            "total_bits": beta_cfg.get("total_bits", 9),
            "integer_bits": beta_cfg.get("integer_bits", 4),
        },
    }

    for layer in model.layers:
        weights = layer.get_weights()

        if len(weights) == 0:
            continue

        for i, w in enumerate(weights):
            # 1. Save original float weight
            if w.ndim <= 2:
                filename = os.path.join(weight_dir, f"{layer.name}_weight_{i}.txt")
                np.savetxt(filename, w, fmt="%.9f")
            else:
                filename = os.path.join(weight_dir, f"{layer.name}_weight_{i}_flat.txt")
                np.savetxt(filename, w.reshape(-1), fmt="%.9f")

            print(f"Saved float weight txt: {filename}")

            # 2. Save quantized gamma/beta weights
            if layer.name in qmap:
                total_bits = qmap[layer.name]["total_bits"]
                integer_bits = qmap[layer.name]["integer_bits"]

                w_q, w_int = quantize_weight_np(
                    w,
                    total_bits=total_bits,
                    integer_bits=integer_bits,
                )

                q_float_path = os.path.join(
                    quant_dir,
                    f"{layer.name}_apfixed_{total_bits}_{integer_bits}_float.txt",
                )

                q_int_path = os.path.join(
                    quant_dir,
                    f"{layer.name}_apfixed_{total_bits}_{integer_bits}_int.txt",
                )

                if w_q.ndim <= 2:
                    np.savetxt(q_float_path, w_q, fmt="%.9f")
                    np.savetxt(q_int_path, w_int, fmt="%d")
                else:
                    np.savetxt(q_float_path, w_q.reshape(-1), fmt="%.9f")
                    np.savetxt(q_int_path, w_int.reshape(-1), fmt="%d")

                print(f"Saved quantized float weight txt: {q_float_path}")
                print(f"Saved quantized int weight txt:   {q_int_path}")
                print(
                    f"{layer.name}: "
                    f"float range=({np.min(w):.6f}, {np.max(w):.6f}), "
                    f"quant range=({np.min(w_q):.6f}, {np.max(w_q):.6f})"
                )


def save_code_snapshot(path_out):
    code_dir = os.path.join(path_out, "code_snapshot")
    os.makedirs(code_dir, exist_ok=True)

    files_to_save = [
        sys.argv[0],
        "models.py",
        "loss.py",
        "utils.py",
        # "train_baselineModel.py",
        "Write_MET_binned_histogram.py",
        # "cyclical_learning_rate.py",
        "draw_turnon.py",
    ]

    for file_path in files_to_save:
        if os.path.exists(file_path):
            dst = os.path.join(code_dir, Path(file_path).name)
            shutil.copy2(file_path, dst)
            print(f"Saved code snapshot: {dst}")
        else:
            print(f"Skipped missing file: {file_path}")


def save_config_snapshot(cfg, path_out):
    config_path = cfg.get("_config_path", None)

    if config_path is None:
        return

    dst = os.path.join(path_out, "config.yml")
    shutil.copy2(config_path, dst)
    print(f"Saved config snapshot: {dst}")