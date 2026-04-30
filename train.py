import os
import sys
import shutil
from pathlib import Path
import argparse
import yaml
import numpy as np
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

# Custom modules
from models import *
from utils import *
from loss import *


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


def get_callbacks(path_out, cfg):
    min_lr = cfg["callbacks"]["min_lr"]
    early_stopping_patience = cfg["callbacks"]["early_stopping_patience"]
    reduce_lr_patience = cfg["callbacks"]["reduce_lr_patience"]
    reduce_lr_factor = cfg["callbacks"]["reduce_lr_factor"]
    
    # early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping_patience, verbose=1, restore_best_weights=True)

    csv_logger = CSVLogger(f'{path_out}/loss_history.log')

    # model checkpoint callback
    model_checkpoint = ModelCheckpoint(
        os.path.join(path_out, "model.keras"),
        monitor="val_loss",
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode="auto",
        save_freq="epoch",
    )
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=reduce_lr_factor, patience=reduce_lr_patience, min_lr=min_lr, cooldown=3, verbose=1)

    lr_scale = 1.
    # clr = CyclicLR(base_lr=0.0003*lr_scale, max_lr=0.001*lr_scale, step_size=sample_size/batch_size, mode='triangular2')

    stop_on_nan = tf.keras.callbacks.TerminateOnNaN()

    # callbacks = [early_stopping, clr, stop_on_nan, csv_logger, model_checkpoint]
    callbacks = [early_stopping, reduce_lr, stop_on_nan, csv_logger, model_checkpoint]

    return callbacks
        

def train(cfg):
    custom_loss = hybrid_vector_recoil_loss()

    epochs = cfg["training"]["epochs"]
    batch_size = cfg["training"]["batch_size"]
    path_in = cfg["training"]["input"]
    path_out = cfg["training"]["output"]
    normFac = cfg["training"]["norm_factor"]

    os.makedirs(path_out, exist_ok=True)

    # Save exact training code before training starts
    save_code_snapshot(path_out)
    save_config_snapshot(cfg, path_out)

    # gpu_index = cfg["gpu"]["gpu_index"]
    # print("Built with CUDA:", tf.test.is_built_with_cuda())
    # print("Physical GPUs:", tf.config.list_physical_devices("GPU"))
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 0:
        tf.config.set_visible_devices(gpus[gpu_index], "GPU")
        tf.config.experimental.set_memory_growth(gpus[gpu_index], True)
        print(f"Using GPU index: {gpu_index}")
    else:
        print("No GPU found. Using CPU.")

    h5files = []
    for p in path_in:
        if p.endswith(".h5"):
            h5files.append(p)
        else:
            h5files += glob(os.path.join(p, "*.h5"))

    if len(h5files) == 0:
        raise RuntimeError("No input h5 files found.")

    print("Input h5 files:")
    for f in h5files:
        print(f"  {f}")

    Xorg, Y, Z = read_input(h5files)

    X_features, X_pxpy = preProcessingForTinyTableModel(Xorg, normFac)

    Xr = [X_features, X_pxpy]
    Yr = Y[:, 0:2] / normFac

    indices = np.arange(len(Yr))
    indices_train, indices_test = train_test_split(indices, test_size=1.0 / 7.0, random_state=7)
    indices_train, indices_valid = train_test_split(indices_train, test_size=1.0 / 6.0, random_state=7)

    print(f"→ Total samples: {len(Yr)}")
    print(f"→ train samples: {len(indices_train)}")
    print(f"→ valid samples: {len(indices_valid)}")
    print(f"→ test  samples: {len(indices_test)}")

    Xr_train = [x[indices_train] for x in Xr]
    Xr_valid = [x[indices_valid] for x in Xr]

    Yr_train = Yr[indices_train]
    Yr_valid = Yr[indices_valid]

    n_features = int(np.shape(X_features)[2])

    quant_cfg = cfg.get("quantization", {})
    gamma_cfg = quant_cfg.get("gamma", {})
    beta_cfg = quant_cfg.get("beta", {})
    
    n_PID = cfg["model_cfg"]["NUM_PID"]
    n_ETA = cfg["model_cfg"]["NUM_ETA"]
    n_PT = cfg["model_cfg"]["NUM_PT"]

    keras_model = TinyTableModel(
        NUM_Features=n_features,
        NUM_PID=n_PID,
        NUM_ETA=n_ETA,
        NUM_PT=n_PT,

        quantize_embedding=quant_cfg.get("enable", False),

        gamma_total_bits=gamma_cfg.get("total_bits", 9),
        gamma_integer_bits=gamma_cfg.get("integer_bits", 2),

        beta_total_bits=beta_cfg.get("total_bits", 9),
        beta_integer_bits=beta_cfg.get("integer_bits", 4),
    )

    optimizer = optimizers.Adam(learning_rate=1e-2, clipnorm=1.0)

    keras_model.compile(
        loss=custom_loss,
        optimizer=optimizer,
        metrics=["mean_absolute_error", "mean_squared_error"],
    )

    print(keras_model.summary())

    history = keras_model.fit(
        Xr_train,
        Yr_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_data=(Xr_valid, Yr_valid),
        callbacks=get_callbacks(path_out, cfg),
    )

    # At this point, EarlyStopping restored the best weights.
    save_model_weights_txt(keras_model, path_out, cfg)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="path to training config yaml file"
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    model = cfg["training"].get("model", "TinyTables")
    output = cfg["training"]["output"]
    cfg["_config_path"] = args.config

    os.makedirs(output, exist_ok=True)

    if model == "TinyTables":
        train(cfg)
    elif model == "QDeepSet":
        # train_loadAllData(cfg)
        pass
    else:
        raise ValueError(f"Unknown model: {model}")


if __name__ == "__main__":
    main()
