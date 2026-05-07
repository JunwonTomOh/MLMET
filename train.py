# import os
# import sys
# import shutil
# from pathlib import Path
# import argparse
# import yaml
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import optimizers
# from tensorflow.keras.callbacks import (
#     ReduceLROnPlateau,
#     ModelCheckpoint,
#     EarlyStopping,
#     CSVLogger,
# )
# from sklearn.model_selection import train_test_split
# from glob import glob

# Custom modules
from models import *
from utils import *
from loss import *
from train_utils import *


def get_callbacks(path_out, cfg):
    min_lr = cfg["callbacks"]["min_lr"]
    early_stopping_patience = cfg["callbacks"]["early_stopping_patience"]
    reduce_lr_patience = cfg["callbacks"]["reduce_lr_patience"]
    reduce_lr_factor = cfg["callbacks"]["reduce_lr_factor"]
    monitor = cfg["callbacks"]["monitor"]
    
    # early stopping callback
    early_stopping = EarlyStopping(monitor=monitor, patience=early_stopping_patience, verbose=1, restore_best_weights=True)

    csv_logger = CSVLogger(f'{path_out}/loss_history.log')

    # model checkpoint callback
    model_checkpoint = ModelCheckpoint(
        os.path.join(path_out, "model.keras"),
        monitor=monitor,
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode="auto",
        save_freq="epoch",
    )
    reduce_lr = ReduceLROnPlateau(monitor=monitor, factor=reduce_lr_factor, patience=reduce_lr_patience, min_lr=min_lr, cooldown=3, verbose=1)

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

    setup_gpu(cfg)

    Xorg, Y, Z = load_h5_inputs(path_in)

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

    optimizer = optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)
    # optimizer = tf.keras.optimizers.AdamW(
    #     learning_rate=1e-3,
    #     weight_decay=1e-4,
    #     clipnorm=1.0,
    # )
    keras_model.compile(
        loss=custom_loss,
        # loss="mean_squared_error",
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
