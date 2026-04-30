import tensorflow
import tensorflow.keras.backend as K
from tensorflow.keras import optimizers, initializers
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

import numpy as np
import tables
import matplotlib.pyplot as plt
import argparse
import math
#import setGPU
import time
import os
import pathlib
import datetime
import tqdm
import h5py
from glob import glob
import itertools

# Import custom modules
from draw_turnon import *
from models import *
from utils import *
from loss import *
# from DataGenerator import DataGenerator

import matplotlib.pyplot as plt
import mplhep as hep


def prepare_inputs(event_name):
    event_file = [f"./{event_name}.h5"]

    print("Loading Files...")
    X, Y = read_input(event_file)
    X1, X2, X3, X4 = preProcessing(X, 1)
    
    Xc = [X3, X4]
    Xr = [X1, X2] + Xc
    Yr = Y
    return Xr, Yr


def make_turnon(prj_name, model, turnon_list = ["TTToSemileptonic_238k_input", ], background = "SingleNeutrino_300k_input"):
    Xr_bkg_puppi, Y_bkg = prepare_inputs(background)

    px_puppi_bkg = -np.sum(Xr_bkg_puppi[1][:, :, 0], axis=1)
    py_puppi_bkg = -np.sum(Xr_bkg_puppi[1][:, :, 1], axis=1)
    puppi_bkg_pt = np.hypot(px_puppi_bkg, py_puppi_bkg)
    ml_bkg_pt = predict_ml_pt(model, Xr_bkg_puppi)
    
    bkg_list = [puppi_bkg_pt, ml_bkg_pt]
    
    for signal in turnon_list:
        Xr_sig, Yr_sig = prepare_inputs(signal)

        true_sig_pt, _ = to_ptphi(Yr_sig)

        px_puppi_sig = -np.sum(Xr_sig[1][:, :, 0], axis=1)  # shape [N]
        py_puppi_sig = -np.sum(Xr_sig[1][:, :, 1], axis=1)  # shape [N]
        puppi_sig_pt = np.hypot(px_puppi_sig, py_puppi_sig)

        ml_sig_pt = predict_ml_pt(model, Xr_sig)

        sig_list = [[true_sig_pt, puppi_sig_pt], [true_sig_pt, ml_sig_pt]]

        save_turnon(bkg_list, sig_list, prj_name, signal)


def get_callbacks(path_out, sample_size, batch_size):
    # early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=80, verbose=1, restore_best_weights=False)

    csv_logger = CSVLogger(f'{path_out}loss_history.log')

    # model checkpoint callback
    # this saves our model architecture + parameters into model.h5
    model_checkpoint = ModelCheckpoint(f'{path_out}model.h5', monitor='val_loss',
                                       verbose=0, save_best_only=True,
                                       save_weights_only=False, mode='auto',
                                       period=1)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7, cooldown=3, verbose=1)

    lr_scale = 1.
    # clr = CyclicLR(base_lr=0.0003*lr_scale, max_lr=0.001*lr_scale, step_size=sample_size/batch_size, mode='triangular2')

    stop_on_nan = tensorflow.keras.callbacks.TerminateOnNaN()

    # callbacks = [early_stopping, clr, stop_on_nan, csv_logger, model_checkpoint]
    callbacks = [early_stopping, reduce_lr, stop_on_nan, csv_logger, model_checkpoint]

    return callbacks


def test(Yr_test, predict_test, PUPPI_pt, path_out):

    MakePlots(Yr_test, predict_test, PUPPI_pt, path_out=path_out)

    Yr_test = convertXY2PtPhi(Yr_test)
    predict_test = convertXY2PtPhi(predict_test)
    PUPPI_pt = convertXY2PtPhi(PUPPI_pt)

    event = 'TTToSemileptonic'

    extract_result(predict_test, Yr_test, path_out, event, 'ML')
    extract_result(PUPPI_pt, Yr_test, path_out, event, 'PU')

    MET_rel_error_opaque(predict_test[:, 0], PUPPI_pt[:, 0], Yr_test[:, 0], name=''+path_out+'/rel_error_opaque.png')
    MET_binned_predict_mean_opaque(predict_test[:, 0], PUPPI_pt[:, 0], Yr_test[:, 0], 20, 0, 500, 0, '.', name=''+path_out+'/PrVSGen.png')

    Phi_abs_error_opaque(PUPPI_pt[:, 1], predict_test[:, 1], Yr_test[:, 1], name=path_out+'/Phi_abs_err')
    Pt_abs_error_opaque(PUPPI_pt[:, 0], predict_test[:, 0], Yr_test[:, 0], name=path_out+'/Pt_abs_error')


def train_loadAllData(args):
    # gpus = tf.config.list_physical_devices('GPU')
    print("Built with CUDA:", tf.test.is_built_with_cuda())
    print("Physical GPUs:", tf.config.list_physical_devices('GPU'))
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(gpus[2], 'GPU')  # GPU:1만 사용
    tf.config.experimental.set_memory_growth(gpus[2], True) 

    normFac = 1.
    # custom_loss = tf.keras.losses.MSE()
    custom_loss = custom_loss_wrapper(normFac)
    # custom_loss = hybrid_vector_recoil_loss()
    # custom_loss = met_vec_loss(lam_mag=2.0, lam_dir=0.2)
    epochs = args.epochs
    batch_size = args.batch_size
    path_in = args.input
    path_out = args.output
    
    maxNPF = 128
    n_features_pf = 6
    n_features_pf_cat = 2
    quantized = args.quantized
    # quantized = [32, 16]
    units = args.units
    t_mode = 1

    h5files = []
    for p in path_in:
        if p.endswith('.h5'):
            h5files.append(p)
        else:
            h5files += glob(os.path.join(p, '*.h5'))

    Xorg, Y = read_input(h5files)
    
    Y = Y / -normFac

    Xi, Xp, Xc1, Xc2 = preProcessing(Xorg, normFac)
    Xc = [Xc1, Xc2]

    emb_input_dim = {
        i: int(np.max(Xc[i][0:1000])) + 1 for i in range(n_features_pf_cat)
    }

    Yr = Y
    Xr = [Xi, Xp] + Xc

    indices = np.array([i for i in range(len(Yr))])
    indices_train, indices_test = train_test_split(indices, test_size=1./7., random_state=7)
    indices_train, indices_valid = train_test_split(indices_train, test_size=1./6., random_state=7)

    print(f"→ Total samples: {len(Yr)}")
    print(f"→ train samples: {len(indices_train)}")
    print(f"→ valid samples: {len(indices_valid)}")
    print(f"→ test  samples: {len(indices_test)}")
    # roughly the same split as the data generator workflow (train:valid:test=5:1:1)

    Xr_train = [x[indices_train] for x in Xr]
    Xr_test = [x[indices_test] for x in Xr]
    Xr_valid = [x[indices_valid] for x in Xr]
    Yr_train = Yr[indices_train]
    Yr_test = Yr[indices_test]
    Yr_valid = Yr[indices_valid]


    logit_total_bits = int(quantized[0])
    logit_int_bits = int(quantized[1])
    activation_total_bits = int(quantized[0])
    activation_int_bits = int(quantized[1])

    keras_model = dense_embedding(
                n_features=n_features_pf,
                emb_out_dim=2,
                n_features_cat=n_features_pf_cat,
                activation="tanh",
                embedding_input_dim=emb_input_dim,
                number_of_pupcandis=maxNPF,
                t_mode=t_mode,
                with_bias=False,
                units=units,
            )

    # keras_model = dense_embedding_quantized(n_features=n_features_pf,
    #                                         emb_out_dim=2,
    #                                         n_features_cat=n_features_pf_cat,
    #                                         activation_quantizer='quantized_relu',
    #                                         embedding_input_dim=emb_input_dim,
    #                                         number_of_pupcandis=maxNPF,
    #                                         t_mode=t_mode,
    #                                         with_bias=False,

    #                                         logit_quantizer='quantized_bits',
    #                                         # logit_quantizer='ternary',
    #                                         logit_total_bits=logit_total_bits,
    #                                         logit_int_bits=logit_int_bits,
    #                                         activation_total_bits=activation_total_bits,
    #                                         activation_int_bits=activation_int_bits,
    #                                         alpha=1,
    #                                         use_stochastic_rounding=False,
    #                                         units=units)
    


    # Check which model will be used (0 for L1MET Model, 1 for DeepMET Model)
    if t_mode == 0:
        keras_model.compile(optimizer='adam', loss=custom_loss, metrics=['mean_absolute_error', 'mean_squared_error'])
        verbose = 1
    elif t_mode == 1:
        optimizer = optimizers.Adam(lr=1., clipnorm=1.)
        keras_model.compile(loss=custom_loss, optimizer=optimizer,
                            metrics=['mean_absolute_error', 'mean_squared_error'])
        verbose = 1

    # Run training
    print(keras_model.summary())

    start_time = time.time()  # check start time
    history = keras_model.fit(Xr_train,
                              Yr_train,
                              epochs=epochs,
                              batch_size=batch_size,
                              verbose=verbose,  # switch to 1 for more verbosity
                              validation_data=(Xr_valid, Yr_valid),
                              callbacks=get_callbacks(path_out, len(Yr_train), batch_size))

    end_time = time.time()  # check end time




    predict_test = keras_model.predict(Xr_test) * normFac
    PUPPI_pt = normFac * np.sum(Xr_test[1], axis=1)
    Yr_test = normFac * Yr_test

    test(Yr_test, predict_test, PUPPI_pt, path_out)

    if args.drawTurnOn:
        turnon_list = ["TTToSemileptonic_238k_input", "VBFHToInvisible_48k_input", "VBFHToInvisible_48k_151X_input", "TT_PU200_L1METML_299k_input"]
        make_turnon(args.output, keras_model, turnon_list)

    fi = open("{}time.txt".format(path_out), 'w')

    fi.write("Working Time (s) : {}".format(end_time - start_time))
    fi.write("Working Time (m) : {}".format((end_time - start_time)/60.))

    fi.close()
