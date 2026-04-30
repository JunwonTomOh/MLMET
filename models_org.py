import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, BatchNormalization, Dropout, Lambda, Conv1D, SpatialDropout1D, Concatenate, Flatten, Reshape, Multiply, Add, GlobalAveragePooling1D, Activation, Permute
from tensorflow import slice
from tensorflow.keras import initializers

import qkeras
from qkeras.quantizers import quantized_bits
from qkeras.qlayers import QDense, QActivation

import numpy as np
import itertools


def TinyTableModel(
        NUM_Features = 3,
        NUM_PID = 8,
        NUM_ETA = 8,
        NUM_PT = 2**14,
        D = 2,
        number_of_pupcandis=128,
):
    N = number_of_pupcandis

    input_features = Input(shape=(N, NUM_Features), dtype="int32", name='input_features')
    input_pid = Lambda(lambda t: tf.cast(t[:, :, 0], tf.int32), name="pID")(input_features)       # (B,N)
    input_encoded_eta = Lambda(lambda t: tf.cast(t[:, :, 1], tf.int32), name="encoded_eta")(input_features)       # (B,N)
    input_encoded_pT = Lambda(lambda t: tf.cast(t[:, :, 2], tf.int32), name="encoded_pT")(input_features)       # (B,N)

    comb_idx = input_pid * (NUM_ETA * NUM_PT) + input_encoded_eta * NUM_PT + input_encoded_pT

    input_pxpy = Input(shape=(N, 2), dtype="float32", name='inputs_pxpy')

    gamma = Embedding(NUM_PID * NUM_ETA * NUM_PT, D, embeddings_initializer="ones", name="gamma")
    beta = Embedding(NUM_PID * NUM_ETA * NUM_PT, D, embeddings_initializer="zeros", name="beta")

    gamma_emb = gamma(comb_idx)
    beta_emb = beta(comb_idx)

    x = input_pxpy * (((gamma_emb * 2**9) // 1) / 2**9) + beta_emb
    x = -tf.reduce_sum(x, axis=1)

    outputs = x

    keras_model = Model(inputs=[input_features, input_pxpy], outputs=outputs)

    return keras_model


def dense_embedding_quantized(n_features=6,
                              n_features_cat=2,
                              number_of_pupcandis=128,
                              embedding_input_dim={0: 13, 1: 3},
                              emb_out_dim=2,
                              with_bias=True,
                              t_mode=0,
                              logit_total_bits=7,
                              logit_int_bits=2,
                              activation_total_bits=7,
                              logit_quantizer='quantized_bits',
                              activation_quantizer='quantized_relu',
                              activation_int_bits=2,
                              alpha=1,
                              use_stochastic_rounding=False,
                              units=[64, 32, 16]):
    n_dense_layers = len(units)

    logit_quantizer = getattr(qkeras.quantizers, logit_quantizer)(logit_total_bits, logit_int_bits, alpha=alpha, use_stochastic_rounding=use_stochastic_rounding)
    activation_quantizer = getattr(qkeras.quantizers, activation_quantizer)(activation_total_bits, activation_int_bits)

    inputs_cont = Input(shape=(number_of_pupcandis, n_features-2), name='input_cont')
    pxpy = Input(shape=(number_of_pupcandis, 2), name='input_pxpy')

    embeddings = []
    inputs = [inputs_cont, pxpy]
    for i_emb in range(n_features_cat):
        input_cat = Input(shape=(number_of_pupcandis, ), name='input_cat{}'.format(i_emb))
        inputs.append(input_cat)
        embedding = Embedding(
            input_dim=embedding_input_dim[i_emb],
            output_dim=emb_out_dim,
            embeddings_initializer=initializers.RandomNormal(
                mean=0,
                stddev=0.4/emb_out_dim),
            name='embedding{}'.format(i_emb))(input_cat)
        embeddings.append(embedding)

    # can concatenate all 3 if updated in hls4ml, for now; do it pairwise
    # x = Concatenate()([inputs_cont] + embeddings[])
    emb_concat = Concatenate()(embeddings)
    x = Concatenate()([inputs_cont, emb_concat])

    for i_dense in range(n_dense_layers):
        x = QDense(units[i_dense], kernel_quantizer=logit_quantizer, bias_quantizer=logit_quantizer, kernel_initializer='lecun_uniform')(x)
        x = BatchNormalization(momentum=0.95)(x)
        x = QActivation(activation=activation_quantizer)(x)

    


    b = QDense(2, name='met_bias', kernel_quantizer=logit_quantizer, bias_quantizer=logit_quantizer, kernel_initializer=initializers.VarianceScaling(scale=0.02))(x)
    pxpy = Add()([pxpy, b])
    w = QDense(1, name='met_weight', kernel_quantizer=logit_quantizer, bias_quantizer=logit_quantizer, kernel_initializer=initializers.VarianceScaling(scale=0.02))(x)
    # w = QDense(1, name='met_weight', kernel_quantizer=logit_quantizer, bias_quantizer=logit_quantizer, kernel_initializer='lecun_uniform')(x)
    w = BatchNormalization(trainable=False, name='met_weight_minus_one', epsilon=1e-7)(w)
    x = Multiply()([w, pxpy])

    x = GlobalAveragePooling1D(name='output')(x)
    outputs = x

    keras_model = Model(inputs=inputs, outputs=outputs)

    keras_model.get_layer('met_weight_minus_one').set_weights([np.array([1.]), np.array([-1.]), np.array([0.]), np.array([1.])])

    return keras_model

def dense_embedding(n_features=6,
                    n_features_cat=2,
                    activation='relu',
                    number_of_pupcandis=128,
                    embedding_input_dim={0: 13, 1: 3},
                    emb_out_dim=8,
                    with_bias=True,
                    t_mode=0,
                    units=[64, 32, 16]):
    n_dense_layers = len(units)

    inputs_cont = Input(shape=(number_of_pupcandis, n_features-2), name='input_cont')
    pxpy = Input(shape=(number_of_pupcandis, 2), name='input_pxpy')

    embeddings = []
    inputs = [inputs_cont, pxpy]
    for i_emb in range(n_features_cat):
        input_cat = Input(shape=(number_of_pupcandis, ), name='input_cat{}'.format(i_emb))
        inputs.append(input_cat)
        embedding = Embedding(
            input_dim=embedding_input_dim[i_emb],
            output_dim=emb_out_dim,
            embeddings_initializer=initializers.RandomNormal(
                mean=0,
                stddev=0.4/emb_out_dim),
            name='embedding{}'.format(i_emb))(input_cat)
        embeddings.append(embedding)

    # can concatenate all 3 if updated in hls4ml, for now; do it pairwise
    # x = Concatenate()([inputs_cont] + embeddings)
    emb_concat = Concatenate()(embeddings)
    x = Concatenate()([inputs_cont, emb_concat])

    for i_dense in range(n_dense_layers):
        x = Dense(units[i_dense], activation='linear', kernel_initializer='lecun_uniform')(x)
        x = BatchNormalization(momentum=0.95)(x)
        x = Activation(activation=activation)(x)

    if t_mode == 0:
        x = GlobalAveragePooling1D(name='pool')(x)
        x = Dense(2, name='output', activation='linear')(x)

    if t_mode == 1:
        if with_bias:
            b = Dense(2, name='met_bias', activation='linear', kernel_initializer=initializers.VarianceScaling(scale=0.02))(x)
            pxpy = Add()([pxpy, b])
        w = Dense(1, name='met_weight', activation='linear', kernel_initializer=initializers.VarianceScaling(scale=0.02))(x)
        w = BatchNormalization(trainable=False, name='met_weight_minus_one', epsilon=False)(w)
        x = Multiply()([w, pxpy])

        #x = GlobalAveragePooling1D(name='output')(x)
        x = -Lambda(lambda x: K.sum(x, axis=1), name='output')(x)

    if t_mode == 2: # regresses a single met weight rather than 128
        w = Dense(1, name='puppi_weights', activation='linear', kernel_initializer=initializers.VarianceScaling(scale=0.02))(x)
        w = Lambda(lambda x: K.sum(x, axis=1), name='met_weight')(w) 

        pmet = Lambda(lambda x: K.sum(x, axis=1), name='pmet')(pxpy)
        pmet = BatchNormalization(trainable=False, name='pmet_weight_minus_one', epsilon=False)(pmet)

        x = Multiply()([w, pmet])

    outputs = x

    keras_model = Model(inputs=inputs, outputs=outputs)
    
    if t_mode == 1:
        keras_model.get_layer('met_weight_minus_one').set_weights([np.array([1.]), np.array([-1.]), np.array([0.]), np.array([1.])])
    elif t_mode == 2:
        keras_model.get_layer('pmet_weight_minus_one').set_weights([np.array([1.,1.]), np.array([-1.,-1.]), np.array([0.,0.]), np.array([1., 1.])])
    return keras_model


