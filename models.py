import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Lambda


class FeatureSelector(tf.keras.layers.Layer):
    def __init__(self, index, **kwargs):
        super().__init__(**kwargs)
        self.index = index

    def call(self, x):
        return tf.cast(x[:, :, self.index], tf.int32)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1])

    def get_config(self):
        config = super().get_config()
        config.update({"index": self.index})
        return config


class SumParticles(tf.keras.layers.Layer):
    def call(self, x):
        return -tf.reduce_sum(x, axis=1)

    def get_config(self):
        return super().get_config()

        
class FixedPointClipConstraint(tf.keras.constraints.Constraint):
    def __init__(self, total_bits=9, integer_bits=2):
        self.total_bits = total_bits
        self.integer_bits = integer_bits

    def __call__(self, w):
        frac_bits = self.total_bits - self.integer_bits
        scale = tf.cast(2 ** frac_bits, w.dtype)

        min_value = tf.cast(-(2 ** (self.integer_bits - 1)), w.dtype)
        max_value = tf.cast((2 ** (self.integer_bits - 1)) - 1.0 / scale, w.dtype)

        return tf.clip_by_value(w, min_value, max_value)

    def get_config(self):
        return {
            "total_bits": self.total_bits,
            "integer_bits": self.integer_bits,
        }


class QuantizeWeightSTE(tf.keras.layers.Layer):
    def __init__(
        self,
        total_bits=9,
        integer_bits=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.total_bits = total_bits
        self.integer_bits = integer_bits

    def call(self, x):
        frac_bits = self.total_bits - self.integer_bits

        scale = tf.cast(2 ** frac_bits, x.dtype)
        qmin = tf.cast(-(2 ** (self.total_bits - 1)), x.dtype)
        qmax = tf.cast(2 ** (self.total_bits - 1) - 1, x.dtype)

        q = tf.round(x * scale)
        q = tf.clip_by_value(q, qmin, qmax)

        x_q = q / scale

        return x + tf.stop_gradient(x_q - x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "total_bits": self.total_bits,
                "integer_bits": self.integer_bits,
            }
        )
        return config


def TinyTableModel(
        NUM_Features=3,
        NUM_PID=8,
        NUM_ETA=8,
        NUM_PT=2**14,
        D=2,
        number_of_pupcandis=128,
        quantize_embedding=False,
        gamma_total_bits=9,
        gamma_integer_bits=2,
        beta_total_bits=9,
        beta_integer_bits=4,
):
    N = number_of_pupcandis

    input_features = Input(shape=(N, NUM_Features), dtype="int32", name="input_features")

    input_pid = FeatureSelector(0, name="pID")(input_features)
    input_encoded_eta = FeatureSelector(1, name="encoded_eta")(input_features)
    input_encoded_pT = FeatureSelector(2, name="encoded_pT")(input_features)

    comb_idx = (
        input_pid * (NUM_ETA * NUM_PT)
        + input_encoded_eta * NUM_PT
        + input_encoded_pT
    )

    input_pxpy = Input(shape=(N, 2), dtype="float32", name="inputs_pxpy")

    if quantize_embedding:
        gamma_constraint = FixedPointClipConstraint(
            total_bits=gamma_total_bits,
            integer_bits=gamma_integer_bits,
        )

        beta_constraint = FixedPointClipConstraint(
            total_bits=beta_total_bits,
            integer_bits=beta_integer_bits,
        )
    else:
        gamma_constraint = None
        beta_constraint = None

    gamma = Embedding(
        NUM_PID * NUM_ETA * NUM_PT,
        D,
        embeddings_initializer="ones",
        embeddings_constraint=gamma_constraint,
        name="gamma",
    )

    beta = Embedding(
        NUM_PID * NUM_ETA * NUM_PT,
        D,
        embeddings_initializer="zeros",
        embeddings_constraint=beta_constraint,
        name="beta",
    )

    gamma_emb = gamma(comb_idx)
    beta_emb = beta(comb_idx)

    if quantize_embedding:
        gamma_emb = QuantizeWeightSTE(
            total_bits=gamma_total_bits,
            integer_bits=gamma_integer_bits,
            name="gamma_q",
        )(gamma_emb)

        beta_emb = QuantizeWeightSTE(
            total_bits=beta_total_bits,
            integer_bits=beta_integer_bits,
            name="beta_q",
        )(beta_emb)

    x = input_pxpy * gamma_emb + beta_emb
    x = SumParticles(name="sum_particles")(x)
    keras_model = Model(inputs=[input_features, input_pxpy], outputs=x)

    return keras_model