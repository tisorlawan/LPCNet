#!/usr/bin/python3
"""Copyright (c) 2018 Mozilla

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

- Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

- Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import math
import sys

import h5py
import numpy as np
import tensorflow as tf
from diffembed import diff_Embed
from mdense import MDense
from parameters import set_parameter
from tensorflow.compat.v1.keras.layers import CuDNNGRU
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.layers import (
    GRU,
    Activation,
    Add,
    Bidirectional,
    Concatenate,
    Conv1D,
    Dense,
    Embedding,
    GaussianNoise,
    Input,
    Lambda,
    MaxPooling1D,
    Multiply,
    Reshape,
)
from tensorflow.keras.models import Model
from tf_funcs import *

frame_size = 160
pcm_bits = 8
embed_size = 128
pcm_levels = 2**pcm_bits


def interleave(p, samples):
    p2 = tf.expand_dims(p, 3)
    nb_repeats = pcm_levels // (2 * p.shape[2])
    p3 = tf.reshape(
        tf.repeat(tf.concat([1 - p2, p2], 3), nb_repeats), (-1, samples, pcm_levels)
    )
    return p3


def tree_to_pdf(p, samples):
    return (
        interleave(p[:, :, 1:2], samples)
        * interleave(p[:, :, 2:4], samples)
        * interleave(p[:, :, 4:8], samples)
        * interleave(p[:, :, 8:16], samples)
        * interleave(p[:, :, 16:32], samples)
        * interleave(p[:, :, 32:64], samples)
        * interleave(p[:, :, 64:128], samples)
        * interleave(p[:, :, 128:256], samples)
    )


def tree_to_pdf_train(p):
    # FIXME: try not to hardcode the 2400 samples (15 frames * 160 samples/frame)
    return tree_to_pdf(p, 2400)


def tree_to_pdf_infer(p):
    return tree_to_pdf(p, 1)


def quant_regularizer(x):
    Q = 128
    Q_1 = 1.0 / Q
    # return .01 * tf.reduce_mean(1 - tf.math.cos(2*3.1415926535897931*(Q*x-tf.round(Q*x))))
    return 0.01 * tf.reduce_mean(
        K.sqrt(
            K.sqrt(
                1.0001 - tf.math.cos(2 * 3.1415926535897931 * (Q * x - tf.round(Q * x)))
            )
        )
    )


class Sparsify(Callback):
    def __init__(self, t_start, t_end, interval, density, quantize=False):
        super(Sparsify, self).__init__()
        self.batch = 0
        self.t_start = t_start
        self.t_end = t_end
        self.interval = interval
        self.final_density = density
        self.quantize = quantize

    def on_batch_end(self, batch, logs=None):
        # print("batch number", self.batch)
        self.batch += 1
        if (
            self.quantize
            or (
                self.batch > self.t_start
                and (self.batch - self.t_start) % self.interval == 0
            )
            or self.batch >= self.t_end
        ):
            # print("constrain");
            layer = self.model.get_layer("gru_a")
            w = layer.get_weights()
            p = w[1]
            nb = p.shape[1] // p.shape[0]
            N = p.shape[0]
            # print("nb = ", nb, ", N = ", N);
            # print(p.shape)
            # print ("density = ", density)
            for k in range(nb):
                density = self.final_density[k]
                if self.batch < self.t_end and not self.quantize:
                    r = 1 - (self.batch - self.t_start) / (self.t_end - self.t_start)
                    density = 1 - (1 - self.final_density[k]) * (1 - r * r * r)
                A = p[:, k * N : (k + 1) * N]
                A = A - np.diag(np.diag(A))
                # This is needed because of the CuDNNGRU strange weight ordering
                A = np.transpose(A, (1, 0))
                L = np.reshape(A, (N // 4, 4, N // 8, 8))
                S = np.sum(L * L, axis=-1)
                S = np.sum(S, axis=1)
                SS = np.sort(np.reshape(S, (-1,)))
                thresh = SS[round(N * N // 32 * (1 - density))]
                mask = (S >= thresh).astype("float32")
                mask = np.repeat(mask, 4, axis=0)
                mask = np.repeat(mask, 8, axis=1)
                mask = np.minimum(1, mask + np.diag(np.ones((N,))))
                # This is needed because of the CuDNNGRU strange weight ordering
                mask = np.transpose(mask, (1, 0))
                p[:, k * N : (k + 1) * N] = p[:, k * N : (k + 1) * N] * mask
                # print(thresh, np.mean(mask))
            if self.quantize and (
                (
                    self.batch > self.t_start
                    and (self.batch - self.t_start) % self.interval == 0
                )
                or self.batch >= self.t_end
            ):
                if self.batch < self.t_end:
                    threshold = (
                        0.5 * (self.batch - self.t_start) / (self.t_end - self.t_start)
                    )
                else:
                    threshold = 0.5
                quant = np.round(p * 128.0)
                res = p * 128.0 - quant
                mask = (np.abs(res) <= threshold).astype("float32")
                p = mask / 128.0 * quant + (1 - mask) * p

            w[1] = p
            layer.set_weights(w)


class SparsifyGRUB(Callback):
    def __init__(self, t_start, t_end, interval, grua_units, density, quantize=False):
        super(SparsifyGRUB, self).__init__()
        self.batch = 0
        self.t_start = t_start
        self.t_end = t_end
        self.interval = interval
        self.final_density = density
        self.grua_units = grua_units
        self.quantize = quantize

    def on_batch_end(self, batch, logs=None):
        # print("batch number", self.batch)
        self.batch += 1
        if (
            self.quantize
            or (
                self.batch > self.t_start
                and (self.batch - self.t_start) % self.interval == 0
            )
            or self.batch >= self.t_end
        ):
            # print("constrain");
            layer = self.model.get_layer("gru_b")
            w = layer.get_weights()
            p = w[0]
            N = p.shape[0]
            M = p.shape[1] // 3
            for k in range(3):
                density = self.final_density[k]
                if self.batch < self.t_end and not self.quantize:
                    r = 1 - (self.batch - self.t_start) / (self.t_end - self.t_start)
                    density = 1 - (1 - self.final_density[k]) * (1 - r * r * r)
                A = p[:, k * M : (k + 1) * M]
                # This is needed because of the CuDNNGRU strange weight ordering
                A = np.reshape(A, (M, N))
                A = np.transpose(A, (1, 0))
                N2 = self.grua_units
                A2 = A[:N2, :]
                L = np.reshape(A2, (N2 // 4, 4, M // 8, 8))
                S = np.sum(L * L, axis=-1)
                S = np.sum(S, axis=1)
                SS = np.sort(np.reshape(S, (-1,)))
                thresh = SS[round(M * N2 // 32 * (1 - density))]
                mask = (S >= thresh).astype("float32")
                mask = np.repeat(mask, 4, axis=0)
                mask = np.repeat(mask, 8, axis=1)
                A = np.concatenate([A2 * mask, A[N2:, :]], axis=0)
                # This is needed because of the CuDNNGRU strange weight ordering
                A = np.transpose(A, (1, 0))
                A = np.reshape(A, (N, M))
                p[:, k * M : (k + 1) * M] = A
                # print(thresh, np.mean(mask))
            if self.quantize and (
                (
                    self.batch > self.t_start
                    and (self.batch - self.t_start) % self.interval == 0
                )
                or self.batch >= self.t_end
            ):
                if self.batch < self.t_end:
                    threshold = (
                        0.5 * (self.batch - self.t_start) / (self.t_end - self.t_start)
                    )
                else:
                    threshold = 0.5
                quant = np.round(p * 128.0)
                res = p * 128.0 - quant
                mask = (np.abs(res) <= threshold).astype("float32")
                p = mask / 128.0 * quant + (1 - mask) * p

            w[0] = p
            layer.set_weights(w)


class PCMInit(Initializer):
    def __init__(self, gain=0.1, seed=None):
        self.gain = gain
        self.seed = seed

    def __call__(self, shape, dtype=None):
        num_rows = 1
        for dim in shape[:-1]:
            num_rows *= dim
        num_cols = shape[-1]
        flat_shape = (num_rows, num_cols)
        if self.seed is not None:
            np.random.seed(self.seed)
        a = np.random.uniform(-1.7321, 1.7321, flat_shape)
        # a[:,0] = math.sqrt(12)*np.arange(-.5*num_rows+.5,.5*num_rows-.4)/num_rows
        # a[:,1] = .5*a[:,0]*a[:,0]*a[:,0]
        a = a + np.reshape(
            math.sqrt(12)
            * np.arange(-0.5 * num_rows + 0.5, 0.5 * num_rows - 0.4)
            / num_rows,
            (num_rows, 1),
        )
        return self.gain * a.astype("float32")

    def get_config(self):
        return {"gain": self.gain, "seed": self.seed}


class WeightClip(Constraint):
    """Clips the weights incident to each hidden unit to be inside a range"""

    def __init__(self, c=2):
        self.c = c

    def __call__(self, p):
        # Ensure that abs of adjacent weights don't sum to more than 127. Otherwise there's a risk of
        # saturation when implementing dot products with SSSE3 or AVX2.
        return (
            self.c
            * p
            / tf.maximum(
                self.c, tf.repeat(tf.abs(p[:, 1::2]) + tf.abs(p[:, 0::2]), 2, axis=1)
            )
        )
        # return K.clip(p, -self.c, self.c)

    def get_config(self):
        return {"name": self.__class__.__name__, "c": self.c}


constraint = WeightClip(0.992)


def new_lpcnet_model(
    rnn_units1=384,
    rnn_units2=16,
    nb_used_features=20,
    batch_size=128,
    training=False,
    adaptation=False,
    quantize=False,
    flag_e2e=False,
    cond_size=128,
    lpc_order=16,
    lpc_gamma=1.0,
    lookahead=2,
):
    """
    End-to-end LPCNet generator.

    Inputs:
      pcm         (B, T_s, 1)          – μ-law encoded previous samples
      dpcm        (B, T_s, 3)          – one‐hot of last 3 residuals
      feat        (B, T_f, F)          – frame‐rate features
      pitch       (B, T_f, 1)          – quantized pitch
      (if !flag_e2e) lpcoeffs (B, T_f, L)
      (decoder) dec_feat  (B, T_f, C)
                dec_state1 (B, U1)
                dec_state2 (B, U2)

    Internal shapes:
      T_f = number of frames
      T_s = T_f * frame_size
      F   = nb_used_features
      L   = lpc_order
      C   = cond_size
      U1  = rnn_units1
      U2  = rnn_units2

    Outputs:
      m_out       (B, T_s, 2 + pcm_levels) – [tensor_preds, real_preds, μ-law PDF]
      cfeat       (B, T_f, cond_size)     – conditioning features (for decoder)
    """

    # ─────────────────────────────────────────────────────────────────────────
    # 1) Define all inputs
    # ─────────────────────────────────────────────────────────────────────────
    pcm = Input(shape=(None, 1), batch_size=batch_size, name="pcm_in")  # (B, T_s, 1)
    dpcm = Input(shape=(None, 3), batch_size=batch_size, name="dpcm_in")  # (B, T_s, 3)
    feat = Input(
        shape=(None, nb_used_features), batch_size=batch_size, name="feat"
    )  # (B, T_f, F)
    pitch = Input(shape=(None, 1), batch_size=batch_size, name="pitch")  # (B, T_f, 1)

    # Inputs used only to build the decoder sub-model
    dec_feat = Input(shape=(None, cond_size), name="dec_feat")  # (B, T_f, C)
    dec_state1 = Input(shape=(rnn_units1,), name="dec_state1")  # (B, U1)
    dec_state2 = Input(shape=(rnn_units2,), name="dec_state2")  # (B, U2)

    # ─────────────────────────────────────────────────────────────────────────
    # 2) Frame‐rate network → cfeat
    # ─────────────────────────────────────────────────────────────────────────
    padding = "valid" if training else "same"
    # a) Embed pitch → 64-dim, concat with feat → (B, T_f, F+64)
    pembed = Embedding(256, 64, name="embed_pitch")
    pitch_emb = Reshape((-1, 64))(pembed(pitch))
    fr_cat = Concatenate(name="fr_concat")([feat, pitch_emb])

    # b) Two causal Conv1D layers → (B, T_f, C)
    fconv1 = Conv1D(
        cond_size, 3, padding=padding, activation="tanh", name="feature_conv1"
    )
    fconv2 = Conv1D(
        cond_size, 3, padding=padding, activation="tanh", name="feature_conv2"
    )
    x = fconv1(fr_cat)  # (B, T_f, C)
    x = fconv2(x)  # (B, T_f, C)

    # c) Two Dense layers → final conditioning features cfeat (B, T_f, C)
    fdense1 = Dense(cond_size, activation="tanh", name="feature_dense1")
    fdense2 = Dense(cond_size, activation="tanh", name="feature_dense2")
    if flag_e2e and quantize:
        # freeze frame‐rate network when fine‐quantizing
        for layer in (fconv1, fconv2, fdense1, fdense2):
            layer.trainable = False

    cfeat = fdense2(fdense1(x))  # (B, T_f, C)

    # ─────────────────────────────────────────────────────────────────────────
    # 3) End‐to‐End LPC branch: RC → LPC → lpcoeffs
    # ─────────────────────────────────────────────────────────────────────────
    if flag_e2e:
        # lpcoeffs: (B, T_f, L)
        lpcoeffs = diff_rc2lpc(name="rc2lpc")(cfeat)
    else:
        lpcoeffs = Input(
            shape=(None, lpc_order), batch_size=batch_size, name="lpcoeffs"
        )  # (B, T_f, L)

    # ─────────────────────────────────────────────────────────────────────────
    # 4) Sample‐rate predictions (no external upsampling of lpcoeffs!)
    #    ── real_preds for loss
    #    ── tensor_preds for excitation
    # ─────────────────────────────────────────────────────────────────────────
    real_preds = diff_pred(name="real_lpc2preds")([pcm, lpcoeffs])  # (B, T_s, 1)

    # weight LPC coefficients by γ^i
    weighting = lpc_gamma ** np.arange(1, lpc_order + 1, dtype="float32")  # (L,)
    weighted_lpc = Lambda(lambda z: z[0] * z[1], name="weight_lp")(
        [lpcoeffs, weighting]
    )  # (B, T_f, L)

    tensor_preds = diff_pred(name="lpc2preds")([pcm, weighted_lpc])  # (B, T_s, 1)

    # past_errors = μ-law(pcm) – tensor_preds shifted by 1
    error_calc = Lambda(
        lambda x: tf_l2u(x[0] - tf.roll(x[1], 1, axis=1)), name="past_errors"
    )
    past_errors = error_calc([pcm, tensor_preds])  # (B, T_s, 1)

    # ─────────────────────────────────────────────────────────────────────────
    # 5) Embed signals for sample-rate RNN
    # ─────────────────────────────────────────────────────────────────────────
    # a) Concatenate μ-law(pcm), μ-law(tensor_preds), past_errors → (B, T_s, 3)
    sig_cat = Concatenate(name="cpcm")(
        [tf_l2u(pcm), tf_l2u(tensor_preds), past_errors]
    )  # (B, T_s, 3)

    # b) Add uniform noise if desired
    sig_noisy = GaussianNoise(0.3)(sig_cat)  # (B, T_s, 3)

    # c) Embed 3-dim to embed_size via diff_Embed → (B, T_s, 3, embed_size)
    embed = diff_Embed(name="embed_sig", initializer=PCMInit())
    cpcm = embed(sig_noisy)  # (B, T_s, 3, E)
    cpcm = Reshape((-1, embed_size * 3))(cpcm)  # (B, T_s, 3*E)

    # d) Prepare decoder‐side embedding for the separate decoder model
    cpcm_decoder = Reshape((-1, embed_size * 3))(embed(dpcm))  # (B, T_s, 3*E)

    # ─────────────────────────────────────────────────────────────────────────
    # 6) Upsample cfeat to sample rate and prepare RNN input
    # ─────────────────────────────────────────────────────────────────────────
    rep = Lambda(
        lambda x: K.repeat_elements(x, frame_size, 1), name="repeat_to_sample_rate"
    )
    cfeat_sr = rep(cfeat)  # (B, T_s, C)

    # Concatenate for RNN: [ cpcm, cfeat_sr ] → (B, T_s, 3*E + C)
    rnn_in = Concatenate(name="sr_concat")([cpcm, cfeat_sr])

    # ─────────────────────────────────────────────────────────────────────────
    # 7) Stateful GRU A & B, with optional quant/constraint
    # ─────────────────────────────────────────────────────────────────────────
    quantizer = quant_regularizer if quantize else None

    if training:
        rnn = CuDNNGRU(
            rnn_units1,
            return_sequences=True,
            return_state=True,
            name="gru_a",
            stateful=True,
            recurrent_constraint=constraint,
            recurrent_regularizer=quantizer,
        )
        rnn2 = CuDNNGRU(
            rnn_units2,
            return_sequences=True,
            return_state=True,
            name="gru_b",
            stateful=True,
            kernel_constraint=constraint,
            recurrent_constraint=constraint,
            kernel_regularizer=quantizer,
            recurrent_regularizer=quantizer,
        )
    else:
        rnn = GRU(
            rnn_units1,
            return_sequences=True,
            return_state=True,
            recurrent_activation="sigmoid",
            reset_after=True,
            name="gru_a",
            stateful=True,
            recurrent_constraint=constraint,
            recurrent_regularizer=quantizer,
        )
        rnn2 = GRU(
            rnn_units2,
            return_sequences=True,
            return_state=True,
            recurrent_activation="sigmoid",
            reset_after=True,
            name="gru_b",
            stateful=True,
            kernel_constraint=constraint,
            recurrent_constraint=constraint,
            kernel_regularizer=quantizer,
            recurrent_regularizer=quantizer,
        )

    # a) GRU A on full rnn_in
    gru_out1, _ = rnn(rnn_in)  # (B, T_s, U1)
    gru_out1 = GaussianNoise(0.005)(gru_out1)  # (B, T_s, U1)

    # b) GRU B on [gru_out1, cfeat_sr]
    gru_out2, _ = rnn2(Concatenate()([gru_out1, cfeat_sr]))  # (B, T_s, U2)

    # ─────────────────────────────────────────────────────────────────────────
    # 8) Dual‐FC & tree2pdf → μ-law probability
    # ─────────────────────────────────────────────────────────────────────────
    md = MDense(pcm_levels, activation="sigmoid", name="dual_fc")
    ulaw_prob = Lambda(tree_to_pdf_train, name="ulaw_pdf")(
        md(gru_out2)
    )  # (B, T_s, pcm_levels)

    if adaptation:
        # freeze sample-rate network when adapting
        for layer in (rnn, rnn2, md, embed):
            layer.trainable = False

    # ─────────────────────────────────────────────────────────────────────────
    # 9) Final output concatenation
    # ─────────────────────────────────────────────────────────────────────────
    # [tensor_preds (1), real_preds (1), ulaw_prob (pcm_levels)]
    m_out = Concatenate(name="pdf")(
        [tensor_preds, real_preds, ulaw_prob]
    )  # (B, T_s, 2+pcm_levels)

    # ─────────────────────────────────────────────────────────────────────────
    # 10) Build Keras Models
    # ─────────────────────────────────────────────────────────────────────────
    if not flag_e2e:
        model = Model([pcm, feat, pitch, lpcoeffs], m_out, name="lpcnet")
    else:
        model = Model([pcm, feat, pitch], [m_out, cfeat], name="lpcnet_end2end")

    # save attributes for inference
    model.rnn_units1 = rnn_units1
    model.rnn_units2 = rnn_units2
    model.nb_used_features = nb_used_features
    model.frame_size = frame_size

    # ─────────────────────────────────────────────────────────────────────────
    # 11) Encoder & Decoder submodels
    # ─────────────────────────────────────────────────────────────────────────
    if not flag_e2e:
        encoder = Model([feat, pitch], cfeat, name="lpcnet_encoder")
    else:
        encoder = Model([feat, pitch], [cfeat, lpcoeffs], name="lpcnet_e2e_encoder")

    # Build decoder RNN inference model
    dec_in = Concatenate(name="dec_concat")([cpcm_decoder, dec_feat])  # (B, T_s, 3*E+C)
    dec_out1, state1 = rnn(dec_in, initial_state=dec_state1)
    dec_out2, state2 = rnn2(
        Concatenate()([dec_out1, dec_feat]), initial_state=dec_state2
    )
    dec_pdf = Lambda(tree_to_pdf_infer, name="dec_ulaw_pdf")(md(dec_out2))

    decoder = Model(
        [dpcm, dec_feat, dec_state1, dec_state2],
        [dec_pdf, state1, state2],
        name="lpcnet_decoder",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 12) Store non-trainable parameters
    # ─────────────────────────────────────────────────────────────────────────
    set_parameter(model, "lpc_gamma", lpc_gamma, dtype="float64")
    set_parameter(model, "flag_e2e", flag_e2e, dtype="bool")
    set_parameter(model, "lookahead", lookahead, dtype="int32")

    return model, encoder, decoder
