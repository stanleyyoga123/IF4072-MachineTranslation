import typing
from typing import Any

import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, GRU
from tensorflow.keras.layers import Layer

from .attention import BahdanauAttention


class DecoderInput(typing.NamedTuple):
    new_tokens: Any
    enc_output: Any
    mask: Any


class DecoderOutput(typing.NamedTuple):
    logits: Any
    attention_weights: Any


class Decoder(tf.keras.layers.Layer):
    def __init__(self, output_vocab_size, embedding_dim, dec_units):
        super(Decoder, self).__init__()

        self.embedding = tf.keras.layers.Embedding(output_vocab_size, embedding_dim)

        self.gru = tf.keras.layers.GRU(
            dec_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )

        self.attention = BahdanauAttention(dec_units)
        self.Wc = Dense(dec_units, activation=tf.math.tanh, use_bias=False)
        self.fc = Dense(output_vocab_size)

    def __call__(self, inputs, state=None):
        vectors = self.embedding(inputs.new_tokens)
        rnn_output, state = self.gru(vectors, initial_state=state)
        context_vector, attention_weights = self.attention(
            query=rnn_output, value=inputs.enc_output, mask=inputs.mask
        )
        context_and_rnn_output = tf.concat([context_vector, rnn_output], axis=-1)
        attention_vector = self.Wc(context_and_rnn_output)
        logits = self.fc(attention_vector)
        return DecoderOutput(logits, attention_weights), state
