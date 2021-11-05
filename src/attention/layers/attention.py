import tensorflow as tf
from tensorflow.keras.layers import Dense, AdditiveAttention
from tensorflow.keras.layers import Layer


class BahdanauAttention(Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()

        self.w_q = Dense(units, use_bias=False)
        self.w_k = Dense(units, use_bias=False)

        self.attention = AdditiveAttention()

    def __call__(self, query, value, mask):
        q = self.w_q(query)
        k = self.w_k(value)
        q_mask = tf.ones(tf.shape(query)[:-1], dtype=bool)
        v_mask = mask

        context_vector, attention_weights = self.attention(
            inputs=[q, value, k], mask=[q_mask, v_mask], return_attention_scores=True
        )
        return context_vector, attention_weights
