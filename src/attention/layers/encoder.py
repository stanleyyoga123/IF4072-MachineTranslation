from tensorflow.keras.layers import Input, Embedding, GRU
from tensorflow.keras.layers import Layer


class Encoder(Layer):
    def __init__(self, vocab_size, embedding_dim, enc_units):
        super(Encoder, self).__init__()

        self.embedding = Embedding(vocab_size, embedding_dim)

        self.gru1 = GRU(
            enc_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )
        self.gru2 = GRU(
            enc_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )

    def __call__(self, tokens, state=None):
        vectors = self.embedding(tokens)
        output, state = self.gru1(vectors, initial_state=state)
        output, state = self.gru2(output, initial_state=state)
        return output, state
