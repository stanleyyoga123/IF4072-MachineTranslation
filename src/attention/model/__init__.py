import tensorflow as tf
from layers import Encoder, Decoder, DecoderInput


class TrainTranslator(tf.keras.Model):
    def __init__(self, config, vectorizer, use_tf_function=True):
        super().__init__()

        self.input_text_processor = vectorizer.input_text_processor
        self.output_text_processor = vectorizer.output_text_processor

        encoder = Encoder(
            self.input_text_processor.vocabulary_size(),
            config.embedding_dim,
            config.units,
        )
        decoder = Decoder(
            self.output_text_processor.vocabulary_size(),
            config.embedding_dim,
            config.units,
        )

        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.use_tf_function = use_tf_function

    def train_step(self, inputs):
        if self.use_tf_function:
            return self._tf_train_step(inputs)
        else:
            return self._train_step(inputs)

    def _preprocess(self, input_text, target_text):
        input_tokens = self.input_text_processor(input_text)
        target_tokens = self.output_text_processor(target_text)

        input_mask = input_tokens != 0
        target_mask = target_tokens != 0

        return input_tokens, input_mask, target_tokens, target_mask

    def _train_step(self, inputs):
        input_text, target_text = inputs
        input_tokens, input_mask, target_tokens, target_mask = self._preprocess(
            input_text, target_text
        )
        max_target_length = tf.shape(target_tokens)[1]

        with tf.GradientTape() as tape:
            enc_output, enc_state = self.encoder(input_tokens)
            dec_state = enc_state
            loss = tf.constant(0.0)

            for t in range(max_target_length - 1):
                new_tokens = target_tokens[:, t : t + 2]
                step_loss, dec_state = self._loop_step(
                    new_tokens, input_mask, enc_output, dec_state
                )
                loss += step_loss

            average_loss = loss / tf.reduce_sum(tf.cast(target_mask, tf.float32))

        variables = self.trainable_variables
        gradients = tape.gradient(average_loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return {"batch_loss": average_loss}

    @tf.function(
        input_signature=[
            [
                tf.TensorSpec(dtype=tf.string, shape=[None]),
                tf.TensorSpec(dtype=tf.string, shape=[None]),
            ]
        ]
    )
    def _tf_train_step(self, inputs):
        return self._train_step(inputs)

    def _loop_step(self, new_tokens, input_mask, enc_output, dec_state):
        input_token, target_token = new_tokens[:, 0:1], new_tokens[:, 1:2]

        decoder_input = DecoderInput(
            new_tokens=input_token, enc_output=enc_output, mask=input_mask
        )
        dec_result, dec_state = self.decoder(decoder_input, state=dec_state)

        y = target_token
        y_pred = dec_result.logits

        step_loss = self.loss(y, y_pred)

        return step_loss, dec_state
