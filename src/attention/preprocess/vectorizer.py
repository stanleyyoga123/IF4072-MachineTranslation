import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

@tf.keras.utils.register_keras_serializable()
def clean(text):
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, "[^ a-z.?!,¿]", "")
    text = tf.strings.regex_replace(text, "[.?!,¿]", r" \0 ")
    text = tf.strings.strip(text)
    text = tf.strings.join(["<start>", text, "<end>"], separator=" ")
    return text

class Vectorizer:
    def __init__(self, config, input_texts, output_texts, max_vocab_size=True):
        params = {
            "standardize": clean,
            "output_sequence_length": config.max_seq_len,
        }
        if max_vocab_size:
            params["max_tokens"] = config.max_vocab_size

        self.config = config
        self.input_text_processor = TextVectorization(**params)
        self.output_text_processor = TextVectorization(**params)

        self._prepare_dataset(input_texts, output_texts)
        self._adapt(input_texts, output_texts)

    def _prepare_dataset(self, input_texts, output_texts):
        buffer_size = len(input_texts)
        dataset = tf.data.Dataset.from_tensor_slices(
            (input_texts, output_texts)
        ).shuffle(buffer_size)
        self.dataset = dataset.batch(self.config.batch_size)

    def _adapt(self, input_texts, output_texts):
        self.input_text_processor.adapt(input_texts)
        self.output_text_processor.adapt(output_texts)
