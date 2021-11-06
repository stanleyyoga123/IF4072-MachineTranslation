import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

import pickle

from .model import Translator


def translate():
    date_used = "2021-11-07 03.32.43"

    input_text_processor = tf.keras.models.load_model(
        f"bin/input_vectorizer_{date_used}"
    ).layers[0]
    output_text_processor = tf.keras.models.load_model(
        f"bin/output_vectorizer_{date_used}"
    ).layers[0]

    translator = Translator(
        encoder=pickle.load(open(f"bin/attention_encoder_{date_used}.pkl", "rb")),
        decoder=pickle.load(open(f"bin/attention_decoder_{date_used}.pkl", "rb")),
        input_text_processor=input_text_processor,
        output_text_processor=output_text_processor,
    )
    while True:
        str_in = input("Put text here: ")
        if str_in == "<stop>":
            break
        input_text = tf.constant([str_in])
        result = translator.translate(input_text=input_text)

        print(f"Input Text: {str_in}")
        print(f"Translated Text: {result['text'][0].numpy().decode()}")


if __name__ == "__main__":
    translate()
