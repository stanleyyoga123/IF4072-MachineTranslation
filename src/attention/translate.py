import tensorflow as tf


def translate(date_used):
    translator = tf.saved_model.load(f'bin/translator_{date_used}')
    while True:
        str_in = input("Put text here: ")
        if str_in == "<stop>":
            break
        input_text = tf.constant([str_in])
        result = translator.tf_translate(input_text=input_text)

        print(f"Input Text: {str_in}")
        print(f"Translated Text: {result['text'][0].numpy().decode()}")