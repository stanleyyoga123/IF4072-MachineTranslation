import tensorflow as tf
import pandas as pd
from tqdm import tqdm


def clean(text):
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, "[^ a-z.?!,¿]", "")
    text = tf.strings.regex_replace(text, "[.?!,¿]", r" \0 ")
    text = tf.strings.strip(text)
    return text.numpy().decode()

def convert_batch(data, batch_size):
    temp = []
    num_step = len(data) // batch_size + 1
    for i in range(num_step):
        temp.append(data[i*batch_size : (i+1)*batch_size])
    return temp


def evaluate(model_path, df_path, save_path):
    translator = tf.saved_model.load(model_path)
    test = pd.read_csv(df_path)
    # test = test.sample(10)
    input_text = tf.constant(test['id'].values)
    print('Cleaning Target')
    target = [clean(text) for text in test['en'].values]
    print('Translating')
    batch_size = 64
    batches = convert_batch(input_text, batch_size)
    all_pred = []
    for batch in tqdm(batches):
        y_pred = translator.tf_translate(input_text=batch)
        y_pred = [p.numpy().decode() for p in y_pred['text']]
        all_pred += y_pred

    prediction = pd.DataFrame({
        'input': test['id'].values,
        'target': target,
        'preds': all_pred,
    })
    prediction.to_csv(save_path, index=False)