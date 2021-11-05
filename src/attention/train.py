import pandas as pd
import pickle

import tensorflow as tf
from tensorflow.keras.callbacks import Callback

from .model import TrainTranslator
from .config import Config
from .preprocess import Vectorizer
from .loss import MaskedLoss

from datetime import datetime


class BatchLogs(Callback):
    def __init__(self, key):
        self.key = key
        self.logs = []

    def on_train_batch_end(self, n, logs):
        self.logs.append(logs[self.key])


batch_loss = BatchLogs("batch_loss")


class AdapterModel(tf.keras.Model):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder


def save_layer(encoder, decoder, path):
    adapter = AdapterModel(encoder, decoder)
    adapter.save_weights(path)


def clean_dup_nan(df):
    ret = df.dropna()
    ret = ret.drop(ret[ret.duplicated()].index)
    ret = ret.reset_index(drop=True)
    return ret


def train(verbose=True):
    str_date = datetime.now().strftime("%Y-%m-%d %H.%M.%S")
    
    if verbose:
        print("Load Dataframe")

    start = datetime.now()
    df_train = pd.read_csv("data/filtered/train.csv")
    df_val = pd.read_csv("data/filtered/val.csv")
    df_test = pd.read_csv("data/filtered/test.csv")

    if verbose:
        print(f"Time Taken {datetime.now() - start}")

    if verbose:
        print("Clean Dataframe")

    start = datetime.now()
    df_train = clean_dup_nan(df_train)
    df_val = clean_dup_nan(df_val)
    df_test = clean_dup_nan(df_test)

    if Config.sample:
        df_train = df_train.sample(Config.sample)

    if verbose:
        print(f"Time Taken {datetime.now() - start}")
        print(f"Total data train --> {len(df_train)}")

    output_texts = df_train["en"].values
    input_texts = df_train["id"].values

    if verbose:
        print("Vectorizing")

    start = datetime.now()
    vectorizer = Vectorizer(Config, input_texts, output_texts)
    input_conf = {
        "config": vectorizer.input_text_processor.get_config(),
        "weights": vectorizer.input_text_processor.get_weights(),
    }
    output_conf = {
        "config": vectorizer.output_text_processor.get_config(),
        "weights": vectorizer.output_text_processor.get_weights(),
    }
    pickle.dump(
        input_conf,
        open(f"bin/input_text_processor_attention_{str_date}.pkl", "wb"),
    )
    pickle.dump(
        output_conf,
        open(f"bin/output_text_processor_attention_{str_date}.pkl", "wb"),
    )

    if verbose:
        print(f"Time Taken {datetime.now() - start}")

    if verbose:
        print("Building Model")

    train_translator = TrainTranslator(Config, vectorizer)

    # Configure the loss and optimizer
    train_translator.compile(
        optimizer=tf.optimizers.Adam(),
        loss=MaskedLoss(),
    )

    train_translator.fit(
        vectorizer.dataset, epochs=Config.epochs, callbacks=[batch_loss]
    )

    save_layer(
        train_translator.encoder,
        train_translator.decoder,
        f"bin/attention_{str_date}.h5",
    )
