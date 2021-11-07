from src.attention.train import train
from src.attention.translate import translate
from src.attention.evaluate import evaluate

if __name__ == '__main__':
    # evaluate('bin/translator_2021-11-07 05.37.44', 'data/filtered/train.csv', 'data/prediction_train.csv')
    train()