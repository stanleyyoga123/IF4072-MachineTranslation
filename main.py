from src.attention.train import train
from src.attention.translate import translate
from src.attention.evaluate import evaluate

if __name__ == '__main__':
    train()
    # evaluate('bin/translator_2021-11-08 06.21.56', 'data/filtered/train.csv', 'data/prediction_train_256.csv')
    # evaluate('bin/translator_2021-11-08 06.21.56', 'data/filtered/val.csv', 'data/prediction_val_256.csv')
    # evaluate('bin/translator_2021-11-08 06.21.56', 'data/filtered/test.csv', 'data/prediction_test_256.csv')
    # translate('2021-11-08 06.21.56')