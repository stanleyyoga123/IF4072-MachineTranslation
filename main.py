if __name__ == '__main__':
    use = 'attention'

    if use == 't5':
        from src.t5.train import train
        from src.t5.simulate import simulate

        # Do Something
        simulate()

    elif use == 'attention':
        from src.attention.train import train
        from src.attention.translate import translate
        from src.attention.evaluate import evaluate

        # Do Something
        translate('2021-11-08 21.56.14')