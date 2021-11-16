import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a", "--attention", action="store_true", required=False, help="Attention Model"
    )
    parser.add_argument(
        "-t5", "--t5", action="store_true", required=False, help="T5 Model"
    )
    args = parser.parse_args()

    if args.t5:
        from src.t5.train import train
        from src.t5.simulate import simulate

        # Do Something
        train()

    elif args.attention:
        from src.attention.train import train
        from src.attention.translate import translate
        from src.attention.evaluate import evaluate

        # Do Something
        train()