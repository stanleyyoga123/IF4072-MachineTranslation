import argparse
import pytorch_lightning as pl

from model import T5FineTuner
from config import Config

def train():
    hparams = {
        'model_name': Config.model_name,
        'tokenizer_name': Config.tokenizer_name,
        'learning_rate': Config.learning_rate,
        'adam_epsilon': Config.adam_epsilon,
        'weight_decay': Config.weight_decay,
        'train_batch_size': Config.train_batch_size,
        'eval_batch_size': Config.eval_batch_size,
        'max_seq_length': Config.max_seq_length,
        'gradient_accumulation_steps': Config.gradient_accumulation_steps,
        'warmup_steps': Config.warmup_steps,
        'n_gpu': Config.n_gpu,
        'num_train_epochs': Config.num_train_epochs,
    }
    hparams = argparse.Namespace(**hparams)
    train_params = dict(
        accumulate_grad_batches=hparams.gradient_accumulation_steps,
        gpus=hparams.n_gpu,
        max_epochs=hparams.num_train_epochs,
    )
    model = T5FineTuner(hparams)
    trainer = pl.Trainer(**train_params)
    trainer.fit(model)
    model.model.save_pretrained(Config.model_dir)