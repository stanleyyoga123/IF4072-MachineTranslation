class Config:
    test_data = 'data/test.csv'
    train_data = 'data/train.csv'
    model_name = 't5-base'
    tokenizer_name = 't5-base'
    learning_rate = 3e-4
    adam_epsilon = 1e-8
    weight_decay = 0.0
    train_batch_size = 64
    eval_batch_size = 64
    max_seq_length = 64
    gradient_accumulation_steps = 16
    warmup_steps = 0
    n_gpu = 1
    num_train_epochs = 5
    model_dir = 'model'