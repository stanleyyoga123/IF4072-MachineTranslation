import tensorflow as tf
from tensorflow.keras.losses import Loss, SparseCategoricalCrossentropy


class MaskedLoss(Loss):
    def __init__(self):
        self.name = "masked_loss"
        self.loss = SparseCategoricalCrossentropy(from_logits=True, reduction="none")

    def __call__(self, y_true, y_pred):
        loss = self.loss(y_true, y_pred)
        mask = tf.cast(y_true != 0, tf.float32)
        loss *= mask
        return tf.reduce_sum(loss)
