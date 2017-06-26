import keras.backend as K

def precision(y_true, y_pred):
    """
    See: https://github.com/fchollet/keras/commit/a56b1a55182acf061b1eb2e2c86b48193a0e88f7

    Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """
    See: https://github.com/fchollet/keras/commit/a56b1a55182acf061b1eb2e2c86b48193a0e88f7

    Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def mae2d(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true), axis=None)

def positive_mae(y_true, y_pred):
    zero = K.constant(0)
    mask = K.maximum(K.cast(K.greater(y_true, zero), dtype='float32'), K.cast(K.greater(y_pred, zero), dtype='float32'))
    return K.sum(K.abs(y_pred*mask - y_true), axis=None) / (K.sum(mask, axis=None) + K.constant(1))
    
def mae_per_class(y_true, y_pred):
    zero = K.constant(0)
    pos_mask = K.cast(K.greater(y_true, zero), dtype='float32')
    pos_mae = K.sum(K.abs(y_pred*pos_mask - y_true), axis=-1) / (K.sum(pos_mask, axis=-1) + K.constant(1))
    neg_mask = K.cast(K.equal(y_true, zero), dtype='float32')
    neg_mae = K.sum(K.abs(y_pred*neg_mask), axis=-1) / (K.sum(neg_mask, axis=-1) + K.constant(1))
    return pos_mae + K.sqrt(neg_mae)

def count_diff(y_true, y_pred):
    return K.mean(K.sum(y_pred, axis=(1,2,3)) - K.sum(y_true, axis=(1,2,3)), axis=None)
