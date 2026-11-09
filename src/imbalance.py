"""Class-imbalance handling for the referable / non-referable label (Part C).

In our training split about 41% of images are referable (grade >= 2), so the
imbalance is mild. Two standard fixes are provided so the ablation can test
whether either one helps:

    class weight   the loss counts each referable image more (pos_weight = n_neg / n_pos),
                   so both classes contribute equally to the total loss.
    oversampling   referable images are drawn more often when building each epoch,
                   so batches are about 50/50. No images are copied on disk.

Use only one of the two at a time: together they would over-correct.

Usage (PyTorch):
    from src.imbalance import pos_weight, oversampling_weights
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight(y_train)))
    sampler = torch.utils.data.WeightedRandomSampler(oversampling_weights(y_train), len(y_train))
"""

import numpy as np


def class_counts(y):
    """Number of negative and positive labels in a 0/1 array."""
    y = np.asarray(y).astype(int)
    if not set(np.unique(y)) <= {0, 1}:
        raise ValueError("labels must be 0/1")
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("both classes must be present")
    return n_neg, n_pos


def pos_weight(y):
    """Weight for the positive class in BCEWithLogitsLoss: n_neg / n_pos."""
    n_neg, n_pos = class_counts(y)
    return n_neg / n_pos


def class_weights(y):
    """Per-class weights {0: w0, 1: w1} that make both classes count equally.

    Same convention as scikit-learn's class_weight="balanced": n / (2 * n_class).
    """
    n_neg, n_pos = class_counts(y)
    n = n_neg + n_pos
    return {0: n / (2 * n_neg), 1: n / (2 * n_pos)}


def oversampling_weights(y):
    """Per-image sampling weights so each class is drawn with probability 0.5.

    For torch.utils.data.WeightedRandomSampler(weights, num_samples=len(y), replacement=True).
    """
    y = np.asarray(y).astype(int)
    w = class_weights(y)
    return np.where(y == 1, w[1], w[0]).astype(np.float64)
