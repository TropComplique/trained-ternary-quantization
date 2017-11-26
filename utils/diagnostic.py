import numpy as np
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from torch.autograd import Variable
import torch.nn.functional as F


"""Tools for diagnostic of a learned model.

Arguments:
    true: a numpy array of shape (n_samples,) of type int
        with integers in range 0..(n_classes - 1).

    pred: a numpy array of shape (n_samples, n_classes) of type float,
        represents probabilities.

    decode: a dict that maps a class index to a human readable form.
"""


NUM_CLASSES = 200
NUM_VAL_SAMPLES_PER_CLASS = 50  # number of samples per class in the validation dataset


def top_k_accuracy(true, pred, k=[2, 3, 4, 5]):
    n_samples = len(true)
    hits = []
    for i in k:
        hits += [np.equal(pred.argsort(1)[:, -i:], true.reshape(-1, 1)).sum()/n_samples]
    return hits


def per_class_accuracy(true, pred):

    true_ohehot = np.zeros((len(true), NUM_CLASSES))
    for i in range(len(true)):
        true_ohehot[i, true[i]] = 1.0

    pred_onehot = np.equal(pred, pred.max(1).reshape(-1, 1)).astype('int')

    per_class_acc = (true_ohehot*pred_onehot).sum(0)/float(NUM_VAL_SAMPLES_PER_CLASS)
    return per_class_acc  # shape: [NUM_CLASSES]


def most_inaccurate_k_classes(per_class_acc, k, decode):
    most = per_class_acc.argsort()[:k]
    for i in most:
        print(decode[i], per_class_acc[i])


def entropy(pred):

    prob = pred.astype('float64')
    log = np.log2(prob)
    result = -(prob*log).sum(1)

    return result


def model_calibration(true, pred, n_bins=10):
    """
    On Calibration of Modern Neural Networks,
    https://arxiv.org/abs/1706.04599
    """

    hits = np.equal(pred.argmax(1), true)

    fraction_of_positives, mean_predicted_value = calibration_curve(
        hits, pred.max(1), n_bins=n_bins
    )

    plt.plot(mean_predicted_value, fraction_of_positives, '-ok')
    plt.plot([0.0, 1.0], [0.0, 1.0], '--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('confidence')
    plt.ylabel('accuracy')
    plt.title('reliability curve')


def count_params(model):
    # model - pytorch's nn.Module object
    count = 0
    for p in model.parameters():
        count += p.numel()
    return count


def most_confused_classes(val_true, val_pred, decode, min_n_confusions):

    conf_mat = confusion_matrix(val_true, val_pred.argmax(1))

    # not interested in correct predictions
    conf_mat -= np.diag(conf_mat.diagonal())

    # confusion(class A -> class B) + confusion(class B -> class A)
    conf_mat += conf_mat.T

    confused_pairs = np.where(np.triu(conf_mat) >= min_n_confusions)
    confused_pairs = [(k, confused_pairs[1][i]) for i, k in enumerate(confused_pairs[0])]
    confused_pairs = [(decode[i], decode[j]) for i, j in confused_pairs]

    return confused_pairs


def predict(model, val_iterator_no_shuffle, return_erroneous=False):

    # predictions on the validation dataset
    val_predictions = []
    # it will have shape [n_val_samples, n_classes]

    # corresponding true targets
    val_true_targets = []
    # it will have shape [n_val_samples]

    # also you can return samples where
    # the model is making mistakes
    if return_erroneous:
        erroneous_samples = []
        erroneous_targets = []
        erroneous_predictions = []

    model.eval()

    for x_batch, y_batch in tqdm(val_iterator_no_shuffle):

        x_batch = Variable(x_batch.cuda(), volatile=True)
        y_batch = Variable(y_batch.cuda(), volatile=True)
        logits = model(x_batch)

        # compute probabilities
        probs = F.softmax(logits)

        if return_erroneous:
            _, argmax = probs.max(1)
            hits = argmax.eq(y_batch).data
            miss = 1 - hits
            if miss.nonzero().numel() != 0:
                miss = miss.nonzero()[:, 0]  # indices of errors
                erroneous_samples += [x_batch[miss].cpu().data.numpy()]
                erroneous_targets += [y_batch[miss].cpu().data.numpy()]
                erroneous_predictions += [probs[miss].cpu().data.numpy()]

        val_predictions += [probs.cpu().data.numpy()]
        val_true_targets += [y_batch.cpu().data.numpy()]

    val_predictions = np.concatenate(val_predictions, axis=0)
    val_true_targets = np.concatenate(val_true_targets, axis=0)

    if return_erroneous:
        erroneous_samples = np.concatenate(erroneous_samples, axis=0)
        erroneous_targets = np.concatenate(erroneous_targets, axis=0)
        erroneous_predictions = np.concatenate(erroneous_predictions, axis=0)
        return val_predictions, val_true_targets,\
            erroneous_samples, erroneous_targets, erroneous_predictions

    return val_predictions, val_true_targets
