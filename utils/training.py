from torch.autograd import Variable
import torch.nn.functional as F
import time


def optimization_step(model, loss, x_batch, y_batch, optimizer):
    """Make forward pass and update model parameters with gradients."""

    # forward pass
    x_batch, y_batch = Variable(x_batch.cuda()), Variable(y_batch.cuda(async=True))
    logits = model(x_batch)

    # compute logloss
    loss_value = loss(logits, y_batch)
    batch_loss = loss_value.data[0]

    # compute accuracies
    pred = F.softmax(logits)
    batch_accuracy, batch_top5_accuracy = _accuracy(y_batch, pred, top_k=(1, 5))

    # compute gradients
    optimizer.zero_grad()
    loss_value.backward()

    # update params
    optimizer.step()

    return batch_loss, batch_accuracy, batch_top5_accuracy


def train(model, loss, optimization_step_fn,
          train_iterator, val_iterator, n_epochs=30,
          patience=10, threshold=0.01, lr_scheduler=None):
    """
    Train 'model' by minimizing 'loss' using 'optimization_step_fn'
    for parameter updates.
    """

    # collect losses and accuracies here
    all_losses = []

    running_loss = 0.0
    running_accuracy = 0.0
    running_top5_accuracy = 0.0
    n_steps = 0
    start_time = time.time()
    model.train()

    for epoch in range(0, n_epochs):

        # main training loop
        for x_batch, y_batch in train_iterator:

            batch_loss, batch_accuracy, batch_top5_accuracy = optimization_step_fn(
                model, loss, x_batch, y_batch
            )
            running_loss += batch_loss
            running_accuracy += batch_accuracy
            running_top5_accuracy += batch_top5_accuracy
            n_steps += 1

        # evaluation
        model.eval()
        test_loss, test_accuracy, test_top5_accuracy = _evaluate(
            model, loss, val_iterator
        )

        # collect evaluation information and print it
        all_losses += [(
            epoch,
            running_loss/n_steps, test_loss,
            running_accuracy/n_steps, test_accuracy,
            running_top5_accuracy/n_steps, test_top5_accuracy
        )]
        print('{0}  {1:.3f} {2:.3f}  {3:.3f} {4:.3f}  {5:.3f} {6:.3f}  {7:.3f}'.format(
            *all_losses[-1], time.time() - start_time
        ))

        # it watches test accuracy
        # and if accuracy isn't improving then training stops
        if _is_early_stopping(all_losses, patience, threshold):
            print('early stopping!')
            break

        if lr_scheduler is not None:
            # possibly change the learning rate
            lr_scheduler.step(test_accuracy)

        running_loss = 0.0
        running_accuracy = 0.0
        running_top5_accuracy = 0.0
        n_steps = 0
        start_time = time.time()
        model.train()

    return all_losses


def _accuracy(true, pred, top_k=(1,)):

    max_k = max(top_k)
    batch_size = true.size(0)

    _, pred = pred.topk(max_k, 1)
    pred = pred.t()
    correct = pred.eq(true.view(1, -1).expand_as(pred))

    result = []
    for k in top_k:
        correct_k = correct[:k].view(-1).float().sum(0)
        result.append(correct_k.div_(batch_size).data[0])

    return result


def _evaluate(model, loss, val_iterator):

    loss_value = 0.0
    accuracy = 0.0
    top5_accuracy = 0.0
    total_samples = 0

    for x_batch, y_batch in val_iterator:

        x_batch = Variable(x_batch.cuda(), volatile=True)
        y_batch = Variable(y_batch.cuda(async=True), volatile=True)
        n_batch_samples = y_batch.size()[0]
        logits = model(x_batch)

        # compute logloss
        batch_loss = loss(logits, y_batch).data[0]

        # compute accuracies
        pred = F.softmax(logits)
        batch_accuracy, batch_top5_accuracy = _accuracy(y_batch, pred, top_k=(1, 5))

        loss_value += batch_loss*n_batch_samples
        accuracy += batch_accuracy*n_batch_samples
        top5_accuracy += batch_top5_accuracy*n_batch_samples
        total_samples += n_batch_samples

    return loss_value/total_samples, accuracy/total_samples, top5_accuracy/total_samples


def _is_early_stopping(all_losses, patience, threshold):
    """It decides if training must stop."""

    # get current and all past (validation) accuracies
    accuracies = [x[4] for x in all_losses]

    if len(all_losses) > (patience + 4):

        # running average with window size 5
        average = (accuracies[-(patience + 4)] +
                   accuracies[-(patience + 3)] +
                   accuracies[-(patience + 2)] +
                   accuracies[-(patience + 1)] +
                   accuracies[-patience])/5.0

        # compare current accuracy with
        # running average accuracy 'patience' epochs ago
        return accuracies[-1] < (average + threshold)
    else:
        # if not enough epochs to compare with
        return False
