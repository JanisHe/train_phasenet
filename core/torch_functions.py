import numpy as np
from tqdm.auto import tqdm
import torch

from core.utils import is_nan


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    """
    def __init__(self, patience=7, verbose=False, delta=0, path_checkpoint=None, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path_checkpoint (str, None): Path for the checkpoint to be saved to. If not None chechpoints are saved.
                            Default: None
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path_checkpoint
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        Saves model when validation loss decrease.
        """
        if self.verbose:
            self.trace_func(f'\nValidation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f})')

        if self.path:
            torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class Metrics:
    """

    """
    def __init__(self, probabilities, residuals, predictions=None,
                 true_pick_prob=0.5, arrival_residual=10):
        self.probabilities = probabilities
        self.residuals = residuals
        self.true_pick_prob = true_pick_prob
        self.arrival_residual = arrival_residual
        self.predictions = predictions

        self.true_positive = None
        self.false_positive = None
        self.false_negative = None

        if self.predictions is not None:
            self.true_false_positives()

    def __str__(self):
        pass

    def true_false_positives(self):
        self.true_positive = 0
        self.false_positive = 0
        self.false_negative = 0
        for prediction, probability, residual in zip(self.predictions, self.probabilities, self.residuals):
            if not is_nan(prediction):
                if probability >= self.true_pick_prob and abs(residual) <= self.arrival_residual:
                    self.true_positive += 1
                elif probability >= self.true_pick_prob and abs(residual) > self.arrival_residual:
                    self.false_positive += 1
                elif probability < self.true_pick_prob or abs(residual) > self.arrival_residual:
                    self.false_negative += 1

    @property
    def precision(self, eps=1e-6) -> float:
        return (self.true_positive + eps) / (self.true_positive + self.false_positive + eps)

    @property
    def recall(self, eps=1e-6) -> float:
        return (self.true_positive + eps) / (self.true_positive + self.false_negative + eps)

    @property
    def f1_score(self, eps=1e-6) -> float:
        return 2 * ((self.precision * self.recall + eps) / (
                    self.precision + self.recall + eps))


class VectorCrossEntropyLoss:
    """
    Vector cross entropy as definded in Zhu & Beroza (2018).

    H(p, q) = \sum_{i=-1}^{3} \sum_x p_i(x) * log(q_i(x))
    p: true probabiliyt distribution
    q: predicted distribution
    """
    def __init__(self):
        pass

    def __call__(self, y_pred, y_true, eps=1e-10):
        """
        :param y_pred:
        :param y_true:
        :param eps:
        """
        h = y_true * torch.log(y_pred + eps)
        h = h.mean(-1).sum(-1)                 # Mean along sample dimension and sum along pick dimension
        h = h.mean()                           # Mean over batch axis

        return -h


class MeanSquaredError:
    def __init__(self):
        pass

    def __call__(self, y_pred, y_true):
        """

        """
        mse = (y_true - y_pred) ** 2
        mse = mse.mean(-1).sum(-1)
        mse = mse.mean()

        return mse


def train_model(model, train_loader, validation_loader, loss_fn,
                optimizer=None, epochs=50, patience=5, lr_scheduler=None):
    """

    """
    # Initialize lists to track losses
    train_loss = []
    valid_loss = []
    avg_train_loss = []
    avg_valid_loss = []

    # Load optimizer
    if not optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Initialize early stopping class
    early_stopping = EarlyStopping(patience=patience, verbose=False, path_checkpoint=None)

    # Loop over each epoch to start training
    for epoch in range(epochs):
        # Train model (loop over each batch; batch_size is defined in DataLoader)
        # TODO (idea): test model with validation (compute metrics)
        num_batches = len(train_loader)
        with tqdm(total=num_batches, desc=f"Epoch {epoch + 1}", ncols=100,
                  bar_format="{l_bar}{bar} [Elapsed time: {elapsed} {postfix}]") as pbar:
            for batch_id, batch in enumerate(train_loader):
                # Compute prediction and loss
                pred = model(batch["X"].to(model.device))
                loss = loss_fn(y_pred=pred, y_true=batch["y"].to(model.device))

                # Do backpropagation
                optimizer.zero_grad()   # clear the gradients of all optimized variables
                loss.backward()         # backward pass: compute gradient of the loss with respect to model parameters
                optimizer.step()        # perform a single optimization step (parameter update)

                # Compute loss for each batch and write loss to predefined lists
                train_loss.append(loss.item())

                # Update progressbar
                pbar.set_postfix({"loss": str(np.round(loss.item(), 4))})
                pbar.update()

            # Validate the model
            model.eval()     # Close the model for validation / evaluation
            with torch.no_grad():   # Disable gradient calculation
                for batch in validation_loader:
                    pred = model(batch["X"].to(model.device))
                    valid_loss.append(loss_fn(pred, batch["y"].to(model.device)).item())

            # Determine average training and validation loss
            avg_train_loss.append(sum(train_loss) / len(train_loss))
            avg_valid_loss.append(sum(valid_loss) / len(valid_loss))

            # Update progressbar
            pbar.set_postfix(
                {"loss": str(np.round(avg_train_loss[-1], 4)),
                 "val_loss": str(np.round(avg_valid_loss[-1], 4))}
            )

        # Re-open model for next epoch
        model.train()

        # Clear training and validation loss lists for next epoch
        train_loss = []
        valid_loss = []

        # Update learning rate
        if lr_scheduler:
            lr_scheduler.step()

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(avg_valid_loss[-1], model)

        if early_stopping.early_stop:
            print("Validation loss does not decrease further. Early stopping")
            break

    return model, avg_train_loss, avg_valid_loss
