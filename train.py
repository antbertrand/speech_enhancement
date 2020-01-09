import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import backup_utils, params_utils
from model import net, dataset

###############################################################################
# Main


def main():

    verbose = True

    params = params_utils.Params()

    # --- Model
    model = net.MyCNN(params)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=params.learning_rate)
    loss_fn = torch.nn.MSELoss(reduction='mean')

    # Print model's state_dict
    if verbose:
        print("\nModel's state_dict:")
        for param_tensor in model.state_dict():
            print(param_tensor, "\t\t", model.state_dict()
                  [param_tensor].size())

    # Datasets
    # TODO need to separate training set in train and validation sets
    train_set = dataset.CustomDataset(params.train_raw_csv_path,
                                      './data/noise/babble_train.wav', params)
    val_set = dataset.CustomDataset(params.train_raw_csv_path,
                                    './data/noise/babble_val.wav', params)
    # test_set = dataset.CustomDataset(params.test_raw_csv_path,
    #                                  params.test_noise_csv_path, params)

    train(model, optimizer, loss_fn, train_set, val_set, params)


###############################################################################
# Main functions


def train(model, optimizer, loss_fn, train_set, val_set, params, verbose=True):

    logs = TrainingHistory()

    while not logs.early_stop:

        model.train()  # Do it in each loop, because model.eval() called at validation

        if verbose:
            start_t = datetime.datetime.now()
            print('\n' + '='*50)
            # logs.epoch not yet up-to-date, hence `+1`
            print("epoch #{}".format(logs.epoch + 1))

        # To compute mean loss over the full batch
        loss_hist = []
        len_hist = []

        # Each sound is considered as a batch
        for X, Y in train_set.batch_loader():

            # shape of X and Y : (B, C, H, W), where :
            # B is the number of frames of the STFT of the sound
            # C is the number of channels (equals to 1 in our case)
            # H is the height of each input sample (equals to params.nfft//2 +1)
            # W is the width of each input sample (equals to params.n_frames)

            # Feed forward
            Y_pred = model(X)

            # Go from batch to reconstructed STFT
            y_pred = reconstruct(Y_pred)
            y = reconstruct(Y)

            # Compute loss
            loss = loss_fn(y, y_pred)
            loss = loss.sum()

            # Learn
            model.zero_grad()
            loss.backward()
            optimizer.step()

            # Backup loss
            loss_hist.append(loss.data)
            len_hist.append(y_pred.shape[2])

            loss_mean = sum(torch.tensor(loss_hist) *
                            torch.tensor(len_hist)) / sum(len_hist)

            # Print info
            if verbose:
                elapsed_t = datetime.datetime.now() - start_t
                elapsed_t_str = '{:02.0f}:{:02.0f}'.format(
                    *divmod(elapsed_t.seconds, 60))
                print("      loss: {:.3f} (elapsed: {})".format(
                    loss_mean, elapsed_t_str), end='\r')

        # Re-print info, but not to be erased
        if verbose:
            print("      loss: {:.3f} (elapsed: {})".format(
                loss_mean, elapsed_t_str))

        # Compute val loss
        val_loss = validate(model, loss_fn, val_set)
        if verbose:
            elapsed_t = datetime.datetime.now() - start_t
            elapsed_t_str = '{:02.0f}:{:02.0f}'.format(
                *divmod(elapsed_t.seconds, 60))
            print("  val loss: {:.3f} (elapsed: {})".format(val_loss,
                                                            elapsed_t_str))

        # --- Save model

        # Get the path
        if not params.backup.save_model:
            saved_model_path = None
        else:
            # 'logs' are not yet up-to-date, hence '+1'
            saved_model_path = backup_utils.get_model_saving_path(
                logs.epoch + 1, loss_mean, params)

        # Update logs
        logs.add_values(loss_mean, val_loss, saved_model_path)

        # Save model
        if params.backup.save_model:
            assert saved_model_path == backup_utils.save_checkpoint(
                model, optimizer, loss_mean, logs, params)

        if logs.epoch >= params.max_epoch:
            break

    return logs


def validate(model, loss_fn, dataset):  # TODO check if doesn't need params

    # Set model to evaluation mode
    model.eval()

    loss_hist = []
    len_hist = []

    model.eval()

    for X, Y in dataset.batch_loader():

        # Freeze model's params
        for p in model.parameters():
            p.requires_grad = False

        # Forward
        Y_pred = model(X)

        # Compute loss
        loss = loss_fn(Y_pred, Y)
        if loss.shape != torch.Size([]):
            loss = loss.sum()

        loss_hist.append(loss.data)
        len_hist.append(len(Y_pred))
        # TODO may be the length of the sound, not the total number of frames, but they are normally equal
        # TODO finally, do I compute the loss on the unstacked STFT or not ?

    loss = sum(loss_hist)/sum(len_hist)

    # Unfreeze model's params
    # TODO maybe all params doesn't require grad,
    # TODO and shouldn't have been frozen in first place ... Check it
    for p in model.parameters():
        p.requires_grad = True

    return loss

###############################################################################
# Functions


def reconstruct(X, apodisation=None):
    # TODO add apodisation, alignement and stride arguments

    # TODO
    # If apodisation is None, it means that we only keep one
    # frame for each output
    # Else, we need to do an overlapping summation

    # X of shape (B, C, H, W), with C=1
    # output of shape (C, H, B) in our case (i.e. stride=1, otherwise third dim < B, or idk yet)

    # nframes need to be odd, it's easier
    nframes = X.shape[3]

    # case for alignment == 'center' and apodisation == None
    idx_to_keep = int((nframes - 1) / 2)

    x = X[:, 0, :, idx_to_keep].T.unsqueeze(0)

    return x


###############################################################################
# Classes

class TrainingHistory():

    def __init__(self):

        self.__epoch = 0
        self.__saved_models_paths = []

        self.__train_loss = []
        self.__val_loss = []

        self.patience = 10
        self.margin = 0.005

    # ---------------------------------------------------------------- #
    @property
    def best_model_path(self):

        if not self.early_stop:
            print('WARNING: Training unfinished.')

        idx_bst = np.argmin(self.val_loss)

        return self.saved_models_paths[idx_bst]

    @property
    def best_model_val_loss(self):
        return np.min(self.val_loss)

    @property
    def epoch(self):
        return self.__epoch

    @property
    def saved_models_paths(self):
        return self.__saved_models_paths

    @property
    def early_stop(self):

        if len(self.val_loss) <= self.patience:
            return False

        val_loss_min = min(self.val_loss)
        val_loss_late_min = min(self.val_loss[-self.patience:])

        return bool(val_loss_late_min > ((1 + self.margin) * val_loss_min))

    @property
    def train_loss(self):
        return self.__train_loss

    @property
    def val_loss(self):
        return self.__val_loss

    # ---------------------------------------------------------------- #
    def load_from_other(self, other_dict):

        for k, v in other_dict.items():
            self.__dict__[k] = copy.deepcopy(v)

    def load_from_checkpoint(self, chkpt_path):
        # TODO
        pass

    def add_values(self, train_loss, val_loss, model_path):

        train_loss = float(train_loss)
        val_loss = float(val_loss)

        self.__epoch += 1
        self.__saved_models_paths.append(model_path)

        self.__train_loss.append(train_loss)
        self.__val_loss.append(val_loss)

    # ------------------------------------------------------------------

    def plot_loss(self):
        # TODO label legends, autosave etc
        plt.plot(self.train_loss)
        plt.plot(self.val_loss)


if __name__ == "__main__":
    main()
