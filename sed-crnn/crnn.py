import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import utils
import metrics

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


#######################################################################################
# Custom Dataset Class
#######################################################################################

class SoundEventDataset(Dataset):
    def __init__(self, features, labels):
        """
        Args:
            features (np.ndarray): Feature array of shape (num_samples, channels, freq, time)
            labels (np.ndarray): Label array of shape (num_samples, time, num_classes)
        """
        self.features = features
        self.labels = labels

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        # Convert to torch tensors
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return feature, label


#######################################################################################
# Model Definition
#######################################################################################

class CRNNForest(nn.Module):
    def __init__(self, model_paths, device, args, num_classes):
        super(CRNNForest, self).__init__()
        self.model_paths = model_paths
        self.device = device
        self.num_classes = num_classes
        self.models = nn.ModuleList([
            load_model(
                model_path=path,
                device=device,
                input_channels=args.nb_ch,
                num_classes=num_classes,
                cnn_nb_filt=args.cnn_nb_filt,
                cnn_pool_size=args.cnn_pool_size,
                rnn_nb=args.rnn_nb,
                fc_nb=args.fc_nb,
                dropout_rate=args.dropout_rate
            ) for path in model_paths
        ])

    def forward(self, x):
        outputs = []
        for model in self.models:
            output = model(x)
            outputs.append(output)
        final_output = torch.stack(outputs).sum(dim=0) / len(self.models)
        return final_output

class CRNN(nn.Module):
    def __init__(self, input_channels, num_classes, cnn_nb_filt, cnn_pool_size, rnn_nb, fc_nb, dropout_rate):
        super(CRNN, self).__init__()

        layers = []
        current_channels = input_channels
        for i, pool_size in enumerate(cnn_pool_size):
            layers.append(nn.Conv2d(
                in_channels=current_channels,
                out_channels=cnn_nb_filt,
                kernel_size=3,
                stride=(1, 1),  # Explicitly set stride to prevent downsampling
                padding=1
            ))
            layers.append(nn.BatchNorm2d(cnn_nb_filt))
            layers.append(nn.ReLU())
            layers.append(
                nn.MaxPool2d(kernel_size=(1, pool_size), stride=(1, pool_size)))  # Ensure stride matches kernel size
            layers.append(nn.Dropout(dropout_rate))
            current_channels = cnn_nb_filt  # Update for next layer

        self.cnn = nn.Sequential(*layers)

        # Initialize GRU layers with dynamic input_size
        self.rnn = nn.ModuleList()
        input_size = cnn_nb_filt  # Start with cnn_nb_filt
        for r in rnn_nb:
            self.rnn.append(nn.GRU(
                input_size=input_size,
                hidden_size=r,
                batch_first=True,
                bidirectional=True
            ))
            input_size = r * 2  # Update input_size for the next GRU layer

        # Fully connected layers
        fc_layers = []
        for _f in fc_nb:
            fc_layers.append(nn.Linear(input_size, _f))  # input_size is updated after last GRU
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(dropout_rate))
            input_size = _f  # Update for the next FC layer
        self.fc = nn.Sequential(*fc_layers)

        # Output layer
        self.output = nn.Linear(fc_nb[-1], num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, channels, freq, time)
        x = self.cnn(x)  # Shape: (batch, 128, 40, 256)

        # Average over the frequency dimension to reduce dimensionality
        x = x.mean(dim=2)  # Shape: (batch, 128, 256)

        # Permute to (batch, time, channels)
        x = x.permute(0, 2, 1)  # Shape: (batch, 256, 128)

        # RNN layers
        for gru in self.rnn:
            x, _ = gru(x)  # Shape after each GRU: (batch, 256, 64)

        # Fully connected layers
        x = self.fc(x)  # Shape: (batch, 256, 32)

        # Output layer
        x = self.output(x)  # Shape: (batch, 256, 6)
        x = self.sigmoid(x)  # Shape: (batch, 256, 6)

        return x  # (batch, 256, 6)


#######################################################################################
# Utility Functions
#######################################################################################

def load_data(feat_folder, is_mono, fold):
    """
    Load normalized features and labels for a specific fold.

    Parameters:
        feat_folder (str): Directory where features are stored.
        is_mono (bool): Whether features are mono-channel.
        fold (int): Fold number.

    Returns:
        X_train (np.ndarray): Training features.
        Y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Testing features.
        Y_test (np.ndarray): Testing labels.
    """
    feat_file_fold = os.path.join(feat_folder, f'mbe_{"mon" if is_mono else "bin"}_fold{fold}.npz')
    if not os.path.exists(feat_file_fold):
        raise FileNotFoundError(f"Feature file {feat_file_fold} does not exist.")

    with np.load(feat_file_fold) as dmp:
        X_train, Y_train, X_test, Y_test = dmp['arr_0'], dmp['arr_1'], dmp['arr_2'], dmp['arr_3']
    return X_train, Y_train, X_test, Y_test


def preprocess_data(X, Y, X_test, Y_test, seq_len, nb_ch):
    """
    Preprocess the data by splitting into sequences and adjusting channels.

    Parameters:
        X (np.ndarray): Training features.
        Y (np.ndarray): Training labels.
        X_test (np.ndarray): Testing features.
        Y_test (np.ndarray): Testing labels.
        seq_len (int): Sequence length.
        nb_ch (int): Number of channels.

    Returns:
        X, Y, X_test, Y_test (np.ndarray): Preprocessed features and labels.
    """
    # Assuming utils.split_in_seqs and utils.split_multi_channels are available
    X = utils.split_in_seqs(X, seq_len)
    Y = utils.split_in_seqs(Y, seq_len)
    X_test = utils.split_in_seqs(X_test, seq_len)
    Y_test = utils.split_in_seqs(Y_test, seq_len)

    X = utils.split_multi_channels(X, nb_ch)
    X_test = utils.split_multi_channels(X_test, nb_ch)
    return X, Y, X_test, Y_test


def plot_functions(nb_epoch, tr_loss, val_loss, f1, er, fig_path):
    """
    Plot training and validation loss, F1 score, and ER over epochs.

    Parameters:
        nb_epoch (int): Number of epochs.
        tr_loss (list): Training loss history.
        val_loss (list): Validation loss history.
        f1 (list): F1 score history.
        er (list): ER history.
        fig_path (str): Path to save the plot.
    """
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.plot(range(1, nb_epoch + 1), tr_loss, label='Train Loss')
    plt.plot(range(1, nb_epoch + 1), val_loss, label='Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(2, 1, 2)
    plt.plot(range(1, nb_epoch + 1), f1, label='F1 Score')
    plt.plot(range(1, nb_epoch + 1), er, label='ER')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')

    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    print(f'Figure saved: {fig_path}')


def load_model(model_path, device, input_channels=1, num_classes=6, cnn_nb_filt=128,
              cnn_pool_size=[1,1,1], rnn_nb=[32,32], fc_nb=[32], dropout_rate=0.5):
    """
    Load the trained CRNN model.

    Parameters:
        model_path (str): Path to the model's state_dict.
        device (torch.device): Device to load the model on.
        input_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        cnn_nb_filt (int): Number of CNN filters.
        cnn_pool_size (list): List of pooling sizes for CNN layers.
        rnn_nb (list): List of hidden sizes for GRU layers.
        fc_nb (list): List of hidden sizes for FC layers.
        dropout_rate (float): Dropout rate.

    Returns:
        model (nn.Module): Loaded CRNN model.
    """
    model = CRNN(
        input_channels=input_channels,
        num_classes=num_classes,
        cnn_nb_filt=cnn_nb_filt,
        cnn_pool_size=cnn_pool_size,
        rnn_nb=rnn_nb,
        fc_nb=fc_nb,
        dropout_rate=dropout_rate
    )
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


#######################################################################################
# MAIN SCRIPT STARTS HERE
#######################################################################################

def main():
    import time  # Ensure the time module is imported

    is_mono = True  # True: mono-channel input, False: binaural input

    feat_folder = '../TUT-sound-events-2017-development/feat/'
    fig_name = f'{"mon" if is_mono else "bin"}_{time.strftime("%Y_%m_%d_%H_%M_%S")}.png'

    nb_ch = 1 if is_mono else 2
    batch_size = 128  # Decrease this if you want to run on smaller GPU's
    seq_len = 256  # Frame sequence length. Input to the CRNN.
    nb_epoch = 500  # Training epochs
    patience = int(0.25 * nb_epoch)  # Patience for early stopping

    # Number of frames in 1 second, required to calculate F and ER for 1 sec segments.
    # Make sure the nfft and sr are the same as in feature.py
    sr = 44100
    nfft = 2048
    frames_1_sec = int(sr / (nfft / 2.0))

    print('\n\nUNIQUE ID:', fig_name)
    print(f'TRAINING PARAMETERS: nb_ch: {nb_ch}, seq_len: {seq_len}, batch_size: {batch_size}, '
          f'nb_epoch: {nb_epoch}, frames_1_sec: {frames_1_sec}')

    # Folder for saving model and training curves
    models_dir = 'models/'
    utils.create_folder(models_dir)

    # CRNN model definition parameters
    cnn_nb_filt = 128  # CNN filter size
    cnn_pool_size = [1, 1, 1]  # Pooling sizes to prevent downsampling time dimension
    rnn_nb = [32, 32]  # Number of RNN nodes. Length = number of RNN layers
    fc_nb = [32]  # Number of FC nodes. Length = number of FC layers
    dropout_rate = 0.5  # Dropout after each layer
    print(f'MODEL PARAMETERS:\n cnn_nb_filt: {cnn_nb_filt}, cnn_pool_size: {cnn_pool_size}, '
          f'rnn_nb: {rnn_nb}, fc_nb: {fc_nb}, dropout_rate: {dropout_rate}')

    avg_er = []
    avg_f1 = []

    for fold in [1, 2, 3, 4]:
        print('\n\n----------------------------------------------')
        print(f'FOLD: {fold}')
        print('----------------------------------------------\n')

        try:
            # Load normalized features and labels, pre-process them
            X_train, Y_train, X_test, Y_test = load_data(feat_folder, is_mono, fold)
            X_train, Y_train, X_test, Y_test = preprocess_data(X_train, Y_train, X_test, Y_test, seq_len, nb_ch)

            # Transpose features to [num_samples, channels, freq, time]
            X_train = X_train.transpose(0, 1, 3,
                                        2)  # From [num_samples, channels, time, freq] to [num_samples, channels, freq, time]
            X_test = X_test.transpose(0, 1, 3, 2)

            # Verify the shapes
            # print(f'After Transpose - X_train shape: {X_train.shape}')  # Expected: [num_samples, channels, freq, time]
            # print(f'After Transpose - X_test shape: {X_test.shape}')

        except Exception as e:
            print(f"Error loading or preprocessing data for fold {fold}: {e}")
            continue

        # Feature Normalization
        scaler = StandardScaler()
        num_samples, channels, freq, time_steps = X_train.shape
        X_train_reshaped = X_train.reshape(-1, channels * freq)
        X_test_reshaped = X_test.reshape(-1, channels * freq)

        scaler.fit(X_train_reshaped)
        joblib.dump(scaler, 'scaler.pkl')
        X_train = scaler.transform(X_train_reshaped).reshape(num_samples, channels, freq, time_steps)
        X_test = scaler.transform(X_test_reshaped).reshape(X_test.shape[0], channels, freq, time_steps)

        # Create Dataset and DataLoader
        train_dataset = SoundEventDataset(X_train, Y_train)
        test_dataset = SoundEventDataset(X_test, Y_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        # Initialize Model
        num_classes = Y_train.shape[-1]
        model = CRNN(
            input_channels=nb_ch,
            num_classes=num_classes,
            cnn_nb_filt=cnn_nb_filt,
            cnn_pool_size=cnn_pool_size,
            rnn_nb=rnn_nb,
            fc_nb=fc_nb,
            dropout_rate=dropout_rate
        )
        model.to(device)
        # print(model)  # Print model architecture for verification

        # Define Loss and Optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # Training Loop with Early Stopping
        best_er = float('inf')
        best_f1 = 0.0
        best_conf_mat = None
        best_epoch = 0
        patience_counter = 0

        tr_loss_history = []
        val_loss_history = []
        f1_history = []
        er_history = []

        for epoch in range(1, nb_epoch + 1):
            model.train()
            epoch_tr_loss = 0.0
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)

                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

                epoch_tr_loss += loss.item() * batch_features.size(0)

            epoch_tr_loss /= len(train_loader.dataset)
            tr_loss_history.append(epoch_tr_loss)

            # Validation
            model.eval()
            epoch_val_loss = 0.0
            all_preds = []
            all_targets = []
            with torch.no_grad():
                for batch_features, batch_labels in test_loader:
                    batch_features = batch_features.to(device)
                    batch_labels = batch_labels.to(device)

                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_labels)
                    epoch_val_loss += loss.item() * batch_features.size(0)

                    all_preds.append(outputs.cpu().numpy())
                    all_targets.append(batch_labels.cpu().numpy())

            epoch_val_loss /= len(test_loader.dataset)
            val_loss_history.append(epoch_val_loss)

            # Concatenate all predictions and targets
            all_preds = np.concatenate(all_preds, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)

            # Apply threshold
            posterior_thresh = 0.5
            pred_thresh = (all_preds > posterior_thresh).astype(int)

            # Compute Metrics
            try:
                score_list = metrics.compute_scores(pred_thresh, all_targets, frames_in_1_sec=frames_1_sec)
                f1_overall = score_list.get('f1_overall_1sec', 0.0)
                er_overall = score_list.get('er_overall_1sec', float('inf'))
            except Exception as e:
                print(f"Error computing metrics: {e}")
                f1_overall = 0.0
                er_overall = float('inf')

            f1_history.append(f1_overall)
            er_history.append(er_overall)

            # Compute Confusion Matrix
            test_pred_cnt = np.sum(pred_thresh, axis=2).reshape(-1)
            Y_test_cnt = np.sum(all_targets, axis=2).reshape(-1)
            conf_mat = confusion_matrix(Y_test_cnt, test_pred_cnt)
            conf_mat = conf_mat / (np.sum(conf_mat, axis=1, keepdims=True) + 1e-10)

            # Check for improvement
            if er_overall < best_er:
                best_er = er_overall
                best_f1 = f1_overall
                best_conf_mat = conf_mat
                best_epoch = epoch
                patience_counter = 0
                # Save the model
                model_path = os.path.join(models_dir, f'{fig_name}_fold_{fold}_model.pth')
                torch.save(model.state_dict(), model_path)
                print(f'Model saved: {model_path}')
            else:
                patience_counter += 1

            # Logging
            print(f'Epoch [{epoch}/{nb_epoch}] - '
                  f'Train Loss: {epoch_tr_loss:.4f} - '
                  f'Val Loss: {epoch_val_loss:.4f} - '
                  f'F1: {f1_overall:.4f} - ER: {er_overall:.4f} - '
                  f'Best ER: {best_er:.4f} at epoch {best_epoch}')

            # Plotting after each epoch
            plot_path = os.path.join(models_dir, f'{fig_name}_fold_{fold}.png')
            plot_functions(epoch, tr_loss_history, val_loss_history, f1_history, er_history, plot_path)

            # Early Stopping
            if patience_counter > patience:
                print(f'Early stopping at epoch {epoch}')
                break

        avg_er.append(best_er)
        avg_f1.append(best_f1)
        print(f'Saved model for the best_epoch: {best_epoch} with best_f1: {best_f1} and best_er: {best_er}')
        print(f'Best Confusion Matrix:\n{best_conf_mat}')
        print(f'Diagonal of Confusion Matrix: {np.diag(best_conf_mat)}')

    print('\n\nMETRICS FOR ALL FOUR FOLDS:')
    print(f'avg_er: {avg_er}, avg_f1: {avg_f1}')
    print(f'MODEL AVERAGE OVER FOUR FOLDS: avg_er: {np.mean(avg_er)}, avg_f1: {np.mean(avg_f1)}')


if __name__ == '__main__':
    main()
