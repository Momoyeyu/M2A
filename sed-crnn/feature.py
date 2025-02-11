import os
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import MelSpectrogram, Resample
from sklearn.preprocessing import LabelEncoder

# Ensure reproducibility
torch.manual_seed(42)

# ###################################################################
#                              Utilities
# ###################################################################

def create_folder(folder_path):
    """Create a folder if it doesn't exist."""
    os.makedirs(folder_path, exist_ok=True)

# ###################################################################
#                              Audio Loading
# ###################################################################

def load_audio(filename, mono=True, target_fs=44100):
    """
    Load an audio file and return the audio data and sample rate using torchaudio.

    Parameters:
        filename (str): Path to the audio file.
        mono (bool): Whether to convert to mono by averaging channels.
        target_fs (int): Target sample rate for resampling.

    Returns:
        y (torch.Tensor): Loaded audio data.
        sample_rate (int): Sample rate of the audio data.
    """
    try:
        y, sample_rate = torchaudio.load(filename)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None, None

    if mono:
        y = torch.mean(y, dim=0, keepdim=True)

    if sample_rate != target_fs:
        resampler = Resample(orig_freq=sample_rate, new_freq=target_fs)
        y = resampler(y)
        sample_rate = target_fs

    # Normalize audio to float32 in range [-1, 1]
    y = y.float() / (2**31)

    return y.squeeze(0), sample_rate  # Remove channel dimension if mono

# ###################################################################
#                          Description Loading
# ###################################################################

def load_desc_file(desc_file, class_labels):
    """
    Load the description file containing event annotations.

    Parameters:
        desc_file (str): Path to the description file.
        class_labels (dict): Mapping from class names to integer labels.

    Returns:
        desc_dict (dict): Dictionary mapping audio filenames to their event annotations.
    """
    desc_dict = {}
    with open(desc_file, 'r') as f:
        for line in f:
            words = line.strip().split('\t')
            filename = os.path.basename(words[0])
            if filename not in desc_dict:
                desc_dict[filename] = []
            # (start time, end time, event class)
            start_time = float(words[2])
            end_time = float(words[3])
            event_class = class_labels.get(words[-1], -1)
            if event_class != -1:
                desc_dict[filename].append([start_time, end_time, event_class])
    return desc_dict

# ###################################################################
#                          Feature Extraction
# ###################################################################

def extract_mbe(y, sr, n_fft=2048, hop_length=None, nb_mel=40):
    """
    Extract Mel-Band Energies (MBE) from audio data using torchaudio.

    Parameters:
        y (torch.Tensor): Audio time series.
        sr (int): Sample rate.
        n_fft (int): FFT window size.
        hop_length (int): Hop length for STFT. Defaults to n_fft // 2.
        nb_mel (int): Number of Mel bands.

    Returns:
        mel_spec (torch.Tensor): Mel-Band Energy spectrogram [time, mel_bands].
    """
    if hop_length is None:
        hop_length = n_fft // 2

    mel_spectrogram = MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=nb_mel,
        center=True,
        power=2.0
    )
    mel_spec = mel_spectrogram(y)  # [n_mel, time]

    # Convert power spectrogram to log scale (log-mel)
    mel_spec = torch.log(mel_spec + 1e-10)  # Avoid log(0)

    return mel_spec.transpose(0, 1)  # [time, mel_bands]

# ###################################################################
#                           Main Processing
# ###################################################################

def main():
    is_mono = True
    class_labels = {
        'brakes squeaking': 0,
        'car': 1,
        'children': 2,
        'large vehicle': 3,
        'people speaking': 4,
        'people walking': 5
    }

    # Locations of data
    folds_list = [1, 2, 3, 4]
    evaluation_setup_folder = '../TUT-sound-events-2017-development/evaluation_setup'
    audio_folder = '../TUT-sound-events-2017-development/audio/street'

    # Output folder for features
    feat_folder = '../TUT-sound-events-2017-development/feat/'
    create_folder(feat_folder)

    # User-defined parameters
    nfft = 2048
    win_len = nfft
    hop_len = win_len // 2
    nb_mel_bands = 40
    target_sr = 44100

    # -------------------------------------------------------------------
    # Feature extraction and label generation
    # -------------------------------------------------------------------
    # Load labels
    train_file = os.path.join(evaluation_setup_folder, 'street_fold1_train.txt')
    evaluate_file = os.path.join(evaluation_setup_folder, 'street_fold1_evaluate.txt')
    desc_dict = load_desc_file(train_file, class_labels)
    desc_dict.update(load_desc_file(evaluate_file, class_labels))  # Contains labels for all audio in the dataset

    # Extract features for all audio files and save them along with labels
    for audio_filename in os.listdir(audio_folder):
        audio_file = os.path.join(audio_folder, audio_filename)
        print(f'Extracting features and labels for: {audio_file}')
        y, sr_loaded = load_audio(audio_file, mono=is_mono, target_fs=target_sr)
        if y is None:
            print(f"Failed to load {audio_file}. Skipping.")
            continue

        mbe = extract_mbe(y, sr_loaded, n_fft=nfft, hop_length=hop_len, nb_mel=nb_mel_bands)

        if audio_filename not in desc_dict:
            print(f"No annotations found for {audio_filename}. Skipping.")
            continue

        label = torch.zeros((mbe.shape[0], len(class_labels)), dtype=torch.float32)
        tmp_data = torch.tensor(desc_dict[audio_filename])  # [num_events, 3]
        frame_start = torch.floor(tmp_data[:, 0] * sr_loaded / hop_len).long()
        frame_end = torch.ceil(tmp_data[:, 1] * sr_loaded / hop_len).long()
        se_class = tmp_data[:, 2].long()

        for ind, val in enumerate(se_class):
            # Ensure indices are within the range
            start = torch.clamp(frame_start[ind], min=0)
            end = torch.clamp(frame_end[ind], max=mbe.shape[0])
            label[start:end, val] = 1

        # Convert tensors to numpy for saving
        mbe_np = mbe.numpy()
        label_np = label.numpy()

        tmp_feat_file = os.path.join(feat_folder, f'{audio_filename}_{"mon" if is_mono else "bin"}.npz')
        np.savez(tmp_feat_file, mbe=mbe_np, label=label_np)

    # -------------------------------------------------------------------
    # Feature Normalization
    # -------------------------------------------------------------------
    for fold in folds_list:
        train_file = os.path.join(evaluation_setup_folder, f'street_fold{fold}_train.txt')
        evaluate_file = os.path.join(evaluation_setup_folder, f'street_fold{fold}_evaluate.txt')
        train_dict = load_desc_file(train_file, class_labels)
        test_dict = load_desc_file(evaluate_file, class_labels)

        X_train, Y_train, X_test, Y_test = [], [], [], []

        # Load training features
        for key in train_dict.keys():
            tmp_feat_file = os.path.join(feat_folder, f'{key}_{"mon" if is_mono else "bin"}.npz')
            if not os.path.exists(tmp_feat_file):
                print(f"Feature file {tmp_feat_file} does not exist. Skipping.")
                continue
            data = np.load(tmp_feat_file)
            tmp_mbe = torch.tensor(data['mbe'], dtype=torch.float32)
            tmp_label = torch.tensor(data['label'], dtype=torch.float32)
            X_train.append(tmp_mbe)
            Y_train.append(tmp_label)

        # Load testing features
        for key in test_dict.keys():
            tmp_feat_file = os.path.join(feat_folder, f'{key}_{"mon" if is_mono else "bin"}.npz')
            if not os.path.exists(tmp_feat_file):
                print(f"Feature file {tmp_feat_file} does not exist. Skipping.")
                continue
            data = np.load(tmp_feat_file)
            tmp_mbe = torch.tensor(data['mbe'], dtype=torch.float32)
            tmp_label = torch.tensor(data['label'], dtype=torch.float32)
            X_test.append(tmp_mbe)
            Y_test.append(tmp_label)

        if not X_train or not X_test:
            print(f"No data found for fold {fold}. Skipping normalization.")
            continue

        # Concatenate all tensors
        X_train = torch.cat(X_train, dim=0)
        Y_train = torch.cat(Y_train, dim=0)
        X_test = torch.cat(X_test, dim=0)
        Y_test = torch.cat(Y_test, dim=0)

        # Normalize the training data
        mean = X_train.mean(dim=0, keepdim=True)
        std = X_train.std(dim=0, keepdim=True)
        X_train_normalized = (X_train - mean) / (std + 1e-10)
        X_test_normalized = (X_test - mean) / (std + 1e-10)  # Use training mean and std

        normalized_feat_file = os.path.join(feat_folder, f'mbe_{"mon" if is_mono else "bin"}_fold{fold}.pt')
        torch.save({
            'X_train': X_train_normalized,
            'Y_train': Y_train,
            'X_test': X_test_normalized,
            'Y_test': Y_test
        }, normalized_feat_file)
        print(f'Normalized feature file saved: {normalized_feat_file}')

if __name__ == "__main__":
    main()
