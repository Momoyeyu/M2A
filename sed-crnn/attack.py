import torch
import torchaudio
import numpy as np
import random
import os
import datetime
import soundfile as sf
from torch import nn, optim
from crnn import CRNNForest
import logging
import math
import time

random.seed(42)

class ArgsNamespace:
    """
    A simple class to hold attack parameters.
    """
    def __init__(self,
                 wav_input,
                 model_path,
                 posterior_thresh=0.5,
                 save_output=False,
                 output_dir='AEs/',
                 attack_iters=300,
                 delta_duration=3.0,
                 tau=0.02,
                 sample_rate=44100,
                 n_fft=2048,
                 hop_len=1024,
                 nb_mel_bands=40,
                 seq_len=256,
                 nb_ch=1,
                 num_classes=6,
                 cnn_nb_filt=128,
                 cnn_pool_size=[1, 1, 1],
                 rnn_nb=[32, 32],
                 fc_nb=[32],
                 dropout_rate=0.5,
                 start_time=None,
                 end_time=None,
                 attack_type=None,
                 target_label=None,
                 use_preservation_loss=True,
                 alpha=10,
                 use_cw=False,
                 use_faag=False):
        self.wav_input = wav_input
        self.model_path = model_path
        self.posterior_thresh = posterior_thresh
        self.save_output = save_output
        self.output_dir = output_dir
        self.attack_iters = attack_iters
        self.delta_duration = delta_duration
        self.tau = tau
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.nb_mel_bands = nb_mel_bands
        self.seq_len = seq_len
        self.nb_ch = nb_ch
        self.num_classes = num_classes
        self.cnn_nb_filt = cnn_nb_filt
        self.cnn_pool_size = cnn_pool_size
        self.rnn_nb = rnn_nb
        self.fc_nb = fc_nb
        self.dropout_rate = dropout_rate
        self.start_time = start_time
        self.end_time = end_time
        self.attack_type = attack_type
        self.target_label = target_label
        self.use_preservation_loss = use_preservation_loss
        self.alpha = alpha
        self.use_cw = use_cw
        self.use_faag = use_faag
        if self.use_cw:  # First difference: preservation loss
            logging.info("Testing attack performance with C&W settings")
            self.use_preservation_loss = False
            self.use_faag = False
        if self.use_preservation_loss:
            logging.info(f"Using preservation loss with alpha: {self.alpha}")

class Editor:
    class_labels = {
        'brakes squeaking': 0,
        'car': 1,
        'children': 2,
        'large vehicle': 3,
        'people speaking': 4,
        'people walking': 5
    }

    def __init__(self, args):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.wav_input_path = args.wav_input
        self.save_output = args.save_output
        self.output_dir = args.output_dir
        self.attack_iters = args.attack_iters
        self.delta_duration = args.delta_duration  # in seconds
        self.tau = args.tau  # Maximum perturbation amplitude
        self.model_path = args.model_path
        self.posterior_thresh = args.posterior_thresh
        self.seq_len = args.seq_len
        self.num_classes = args.num_classes
        self.start_time = args.start_time
        self.end_time = args.end_time
        self.attack_type = args.attack_type
        self.target_label = args.target_label
        self.use_preservation_loss = args.use_preservation_loss
        self.alpha = args.alpha
        self.use_cw = args.use_cw
        self.use_faag = args.use_faag

        # Load the trained CRNN model with fixed parameters
        self.crnn = CRNNForest(self.model_path, self.device, args, self.num_classes)
        self.crnn.train()  # Set model to training mode to enable backward pass

        # Define spectrogram transformer (must match training)
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=args.sample_rate,
            n_fft=args.n_fft,
            hop_length=args.hop_len,
            n_mels=args.nb_mel_bands,
            center=True,
            power=2.0
        ).to(self.device)

        # Audio parameters
        self.sample_rate = args.sample_rate
        self.n_fft = args.n_fft
        self.hop_len = args.hop_len
        self.nb_mel_bands = args.nb_mel_bands
        self.seq_len = args.seq_len

        # Determine delta_length based on start_time and end_time
        if self.start_time is not None and self.end_time is not None:
            if self.end_time <= self.start_time:
                logging.error("end_time must be greater than start_time. Falling back to random segment.")
                self.start_time = None
                self.end_time = None
                self.delta_length = int(self.delta_duration * self.sample_rate)
            else:
                self.delta_length = int((self.end_time - self.start_time) * self.sample_rate)
        else:
            self.delta_length = int(self.delta_duration * self.sample_rate)  # Number of samples in delta

    def load_audio(self, path=None):
        """
        Load and preprocess the input audio.
        """
        if path is None:
            path = self.wav_input_path
        try:
            y, sr = torchaudio.load(path)
            y = y.to(torch.float32)
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
                y = resampler(y)
            if y.shape[0] > 1:
                y = torch.mean(y, dim=0, keepdim=True)  # Convert to mono

            # Normalize to [-1, 1]
            y = y / torch.max(torch.abs(y))
            target_samples = self.sample_rate * 25
            current_samples = y.shape[1]
            if current_samples > target_samples:
                start_index = random.randint(0, current_samples - target_samples)
                y = y[:, start_index:start_index + target_samples]
            # Ensure the tensor is 1D to match delta's shape
            return y.squeeze(0)
        except Exception as e:
            logging.error(f"Error loading audio {self.wav_input_path}: {e}")
            return None

    def select_random_segment(self, total_samples):
        if total_samples <= self.delta_length:
            logging.error("Audio is shorter than the perturbation duration.")
            return 0, total_samples  # Return integer indices
        max_start = total_samples - self.delta_length
        start_idx = random.randint(0, max_start)
        end_idx = start_idx + self.delta_length
        return start_idx, end_idx

    def calc_spectrogram(self, y):
        """
        Calculate the Mel spectrogram of the audio.
        """
        spectrogram = self.mel_spectrogram(y)
        spectrogram = torch.log1p(spectrogram)  # Log scaling
        # Normalize
        mean = spectrogram.mean()
        std = spectrogram.std()
        spectrogram = (spectrogram - mean) / (std + 1e-10)
        return spectrogram.unsqueeze(0)  # [batch, n_mels, time]

    def attack(self):
        y = self.load_audio()
        if y is None:
            logging.error("Failed to load audio. Exiting attack.")
            return
        y = y.to(self.device)

        # Step 0: Select perturbation segment (covert time to idx)
        if self.start_time is not None and self.end_time is not None:
            # get index where delta should be applied to the origin audio
            start_idx = int(self.start_time * self.sample_rate)
            end_idx = int(self.end_time * self.sample_rate)
            # Verify if the index is within the audio length range
            total_samples = y.shape[0]
            if start_idx < 0 or end_idx > total_samples:
                logging.error("Specified start_time and/or end_time are out of bounds. Exiting attack.")
                return
            # Verify if the disturbance length is consistent
            expected_length = self.delta_length
            actual_length = end_idx - start_idx
            if actual_length != expected_length:
                logging.warning(f"Specified perturbation length ({actual_length} samples) "
                                f"does not match expected delta_length ({expected_length} samples).")
            logging.info(f"Using specified segment from {start_idx} to {end_idx} for perturbation.")
        else:
            # Use randomly selected segments
            start_idx, end_idx = self.select_random_segment(y.shape[0])
        # Check if perturbation covers the entire audio
        logging.info(
            f"Perturbation segment: start_time={start_idx / self.sample_rate:.3f}s, end_time={end_idx / self.sample_rate:.3f}s")

        # Step 1: Initialize perturbation delta on device
        if self.use_cw:  # Second difference: full delta settings used. only used when attacking task like SEC
            delta = torch.zeros(y.shape[0], requires_grad=True, device=self.device)
        else:
            delta = torch.zeros(end_idx - start_idx, requires_grad=True, device=self.device)
        delta.data.uniform_(-self.tau, self.tau)  # Initialize delta within [-tau, tau]

        optimizer = optim.Adam([delta], lr=1e-3)
        criterion = nn.BCELoss()

        y_org = y.clone().detach()

        spec_org = self.calc_spectrogram(y_org)  # [time] -> [1, n_mels, frame]
        spec_org = spec_org.unsqueeze(1)  # Add channel dimension -> [1, 1, n_mels, frame]
        output_org = self.crnn(spec_org)  # [batch, frame, num_classes]

        # Determine which time frames correspond to the perturbed segment
        start_frame = int(start_idx / self.hop_len)
        end_frame = int(end_idx / self.hop_len)
        num_frames = output_org.shape[1]
        start_frame = max(0, min(start_frame, num_frames - 1))
        end_frame = max(0, min(end_frame, num_frames - 1))

        # Create target labels based on attack_type and target_label
        target = output_org.clone().detach()  # Ensure 'output' is defined before this line

        # Step 2: Edit target output
        if self.attack_type is not None and self.target_label is not None:
            if self.attack_type not in ["mirage", "mute"]:
                logging.error("Invalid attack_type. It must be either 'mirage' or 'mute'. Exiting attack.")
                return
            if self.target_label not in self.class_labels:
                logging.error(f"Invalid target_label '{self.target_label}'. Exiting attack.")
                return

            target_class_idx = self.class_labels[self.target_label]

            if self.attack_type == "mirage":
                # Set the target_label's output to 1 in the perturbed segment
                target[:, start_frame:end_frame, target_class_idx] = 1.0
                logging.info(
                    f"Attack type: 'mirage' - setting label '{self.target_label}' to 1 in the perturbed segment.")
            elif self.attack_type == "mute":
                # Set the target_label's output to 0 in the perturbed segment
                target[:, start_frame:end_frame, target_class_idx] = 0.0
                logging.info(
                    f"Attack type: 'mute' - setting label '{self.target_label}' to 0 in the perturbed segment.")
        else:
            logging.error("No attack_type or target_label specified.")
            return

        attack_start_time = time.time()

        # Step 3: Enter the optimization loop
        for iteration in range(1, self.attack_iters + 1):
            optimizer.zero_grad()

            # Step 3.1: Apply perturbation
            y_adv = y.clone().detach()
            if self.use_cw:  # Second difference: full delta settings used. only used when attacking task like SEC
                y_adv += delta
            else:
                y_adv[start_idx:end_idx] += delta
            y_adv = torch.clamp(y_adv, -1.0, 1.0)  # Ensure audio stays in [-1, 1]

            # Step 3.2: Get model output and Calculate optimization loss
            spec = self.calc_spectrogram(y_adv)  # [time] -> [1, n_mels, frame]
            spec = spec.unsqueeze(1)  # Add channel dimension -> [1, 1, n_mels, frame]
            if self.use_faag:
                # [batch, channel, feature, frame] -> [batch, frame, num_classes]
                output = self.crnn(spec[:, :, :, start_frame:end_frame])

                adv_loss = criterion(output[:, :, target_class_idx],
                                     target[:, start_frame:end_frame, target_class_idx])
            else:
                output = self.crnn(spec)  # [batch, channel, feature, time] -> [batch, time, num_classes]
                adv_loss = criterion(output[:, start_frame:end_frame, target_class_idx],
                                     target[:, start_frame:end_frame, target_class_idx])
                # preservation loss consider global contexts, which may decease UER but also decease ASR
                preservation_mask = torch.ones_like(output, dtype=torch.float)
                preservation_mask[:, start_frame:end_frame, target_class_idx] = 0
                preservation_loss = bce_preservation_loss(output, target, preservation_mask)

            if self.use_preservation_loss:
                loss = adv_loss + preservation_loss * self.alpha
            else:
                loss = adv_loss

            # Step 3.3: Backpropagate and step
            loss.backward()
            optimizer.step()

            # Step 3.4: Clip delta to stay within [-tau, tau]
            delta.data = torch.clamp(delta.data, -self.tau, self.tau)

            # Logging
            if iteration % 100 == 0 or iteration == 1:
                logging.info(f"Iteration {iteration}/{self.attack_iters}, adv loss: {adv_loss.item():.6f}" +
                             (f", preservation loss: {preservation_loss.item():.6f}" if self.use_preservation_loss else ""))

            # Early stopping if loss is sufficiently low
            if loss.item() < 1e-4:
                logging.info(f"Attack succeeded at iteration {iteration}")
                break

        attack_end_time = time.time()

        # Step 4: Apply the final perturbation
        with torch.no_grad():
            y_adv = y.clone().detach()
            if self.use_cw:  # Second difference: full delta settings used. only used when attacking task like SEC
                y_adv += delta
            else:
                y_adv[start_idx:end_idx] += delta
            y_adv = torch.clamp(y_adv, -1.0, 1.0)

        # Step 5: Save the adversarial audio
        adv_audio = y_adv.cpu().numpy()

        # Calculate the start and end time of the disturbance (in seconds)
        if self.start_time is not None and self.end_time is not None:
            perturb_start_time = self.start_time
            perturb_end_time = self.end_time
        else:
            # If start_time and end_time are not specified, calculate the time based on start_idx and end_idx
            perturb_start_time = start_idx / self.sample_rate
            perturb_end_time = end_idx / self.sample_rate

        perturb_start_time_str = f"{perturb_start_time:.3f}"
        perturb_end_time_str = f"{perturb_end_time:.3f}"

        adv_filename = (f"adv_{os.path.splitext(os.path.basename(self.wav_input_path))[0]}"
                        f"_{self.attack_type}_{self.target_label}_"
                        f"{perturb_start_time_str}-{perturb_end_time_str}.wav")
        if self.use_cw:
            adv_filename = "sec_" + adv_filename
        adv_path = os.path.join(self.output_dir, adv_filename)
        if self.save_output:
            sf.write(adv_path, adv_audio, self.sample_rate)
            logging.info(f"Adversarial audio saved to {adv_path}")

        # Move original to device (already on device)
        original = y.clone().detach()  # y is already on device

        # Step 6: Evaluate the attack
        return self.evaluate_attack(original, y_adv, start_idx, end_idx, attack_end_time - attack_start_time)

    def evaluate_attack(self, original, y_adv, start_idx, end_idx, attack_time):
        """
        Evaluate the success of the attack based on attack_type and target_label,
        and calculate the Signal-to-Noise Ratio (SNR) between original and adversarial audio.

        Args:
            original (torch.Tensor): Original audio tensor.
            y_adv (torch.Tensor): Adversarial audio tensor.
            start_idx (int): Start sample index of perturbation.
            end_idx (int): End sample index of perturbation.
            attack_time: Attack time consumption.

        Returns:
            dict: A dictionary containing 'success' (bool) and 'SNR' (float).
        """
        spec_org = self.calc_spectrogram(original)  # [time] -> [1, n_mels, frame]
        spec_org = spec_org.unsqueeze(1)  # [1, 1, n_mels, frame] ([batch, channel, feature, frame])
        spec_adv = self.calc_spectrogram(y_adv)
        spec_adv = spec_adv.unsqueeze(1)

        self.crnn.eval()
        with torch.no_grad():
            logits_org = self.crnn(spec_org)  # [batch, channel, feature, frame] -> [batch, frame, num_classes]
            logits_adv = self.crnn(spec_adv)  # [batch, channel, feature, frame] -> [batch, frame, num_classes]

        start_frame = int(start_idx / self.hop_len)
        end_frame = int(end_idx / self.hop_len)
        num_frames = logits_adv.shape[1]
        start_frame = max(0, min(start_frame, num_frames - 1))
        end_frame = max(0, min(end_frame, num_frames - 1))

        evaluation_result = {
            "EP": 0,
            "ASR": 0,
            "UER": 0,
            "SNR": 0.0,
            "Time": attack_time
        }

        pred_org = (logits_org > self.posterior_thresh)
        pred_adv = (logits_adv > self.posterior_thresh)
        se = 0
        fe = 0
        ne = (pred_org == pred_adv).sum().item()
        ue = (pred_org != pred_adv).sum().item()

        if self.attack_type is not None and self.target_label is not None:
            if self.target_label not in self.class_labels:
                logging.error(f"Invalid target_label '{self.target_label}'. Evaluation skipped.")
                return evaluation_result

            target_class_idx = self.class_labels[self.target_label]

            target_pred_org = pred_org[0, start_frame:end_frame, target_class_idx]
            target_pred_adv = pred_adv[0, start_frame:end_frame, target_class_idx]
            ue -= (target_pred_org != target_pred_adv).sum().item()
            ne -= (target_pred_org == target_pred_adv).sum().item()
            if self.attack_type == "mute":
                se += (target_pred_adv == False).sum().item()
                fe += (target_pred_adv == True ).sum().item()
            elif self.attack_type == "mirage":
                se += (target_pred_adv == True ).sum().item()
                fe += (target_pred_adv == False).sum().item()
        else:
            logging.warning("attack_type or target_label not specified. Evaluation skipped.")

        evaluation_result["EP"], evaluation_result["ASR"], evaluation_result["UER"] = calc_matrix(se, fe, ue, ne)
        logging.info(f"Edit Precision (EP): {evaluation_result['EP'] * 100:.2f}%")
        logging.info(f"Attack Success Rate (ASR): {evaluation_result['ASR'] * 100:.2f}%")
        logging.info(f"Unintended Edit Rate (UER): {evaluation_result['UER'] * 100:.2f}%")
        logging.info(f"Attack time consumption: {evaluation_result['Time']:.2f}")
        try:
            snr_value = calculate_snr(original, y_adv)
            evaluation_result["SNR"] = snr_value
            logging.info(f"Signal-to-Noise Ratio (SNR): {snr_value:.2f} dB")
        except ValueError as ve:
            logging.error(f"SNR Calculation Error: {ve}")
            evaluation_result["SNR"] = None
        return evaluation_result

def calc_matrix(se, fe, ue, ne):
    EP = (se + ne) / (se + fe + ue + ne)
    ASR = se / (se + fe)
    UER = ue / (ue + ne)
    return EP, ASR, UER

def read_data(data_dir, max_length):
    wav_list = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".wav"):
                wav_path = os.path.join(root, file)
                wav_list.append(wav_path)
    repeat_times = math.ceil(max_length / len(wav_list))
    repeated_list = wav_list * repeat_times
    result = repeated_list[:max_length]
    return result

def calculate_snr(original, adversarial):
    """
    Calculate the Signal-to-Noise Ratio (SNR) between the original and adversarial audio.

    Args:
        original (torch.Tensor or np.ndarray): Original audio tensor or numpy array.
        adversarial (torch.Tensor or np.ndarray): Adversarial audio tensor or numpy array.

    Returns:
        float: The calculated SNR in decibels (dB).
    """
    if isinstance(original, torch.Tensor):
        original = original.cpu().numpy()
    if isinstance(adversarial, torch.Tensor):
        adversarial = adversarial.cpu().numpy()

    # Ensure both arrays are of the same shape
    if original.shape != adversarial.shape:
        raise ValueError("Original and adversarial audio must have the same shape for SNR calculation.")

    # Calculate the power of the original signal and the noise
    power_signal = np.sum(original ** 2)
    power_noise = np.sum((original - adversarial) ** 2)

    # Handle the case where power_noise is zero (no noise)
    if power_noise == 0:
        return float('inf')

    snr = 10 * np.log10(power_signal / power_noise)
    return snr

def setup_logging(output_dir, log_prefix):
    """
    Set up logging to console and file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Create file handler
    log_filename = os.path.join(output_dir, f'{log_prefix}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

def get_rand_label():
    return random.choice([
        'brakes squeaking',
        'car',
        'children',
        'large vehicle',
        'people speaking',
        'people walking'
    ])

def bce_preservation_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute BCE loss only on masked region (non-target region).

    :param pred: Predictions (batch_size, N, C)
    :param target: Ground truth labels (batch_size, N, C)
    :param mask: Mask tensor (batch_size, N, C), indicating non-target region
    :return: BCE loss computed on masked region (non-target region)
    """
    # Compute BCE loss manually
    bce_loss = - (target * torch.log(torch.sigmoid(pred) + 1e-8) + (1 - target) * torch.log(
        1 - torch.sigmoid(pred) + 1e-8))

    # Apply mask
    masked_loss = bce_loss * mask

    # Normalize by valid elements to avoid division by zero
    return masked_loss.sum() / (mask.sum() + 1e-8)

def main():
    eval_matrix = {
        "avg SNR": 0.0, # Signal to Noise Ratio
        "avg EP": 0.0,  # Average Edit Precision
        "avg ASR": 0.0,
        "avg UER": 0.0,
        "avg Time": 0.0,
    }

    attack_type = 'mirage'
    use_cw = False
    use_faag = False
    use_preservation_loss = True
    alpha = 10
    tau = 0.01
    input_wav_dir = '../TUT-sound-events-2017-development/audio/street'
    wav_list = read_data(input_wav_dir, 50)
    # wav_list = ['audio/a001.wav']
    log_prefix = f"main_{attack_type}"
    if use_cw:
        log_prefix += '_cw'
        use_preservation_loss = False
        use_faag = False
    else:
        if use_faag:
            log_prefix += '_faag'
            use_preservation_loss = False
        if use_preservation_loss:
            log_prefix += f'_alpha({alpha})'
    log_prefix += f'_tau({tau})'
    setup_logging('logs', log_prefix)
    logging.info("Starting Attack on CRNN Model for Sound Event Detection")
    logging.info(f"Total samples: {len(wav_list)}")

    if len(wav_list) == 0:
        logging.warning("No wav files found")
        return

    for i, wav_input in enumerate(wav_list):
        logging.info(
            f"\n\n===============================================================================================================================\n"
            f"=====                                                      No.{i}                                                           =====\n"
            f"===============================================================================================================================\n\n")
        args = ArgsNamespace(
            wav_input=wav_input,
            model_path=[
                'models/clean_mon_2024_12_18_20_59_30_fold_1_model.pth',
                'models/clean_mon_2024_12_18_20_59_30_fold_2_model.pth',
                'models/clean_mon_2024_12_18_20_59_30_fold_3_model.pth',
                'models/clean_mon_2024_12_18_20_59_30_fold_4_model.pth'
            ],
            save_output=False,
            output_dir='AEs/',
            attack_iters=1000,
            delta_duration=3.0,
            tau=tau,
            # start_time=1.0,
            # end_time=4.0,
            attack_type=attack_type,
            target_label=get_rand_label(),
            use_preservation_loss=use_preservation_loss,
            use_faag=use_faag,
            alpha=alpha,
            use_cw=use_cw
        )

        # Initialize attacker first to ensure the output directory exists
        attacker = Editor(args)
        # Perform attack
        result = attacker.attack()
        eval_matrix["avg SNR"] += result["SNR"]
        eval_matrix["avg EP"] += result["EP"]
        eval_matrix["avg ASR"] += result["ASR"]
        eval_matrix["avg UER"] += result["UER"]
        eval_matrix["avg Time"] += result["Time"]

        logging.info("Attack completed.")

    logging.info(f"Average SNR: {eval_matrix['avg SNR'] / len(wav_list)} dB")
    logging.info(f"Average EP: {eval_matrix['avg EP'] / len(wav_list) * 100:.2f}%")
    logging.info(f"Average ASR: {eval_matrix['avg ASR'] / len(wav_list) * 100:.2f}%")
    logging.info(f"Average UER: {eval_matrix['avg UER'] / len(wav_list) * 100:.2f}%")
    logging.info(f"Average Time: {eval_matrix['avg Time'] / len(wav_list):.2f}")

if __name__ == '__main__':
    main()
