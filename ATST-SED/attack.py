import torch
import torchaudio
import numpy as np
import random
import os
import datetime
import soundfile as sf
from torch import nn, optim
import logging
from inference import ATSTSEDInferencer
from desed_task.dataio.datasets_atst_sed import read_audio
import time
import math

random.seed(42)

class ArgsNamespace:
    def __init__(self,
                 wav_input,
                 model_path='./src/stage_2_wo_external.ckpt',
                 config_path="./train/confs/stage2.yaml",
                 save_output=False,
                 output_dir='AEs/',
                 attack_iters=200,
                 delta_duration=1.0,
                 tau=0.05,
                 sample_rate=16000,
                 n_fft=2048,
                 hop_len=1024,
                 nb_mel_bands=128,
                 seq_len=256,
                 start_time=None,
                 end_time=None,
                 attack_type=None,  # 'mirage' or 'mute'
                 target_label=None,
                 use_preservation_loss=True,
                 alpha=50,  # preservation_loss weight
                 use_cw=False,
                 use_faag=False,
                 posterior_thresh=0.5):
        self.wav_input = wav_input
        self.model_path = model_path
        self.config_path = config_path
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
        self.start_time = start_time
        self.end_time = end_time
        self.attack_type = attack_type
        self.target_label = target_label
        self.use_preservation_loss = use_preservation_loss
        self.alpha = alpha
        self.use_cw = use_cw
        self.use_faag = use_faag
        self.posterior_thresh = posterior_thresh
        if self.use_cw:  # First difference: existing method don't consider preservation loss
            logging.info("Testing attack performance with C&W settings")
            self.use_preservation_loss = False
            self.use_faag = False

class Editor:
    class_labels = {
        "Alarm_bell_ringing": 0,
        "Blender": 1,
        "Cat": 2,
        "Dishes": 3,
        "Dog": 4,
        "Electric_shaver_toothbrush": 5,
        "Frying": 6,
        "Running_water": 7,
        "Speech": 8,
        "Vacuum_cleaner": 9,
    }

    def __init__(self, args):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.wav_input_path = args.wav_input
        self.save_output = args.save_output
        self.output_dir = args.output_dir
        self.attack_iters = args.attack_iters
        self.delta_duration = args.delta_duration
        self.tau = args.tau
        self.model_path = args.model_path
        self.config_path = args.config_path
        self.seq_len = args.seq_len
        self.start_time = args.start_time
        self.end_time = args.end_time
        self.attack_type = args.attack_type
        self.target_label = args.target_label
        self.alpha = args.alpha
        self.use_cw = args.use_cw
        self.use_faag = args.use_faag
        self.use_preservation_loss = args.use_preservation_loss
        self.posterior_thresh = args.posterior_thresh

        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Load the trained ATST-SED model with fixed parameters
        self.inferencer = ATSTSEDInferencer(
            self.model_path,
            self.config_path,
            overlap_dur=3)
        self.inferencer.to(self.device)
        # Set model to training mode to enable backward pass
        self.inferencer.model.train()

        # Define spectrogram transformer (must match training)
        self.mel_spectrogram = self.inferencer.feature_extractor.sed_feat_extractor

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

    def load_audio(self, input_path=None):
        """
        Load and preprocess the input audio.
        """
        if input_path is None:
            input_path = self.wav_input_path
        try:
            y, sr = torchaudio.load(input_path)
            y = y.to(torch.float32)
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
                y = resampler(y)
            if y.shape[0] > 1:
                y = torch.mean(y, dim=0, keepdim=True)  # Convert to mono

            # Normalize to [-1, 1]
            y = y / torch.max(torch.abs(y))
            target_samples = self.sample_rate * 10
            current_samples = y.shape[1]
            if current_samples < target_samples:
                num_repeats = target_samples // current_samples + 1
                y = y.repeat(1, num_repeats)[:, :target_samples]
            elif current_samples > target_samples:
                start_index = random.randint(0, current_samples - target_samples)
                y = y[:, start_index:start_index + target_samples]

            # Ensure the tensor is 1D to match delta's shape
            return y.squeeze(0)
        except Exception as e:
            logging.error(f"Error loading audio {self.wav_input_path}: {e}")
            return None

    def select_random_segment(self, total_samples):
        # Check if the audio is shorter than the perturbation duration
        if total_samples <= self.delta_length:
            logging.error("Audio is shorter than the perturbation duration.")
            # Adjust the start and end indices to ensure they are divisible by hop_len
            start_idx = 0
            end_idx = (total_samples // self.hop_len) * self.hop_len
            return start_idx, end_idx

        max_start = ((total_samples - self.delta_length) // self.hop_len) * self.hop_len
        start_idx = random.randint(0, max_start // self.hop_len) * self.hop_len
        end_idx = start_idx + self.delta_length

        # Adjust index
        end_idx = (end_idx // self.hop_len) * self.hop_len
        if end_idx > total_samples:
            end_idx = (total_samples // self.hop_len) * self.hop_len
            start_idx = end_idx - self.delta_length
            if start_idx < 0:
                start_idx = 0

        return start_idx, end_idx

    def attack(self):
        y, _, _, _ = read_audio(
            self.wav_input_path, False, False, None
        )
        if y is None:
            logging.error("Failed to load audio. Exiting attack.")
            return
        y = y.to(self.device)

        # Select perturbation segment (start_idx, end_idx)
        if self.start_time is not None and self.end_time is not None:
            start_idx = int(self.start_time * self.sample_rate)
            end_idx = int(self.end_time * self.sample_rate)

            total_samples = y.shape[0]
            if start_idx < 0 or end_idx > total_samples:
                logging.error("Specified start_time and/or end_time are out of bounds. Exiting attack.")
                return

            expected_length = self.delta_length
            actual_length = end_idx - start_idx
            if actual_length != expected_length:
                logging.warning(f"Specified perturbation length ({actual_length} samples) "
                                f"does not match expected delta_length ({expected_length} samples).")

            logging.info(f"Using specified segment from {start_idx} to {end_idx} for perturbation.")
        else:
            start_idx, end_idx = self.select_random_segment(y.shape[0])

        # Check if perturbation covers the entire audio
        if start_idx == 0 and end_idx == y.shape[0]:
            logging.error("Perturbation segment covers the entire audio. Exiting attack.")
            return

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

        # Step 2: Generate output of origin input
        y_org = y.clone().detach()
        output_org, logits_org = self.inferencer(y_org)  # [batch, time, num_classes]
        logits_org = torch.cat(logits_org).transpose(-1, -2)

        # Determine which time frames correspond to the perturbed segment
        start_frame = int(start_idx / self.hop_len)
        end_frame = int(end_idx / self.hop_len)
        num_frames = logits_org.shape[1]
        start_frame = max(0, min(start_frame, num_frames - 1))
        end_frame = max(0, min(end_frame, num_frames - 1))

        # Create target labels based on attack_type and target_label
        target = logits_org.clone().detach()  # Ensure 'logits' is defined before this line
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

            # Step 3.2: Pass through the model
            if self.use_faag:
                output, logits = self.inferencer(y_adv[start_idx:end_idx])  # [batch, frame, num_classes]
                logits = torch.cat(logits).transpose(-1, -2)

                # Step 3.3: Calculate loss
                adv_loss = criterion(logits[:, :, target_class_idx],
                                     target[:, start_frame:end_frame, target_class_idx])
            else:
                output, logits = self.inferencer(y_adv)  # [batch, frame, num_classes]
                logits = torch.cat(logits).transpose(-1, -2)

                # Step 3.3: Calculate loss
                # Compute loss only on the perturbed segment
                adv_loss = criterion(logits[:, start_frame:end_frame, target_class_idx],
                                     target[:, start_frame:end_frame, target_class_idx])
                # preservation consider global context: which may decease ASR but also decease UER
                preservation_mask = torch.ones_like(logits, dtype=torch.float)
                preservation_mask[:, start_frame:end_frame, target_class_idx] = 0
                preservation_loss = bce_preservation_loss(logits, target, preservation_mask)

            if self.use_preservation_loss:
                loss = adv_loss + preservation_loss * self.alpha
            else:
                loss = adv_loss

            # Step 3.4: Backpropagate and step
            loss.backward()
            optimizer.step()

            # Step 3.5: Clip delta to stay within [-tau, tau]
            delta.data = torch.clamp(delta.data, -self.tau, self.tau)

            # Logging
            if iteration % 100 == 0 or iteration == 1:
                logging.info(f"Iteration {iteration}/{self.attack_iters}, adv loss: {adv_loss.item():.6f}" +
                             (f", preservation loss: {preservation_loss.item():.6f}" if self.use_preservation_loss else ""))

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

        if self.start_time is not None and self.end_time is not None:
            perturb_start_time = self.start_time
            perturb_end_time = self.end_time
        else:
            perturb_start_time = start_idx / self.sample_rate
            perturb_end_time = end_idx / self.sample_rate

        perturb_start_time_str = f"{perturb_start_time:.3f}"
        perturb_end_time_str = f"{perturb_end_time:.3f}"

        adv_filename = (f"adv_{os.path.splitext(os.path.basename(self.wav_input_path))[0]}"
                        f"_{self.attack_type}_{self.target_label}_"
                        f"{perturb_start_time_str}-{perturb_end_time_str}.wav")
        if self.use_cw:
            adv_filename = "cw_" + adv_filename
        adv_path = os.path.join(self.output_dir, adv_filename)
        if self.save_output:
            sf.write(adv_path, adv_audio, self.sample_rate)
            logging.info(f"Adversarial audio saved to {adv_path}")

        # Move original to device (already on device)
        original = y.clone().detach()

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
            attack_time: Time consumption.

        Returns:
            dict: A dictionary containing evaluation results.
        """
        self.inferencer.eval()
        with torch.no_grad():
            decision_org, logits_org = self.inferencer(original)
            decision_adv, logits_adv = self.inferencer(y_adv)  # [batch, time, num_classes]
        self.inferencer.model.train()

        logits_org = torch.cat(logits_org).transpose(-1, -2)
        logits_adv = torch.cat(logits_adv).transpose(-1, -2)

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
                fe += (target_pred_adv == True).sum().item()
            elif self.attack_type == "mirage":
                se += (target_pred_adv == True).sum().item()
                fe += (target_pred_adv == False).sum().item()
        else:
            logging.warning("attack_type or target_label not specified. Evaluation skipped.")

        evaluation_result["EP"], evaluation_result["ASR"], evaluation_result["UER"] = calc_matrix(se, fe, ue, ne)
        logging.info(f"Edit Precision (EP): {evaluation_result['EP'] * 100:.2f}%")
        logging.info(f"Attack Success Rate (ASR): {evaluation_result['ASR'] * 100:.2f}%")
        logging.info(f"Unintended Edit Rate (UER): {evaluation_result['UER'] * 100:.2f}%")
        logging.info(f"Attack time consumption: {evaluation_result['Time']:.2f}")
        # Compute SNR
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
    log_filename = os.path.join(output_dir,
                                f'{log_prefix}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

def get_rand_label():
    return random.choice([
        "Alarm_bell_ringing",
        "Blender",
        "Cat",
        "Dishes",
        "Dog",
        "Electric_shaver_toothbrush",
        "Frying",
        "Running_water",
        "Speech",
        "Vacuum_cleaner",
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
        "avg SNR": 0.0,  # Signal to Noise Ratio
        "avg EP": 0.0,  # Average Edit Precision
        "avg ASR": 0.0,
        "avg UER": 0.0,
        "avg Time": 0.0,
    }

    attack_type = 'mirage'
    use_cw = False
    use_faag = True
    use_preservation_loss = True
    alpha = 50
    tau = 0.05
    input_wav_dir = '../DESED_dataset/strong_label_real (3373)/'
    wav_list = read_data(input_wav_dir, 500)
    log_prefix = 'main_' + attack_type
    if use_cw:
        log_prefix += '_cw'
        use_faag = False
    elif use_faag:
        log_prefix += '_faag'
        use_preservation_loss = False
    if use_preservation_loss:
        log_prefix += f'_alpha({alpha})'
    log_prefix += f'_tau({tau})'
    setup_logging('logs', log_prefix)
    logging.info("Starting Attack on ATST-SED Model for Sound Event Detection")
    logging.info(f"Total samples: {len(wav_list)}")

    for i, wav_input in enumerate(wav_list):
        logging.info(
            f"\n\n===============================================================================================================================\n"
            f"=====                                                      No.{i}                                                           =====\n"
            f"===============================================================================================================================\n\n")
        args = ArgsNamespace(
            wav_input=wav_input,
            model_path='./src/stage_2_wo_external.ckpt',
            config_path="./train/confs/stage2.yaml",
            save_output=False,
            output_dir='AEs/',
            attack_iters=1000,
            delta_duration=3.0,
            tau=tau,
            # start_time=0.0,
            # end_time=4.0,
            use_preservation_loss=use_preservation_loss,
            attack_type=attack_type,
            target_label=get_rand_label(),
            alpha=alpha,
            use_cw=use_cw,
            use_faag=use_faag
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
