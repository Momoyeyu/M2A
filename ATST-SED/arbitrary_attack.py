import torch
import torchaudio
import random
import os
import soundfile as sf
from numba.core.tracing import event
from torch import nn, optim
import logging
from inference import ATSTSEDInferencer
from desed_task.dataio.datasets_atst_sed import read_audio
from attack import calc_matrix, read_data, calculate_snr, setup_logging, get_rand_label, bce_preservation_loss
import time

random.seed(42)

class ArgsNamespace:
    def __init__(self,
                 wav_input,
                 model_path='./src/stage_2_wo_external.ckpt',
                 config_path="./train/confs/stage2.yaml",
                 posterior_thresh=0.5,
                 output_dir='AEs/',
                 save_output=False,
                 attack_iters=1000,
                 delta_duration=3.0,
                 tau=0.05,
                 sample_rate=16000,
                 n_fft=2048,
                 hop_len=1024,
                 nb_mel_bands=128,
                 seq_len=256,
                 edit_set=[],
                 use_preservation_loss=True,
                 alpha=50,
                 use_cw=False):
        self.wav_input = wav_input
        self.model_path = model_path
        self.config_path = config_path
        self.posterior_thresh = posterior_thresh
        self.output_dir = output_dir
        self.save_output = save_output
        self.attack_iters = attack_iters
        self.delta_duration = delta_duration
        self.tau = tau
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.nb_mel_bands = nb_mel_bands
        self.seq_len = seq_len
        self.edit_set = edit_set
        self.use_preservation_loss = use_preservation_loss
        self.alpha = alpha  # weight of preservation_loss
        self.use_cw = use_cw
        if self.use_cw:  # First difference: SEC don't consider "preservation loss"
            logging.info("Testing attack performance with C&W settings")
            self.use_preservation_loss = False

class Event:
    def __init__(self,
                 target_label,  # target label
                 attack_type="mirage",  # attack type ("mirage" or "mute)
                 start_time=None,  # ptb start time
                 end_time=None):  # ptb end time
        assert target_label is not None
        assert attack_type == "mirage" or attack_type == "mute"
        self.target_label = target_label
        self.attack_type = attack_type
        self.start_time = start_time
        self.end_time = end_time
        self.start_idx = None
        self.end_idx = None
        self.start_frame = None
        self.end_frame = None

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
        self.posterior_thresh = args.posterior_thresh
        self.output_dir = args.output_dir
        self.save_output = args.save_output
        self.attack_iters = args.attack_iters
        self.delta_duration = args.delta_duration
        self.tau = args.tau
        self.model_path = args.model_path
        self.config_path = args.config_path
        self.seq_len = args.seq_len

        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Load the trained ATST-SED model with fixed parameters
        self.inferencer = ATSTSEDInferencer(
            self.model_path,
            self.config_path,
            overlap_dur=3)
        self.inferencer.to(self.device)  # Set model to training mode to enable backward pass
        self.inferencer.model.train()  # Set model to training mode to enable backward pass

        # Define spectrogram transformer (must match training)
        self.mel_spectrogram = self.inferencer.feature_extractor.sed_feat_extractor

        # Audio parameters
        self.sample_rate = args.sample_rate
        self.n_fft = args.n_fft
        self.hop_len = args.hop_len
        self.nb_mel_bands = args.nb_mel_bands
        self.seq_len = args.seq_len

        self.edit_set = args.edit_set
        self.use_preservation_loss = args.use_preservation_loss
        self.alpha = args.alpha
        self.delta_length = int(self.delta_duration * self.sample_rate)
        self.use_cw = args.use_cw

    def load_audio(self):
        """
        Load and preprocess the input audio.
        """
        try:
            y, sr = torchaudio.load(self.wav_input_path)  # y: [channels, samples]
            y = y.to(torch.float32)
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
                y = resampler(y)
            if y.shape[0] > 1:
                y = torch.mean(y, dim=0, keepdim=True)  # Convert to mono

            # Normalize to [-1, 1]
            y = y / torch.max(torch.abs(y))
            return y.squeeze(0)  # Ensure the tensor is 1D to match delta's shape
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
        end_idx = (end_idx // self.hop_len) * self.hop_len
        if end_idx > total_samples:
            end_idx = (total_samples // self.hop_len) * self.hop_len
            start_idx = end_idx - self.delta_length
            if start_idx < 0:
                start_idx = 0
        return start_idx, end_idx

    def init_delta(self, total_samples):
        """
        initialize delta for attack

        :param total_samples: total sample of the optimized audio:
        :return: list of delta
        """
        if not self.edit_set:
            logging.error("Event list is empty.")
            return []
        # calculate (start_idx, end_idx)
        for event in self.edit_set:
            if event.start_time is not None and event.end_time is not None:
                start_idx = int(event.start_time * self.sample_rate)
                end_idx = int(event.end_time * self.sample_rate)
                if start_idx < 0 or end_idx > total_samples:
                    logging.error("Specified start_time and/or end_time are out of bounds. Exiting attack.")
                    return
                event.start_idx = start_idx
                event.end_idx = end_idx
            else:
                start_idx, end_idx = self.select_random_segment(total_samples)
                event.start_idx = start_idx
                event.end_idx = end_idx
                event.start_time = event.start_idx / self.sample_rate
                event.end_time = event.end_idx / self.sample_rate
        # sort event_idx by start_idx
        self.edit_set.sort(key=lambda x: x.start_idx)
        # Second difference: full delta settings used. only used when attacking task like SEC
        if self.use_cw:
            delta = {
                "ptb": torch.zeros(total_samples, requires_grad=True, device=self.device),
                "start_idx": 0,
                "end_idx": total_samples,
            }
            # Initialize delta within [-tau, tau]
            delta["ptb"].data = delta["ptb"].data.uniform_(-self.tau, self.tau)
            return [delta]
        # generate delta_list for Eveditor
        event = self.edit_set[0]
        delta_list = [{
            "start_idx": event.start_idx,
            "end_idx": event.end_idx,
        }]
        for event in self.edit_set[1:]:
            # get last delta
            last_delta = delta_list[-1]
            # if time crosses
            if event.start_idx <= last_delta["end_idx"]:
                # merge two event's deltas into a single delta
                last_delta["end_idx"] = max(last_delta["end_idx"], event.end_idx)
            else:  # else, generate new delta
                delta_list.append({
                    "start_idx": event.start_idx,
                    "end_idx": event.end_idx,
                })
        # instantiate deltas
        for delta in delta_list:
            delta["ptb"] = torch.zeros(delta["end_idx"] - delta["start_idx"], requires_grad=True, device=self.device)
            delta["ptb"].data = delta["ptb"].data.uniform_(-self.tau, self.tau)
        return delta_list

    def attack(self):
        y, _, _, _ = read_audio(
            self.wav_input_path, False, False, None
        )
        if y is None:
            logging.error("Failed to load audio. Exiting attack.")
            return
        y = y.to(self.device)

        # Step 1: Initialize perturbation delta on device
        delta_list = self.init_delta(y.shape[0])
        optimizer = optim.Adam([delta["ptb"] for delta in delta_list], lr=1e-3)

        # Define loss function (BCE Loss for the targeted segment)
        criterion = nn.BCELoss()

        # Generate output of origin input
        y_org = y.clone().detach()
        output_org, logits_org = self.inferencer(y_org)  # [batch, time, num_classes]
        logits_org = torch.cat(logits_org).transpose(-1, -2)

        # Determine which time frames correspond to the perturbed segment
        for event in self.edit_set:
            start_frame = int(event.start_idx / self.hop_len)
            end_frame = int(event.end_idx / self.hop_len)
            num_frames = output_org.shape[1]  # can only set perturbed frame when knowing the shape of output
            start_frame = max(0, min(start_frame, num_frames - 1))
            end_frame = max(0, min(end_frame, num_frames - 1))
            event.start_frame = start_frame
            event.end_frame = end_frame

        # Create target labels based on attack_type and target_label
        target = logits_org.clone().detach()  # Ensure 'logits' is defined before this line

        # Step 2: Edit target output
        for event in self.edit_set:
            if event.attack_type is not None and event.target_label is not None:
                if event.attack_type not in ["mirage", "mute"]:
                    logging.error("Invalid attack_type. It must be either 'mirage' or 'mute'. Exiting attack.")
                    return
                if event.target_label not in self.class_labels:
                    logging.error(f"Invalid target_label '{event.target_label}'. Exiting attack.")
                    return

                target_class_idx = self.class_labels[event.target_label]

                if event.attack_type == "mirage":
                    # Set the target_label's output to 1 in the perturbed segment
                    target[:, event.start_frame:event.end_frame, target_class_idx] = 1.0
                    logging.info(
                        f"Attack type: 'mirage' - setting label '{event.target_label}' to 1 from {event.start_frame} to {event.end_frame} frame.")
                elif event.attack_type == "mute":
                    # Set the target_label's output to 0 in the perturbed segment
                    target[:, event.start_frame:event.end_frame, target_class_idx] = 0.0
                    logging.info(
                        f"Attack type: 'mute' - setting label '{event.target_label}' to 0 from {event.start_frame} to {event.end_frame} frame.")
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
                y_adv += delta_list[0]["ptb"]
            else:
                for delta in delta_list:
                    y_adv[delta["start_idx"]:delta["end_idx"]] += delta["ptb"]
            # Ensure audio stays in [-1, 1]
            y_adv = torch.clamp(y_adv, -1.0, 1.0)

            # Step 3.2: Pass through the model: [time] -> [batch, frame, num_classes]
            output, logits = self.inferencer(y_adv)
            logits = torch.cat(logits).transpose(-1, -2)

            # Step 3.3: Calculate optimization loss
            adv_loss = 0
            preservation_mask = torch.ones_like(logits, dtype=torch.float)
            for event in self.edit_set:  # edit_set has been sorted at init_delta
                target_class_idx = self.class_labels[event.target_label]
                adv_loss += criterion(logits[:, event.start_frame:event.end_frame, target_class_idx],
                                      target[:, event.start_frame:event.end_frame, target_class_idx])
                preservation_mask[:, event.start_frame:event.end_frame, target_class_idx] = 0
            # preservation loss consider global context: which may decease UER but also decease ASR
            preservation_loss = bce_preservation_loss(logits, target, preservation_mask)

            if self.use_preservation_loss:
                loss = adv_loss + preservation_loss * self.alpha
            else:
                loss = adv_loss

            # Step 3.4: Backpropagate and step
            loss.backward()
            optimizer.step()

            # Step 3.5: Clip delta to stay within [-tau, tau]
            for delta in delta_list:
                delta["ptb"].data = torch.clamp(delta["ptb"].data, -self.tau, self.tau)

            # Logging
            if iteration % 100 == 0 or iteration == 1:
                logging.info(f"Iteration {iteration}/{self.attack_iters}, adv loss: {adv_loss.item():.6f}" +
                             (f", preservation loss: {preservation_loss.item():.6f}" if self.use_preservation_loss else ""))

        attack_end_time = time.time()

        # Step 4: Apply the final perturbation
        with torch.no_grad():
            y_adv = y.clone().detach()
            if self.use_cw:  # Second difference: full delta settings used. only used when attacking task like SEC
                y_adv += delta_list[0]["ptb"]
            else:
                for delta in delta_list:
                    y_adv[delta["start_idx"]:delta["end_idx"]] += delta["ptb"]
            y_adv = torch.clamp(y_adv, -1.0, 1.0)

        # Step 5: Save the adversarial audio
        adv_audio = y_adv.cpu().numpy()

        adv_filename = f"adv_{os.path.splitext(os.path.basename(self.wav_input_path))[0]}"
        for event in self.edit_set:
            # Calculate the start and end time of the disturbance (in seconds)
            adv_filename += f"_{event.attack_type}_{event.target_label}_{event.start_time:.3f}_{event.end_time:.3f}"
        adv_filename += ".wav"
        if self.use_cw:
            adv_filename = "cw_" + adv_filename
        adv_path = os.path.join(self.output_dir, adv_filename)
        if self.save_output:
            sf.write(adv_path, adv_audio, self.sample_rate)
            logging.info(f"Adversarial audio saved to {adv_path}")

        # Step 6: Evaluate the attack
        original = y.clone().detach()
        return self.evaluate_attack(original, y_adv, attack_end_time - attack_start_time)

    def evaluate_attack(self, audio_org, audio_adv, attack_time):
        """
        Evaluate the success of the attack based on attack_type and target_label,
        and calculate the Signal-to-Noise Ratio (SNR) between original and adversarial audio.

        Args:
            audio_org (torch.Tensor): Original audio tensor.
            audio_adv (torch.Tensor): Adversarial audio tensor.
            attack_time: Attack time consumption.

        Returns:
            dict: A dictionary containing 'success' (bool) and 'SNR' (float).
        """
        self.inferencer.eval()
        with torch.no_grad():
            decision_org, logits_org = self.inferencer(audio_org)
            decision_adv, logits_adv = self.inferencer(audio_adv)  # [batch, time, num_classes]

        logits_org = torch.cat(logits_org).transpose(-1, -2)
        logits_adv = torch.cat(logits_adv).transpose(-1, -2)

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
        # logging.info(f"se:{se}, fe:{fe}, ue:{ue}, ne:{ne}")

        for event in self.edit_set:
            if event.attack_type is not None and event.target_label is not None:
                if event.target_label not in self.class_labels:
                    logging.error(f"Invalid target_label '{event.target_label}'. Evaluation skipped.")
                    return evaluation_result

                target_class_idx = self.class_labels[event.target_label]

                target_pred_org = pred_org[0, event.start_frame:event.end_frame, target_class_idx]
                target_pred_adv = pred_adv[0, event.start_frame:event.end_frame, target_class_idx]
                ue -= (target_pred_org != target_pred_adv).sum().item()
                ne -= (target_pred_org == target_pred_adv).sum().item()
                if event.attack_type == "mute":
                    se += (target_pred_adv == False).sum().item()
                    fe += (target_pred_adv == True).sum().item()
                elif event.attack_type == "mirage":
                    se += (target_pred_adv == True).sum().item()
                    fe += (target_pred_adv == False).sum().item()
            else:
                logging.warning("attack_type or target_label not specified. Evaluation skipped.")

        evaluation_result["EP"], evaluation_result["ASR"], evaluation_result["UER"] = calc_matrix(se, fe, ue, ne)
        logging.info(f"Edit Precision (EP): {evaluation_result['EP'] * 100:.2f}%")
        logging.info(f"Attack Success Rate (ASR): {evaluation_result['ASR'] * 100:.2f}%")
        logging.info(f"Unintended Edit Rate (UER): {evaluation_result['UER'] * 100:.2f}%")
        try:
            snr_value = calculate_snr(audio_org, audio_adv)
            evaluation_result["SNR"] = snr_value
            logging.info(f"Signal-to-Noise Ratio (SNR): {snr_value:.2f} dB")
        except ValueError as ve:
            logging.error(f"SNR Calculation Error: {ve}")
            evaluation_result["SNR"] = None
        return evaluation_result

def rand_edit_set(num, attack_type):
    edit_set = []
    for i in range(num):
        edit_set.append(Event(target_label=get_rand_label(),attack_type=attack_type))
    return edit_set

def main():
    eval_matrix = {
        "avg SNR": 0.0,  # Signal to Noise Ratio
        "avg EP": 0.0,  # Average Edit Precision
        "avg ASR": 0.0,
        "avg UER": 0.0,
        "avg Time": 0.0
    }

    attack_type = 'mute'
    use_cw = False
    use_preservation_loss = True
    edit_num = 5
    alpha = 50
    tau = 0.05
    delta_duration = 2.0
    input_wav_dir = '../DESED_dataset/strong_label_real (3373)/'
    wav_list = read_data(input_wav_dir, 50)
    log_prefix = 'main_arb_' + attack_type
    # log_prefix = f'num({event_num})_ablation_arb_' + attack_type
    if use_cw:
        log_prefix += '_cw'
    elif use_preservation_loss:
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
            output_dir='AEs/',
            save_output=False,
            attack_iters=1000,
            delta_duration=delta_duration,
            tau=tau,
            edit_set=rand_edit_set(edit_num, attack_type),
            # edit_set=[
            #     Event(target_label=get_rand_label(),
            #           # start_time=0.0,
            #           # end_time=2.0,
            #           attack_type=attack_type
            #     ),
            #     Event(target_label=get_rand_label(),
            #           # start_time=0.5,
            #           # end_time=2.5,
            #           attack_type=attack_type
            #     ),
            #     Event(target_label=get_rand_label(),
            #           # start_time=1.0,
            #           # end_time=3.0,
            #           attack_type=attack_type
            #     )
            # ],
            alpha=alpha,
            use_preservation_loss=use_preservation_loss,
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
