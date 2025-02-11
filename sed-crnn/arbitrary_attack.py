import torch
import torchaudio
import random
import os
import soundfile as sf
from torch import nn, optim
from crnn import CRNNForest
import logging
import time
from attack import calc_matrix, read_data, calculate_snr, setup_logging, get_rand_label, bce_preservation_loss

random.seed(42)

class ArgsNamespace:
    def __init__(self,
                 wav_input,
                 model_path,
                 posterior_thresh=0.5,
                 save_output=False,
                 output_dir='AEs/',
                 attack_iters=300,
                 delta_duration=1.0,
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
                 edit_set=[],
                 alpha=10,
                 use_preservation_loss=True,
                 test_sec=False):
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
        self.edit_set = edit_set
        self.alpha = alpha
        self.use_preservation_loss = use_preservation_loss
        self.test_sec = test_sec
        if self.test_sec:  # First difference: preservation loss
            logging.info("Testing attack performance with C&W settings")
            self.use_preservation_loss = False
        if self.use_preservation_loss:
            logging.info(f"Using preservation loss with alpha: {self.alpha}")

class Event:
    def __init__(self,
                 target_label,
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

        # Load the trained CRNN model with fixed parameters
        self.crnn = CRNNForest(self.model_path, self.device, args, self.num_classes)
        self.crnn.train()

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

        self.edit_set = args.edit_set
        self.alpha = args.alpha
        self.use_preservation_loss = args.use_preservation_loss
        self.test_sec = args.test_sec
        self.delta_length = int(self.delta_duration * self.sample_rate)

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

            # cut a 25 seconds audio clip
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

    def init_delta(self, total_samples):
        """
        initialize deltas for attack

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
        if self.test_sec:  # Second difference: full delta settings used. only used when attacking task like SEC
            delta = {
                "ptb": torch.zeros(total_samples, requires_grad=True, device=self.device),
                "start_idx": 0,
                "end_idx": total_samples,
            }
            # Initialize delta within [-tau, tau]
            delta["ptb"].data = delta["ptb"].data.uniform_(-self.tau, self.tau)
            return [delta]
        event = self.edit_set[0]
        delta_list = [{
            "start_idx": event.start_idx,
            "end_idx": event.end_idx,
        }]
        for event in self.edit_set[1:]:
            # get last delta
            last_delta = delta_list[-1]
            # if cross
            if event.start_idx <= last_delta["end_idx"]:
                # merge event into single delta
                last_delta["end_idx"] = max(last_delta["end_idx"], event.end_idx)
            else:
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
        # Load audio
        y = self.load_audio()
        if y is None:
            logging.error("Failed to load audio. Exiting attack.")
            return
        y = y.to(self.device)

        # Step 1: Initialize perturbation delta
        delta_list = self.init_delta(y.shape[0])
        optimizer = optim.Adam([delta["ptb"] for delta in delta_list], lr=1e-3)

        # Define loss function (BCE Loss for the targeted segment)
        criterion = nn.BCELoss()

        y_org = y.clone().detach()
        spec_org = self.calc_spectrogram(y_org)  # [time] -> [1, n_mels, frame]
        spec_org = spec_org.unsqueeze(1)  # Add channel dimension [1, n_mels, frame] -> [1, 1, n_mels, frame]

        # Pass through the model
        output_org = self.crnn(spec_org)  # output: [batch, frame, num_classes]

        # Determine which time frames correspond to the perturbed segment
        for event in self.edit_set:
            start_frame = int(event.start_idx / self.hop_len)
            end_frame = int(event.end_idx / self.hop_len)
            num_frames = output_org.shape[1]
            start_frame = max(0, min(start_frame, num_frames))
            end_frame = max(0, min(end_frame, num_frames))
            event.start_frame = start_frame
            event.end_frame = end_frame

        # Create target labels based on attack_type and target_label
        target = output_org.clone().detach()  # Ensure 'output' is defined before this line

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
            if self.test_sec:  # Second difference: full delta settings used. only used when attacking task like SEC
                y_adv += delta_list[0]["ptb"]
            else:
                for delta in delta_list:
                    y_adv[delta["start_idx"]:delta["end_idx"]] += delta["ptb"]
            # Ensure audio stays in [-1, 1]
            y_adv = torch.clamp(y_adv, -1.0, 1.0)

            # Step 3.2: Pass through model: [time] -> [batch, frame, num_classes]
            spec_adv = self.calc_spectrogram(y_adv)  # [time] -> [1, n_mels, frame]
            spec_adv = spec_adv.unsqueeze(1)  # Add channel dimension -> [1, 1, n_mels, frame]
            output = self.crnn(spec_adv)  # [batch, channel, feature, frame] -> [batch, frame, num_classes]

            # Step 3.3: Calculate loss
            adv_loss = 0
            preservation_mask = torch.ones_like(output, dtype=torch.float)
            for event in self.edit_set:  # edit_set has been sorted at init_delta
                target_class_idx = self.class_labels[event.target_label]
                adv_loss += criterion(output[:, event.start_frame:event.end_frame, target_class_idx],
                                      target[:, event.start_frame:event.end_frame, target_class_idx])
                preservation_mask[:, event.start_frame:event.end_frame, target_class_idx] = 0
            # preservation loss consider global contexts, which may decease UER but also decease ASR
            preservation_loss = bce_preservation_loss(output, target, preservation_mask)

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
            if self.test_sec:  # Second difference: full delta settings used. only used when attacking task like SEC
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
        if self.test_sec:
            adv_filename = "cw_" + adv_filename
        adv_path = os.path.join(self.output_dir, adv_filename)
        if self.save_output:
            sf.write(adv_path, adv_audio, self.sample_rate)
            logging.info(f"Adversarial audio saved to {adv_path}")

        # Step 6: Evaluate the attack
        audio_org = y.clone().detach()
        return self.evaluate_attack(audio_org, y_adv, attack_end_time - attack_start_time)

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
        spec_org = self.calc_spectrogram(audio_org)  # [1, n_mels, time]
        spec_org = spec_org.unsqueeze(1)  # [1, 1, n_mels, time]
        spec_adv = self.calc_spectrogram(audio_adv)
        spec_adv = spec_adv.unsqueeze(1)

        self.crnn.eval()
        with torch.no_grad():
            logits_org = self.crnn(spec_org)  # [batch, time, num_classes]
            logits_adv = self.crnn(spec_adv)  # [batch, time, num_classes]

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
        logging.info(f"Attack time consumption: {evaluation_result['Time']:.2f}")

        try:
            snr_value = calculate_snr(audio_org, audio_adv)
            evaluation_result["SNR"] = snr_value
            logging.info(f"Signal-to-Noise Ratio (SNR): {snr_value:.2f} dB")
        except ValueError as ve:
            logging.error(f"SNR Calculation Error: {ve}")
            evaluation_result["SNR"] = 0.0

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
        "avg Time": 0.0,
    }

    attack_type = 'mute'
    use_cw = False
    use_preservation_loss = True
    edit_num = 8
    alpha = 10
    tau = 0.02
    delta_duration = 2.0
    input_wav_dir = '../TUT-sound-events-2017-development/audio/street'
    wav_list = read_data(input_wav_dir, 50)
    # wav_list = ["audio/a001.wav"]
    # log_prefix = f'num({edit_num})_ablation_arb_' + attack_type
    log_prefix = f'main_arb_' + attack_type
    if use_cw:
        log_prefix += '_cw'
        use_preservation_loss = False
    elif use_preservation_loss:
        log_prefix += f'_alpha({alpha})'
    log_prefix += f'_tau({tau})'
    setup_logging('logs', log_prefix)
    logging.info("Starting Attack on CRNN model for Sound Event Detection")
    logging.info(f"Total samples: {len(wav_list)}")

    for i, wav_input in enumerate(wav_list):
        logging.info(
            f"\n\n===============================================================================================================================\n"
            f"=====                                                      No.{i}                                                           =====\n"
            f"===============================================================================================================================\n\n")
        # Define attack parameters using ArgsNamespace
        args = ArgsNamespace(
            wav_input=wav_input,
            model_path=[
                'models/clean_mon_2024_12_18_20_59_30_fold_1_model.pth',
                'models/clean_mon_2024_12_18_20_59_30_fold_2_model.pth',
                'models/clean_mon_2024_12_18_20_59_30_fold_3_model.pth',
                'models/clean_mon_2024_12_18_20_59_30_fold_4_model.pth'
            ],
            posterior_thresh=0.5,
            save_output=False,
            output_dir='AEs/',
            attack_iters=1000,
            delta_duration=delta_duration,
            tau=tau,
            edit_set=rand_edit_set(edit_num, attack_type),
            # edit_set=[
            #     Event(target_label=get_rand_label(),
            #           # start_time=1.0,
            #           # end_time=3.0,
            #           attack_type=attack_type
            #           ),
            #     Event(target_label=get_rand_label(),
            #           # start_time=1.5,
            #           # end_time=3.5,
            #           attack_type=attack_type
            #           ),
            #     Event(target_label=get_rand_label(),
            #           # start_time=2.0,
            #           # end_time=4.0,
            #           attack_type=attack_type
            #           ),
            # ],
            alpha=alpha,
            use_preservation_loss=use_preservation_loss,
            test_sec=use_cw
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
