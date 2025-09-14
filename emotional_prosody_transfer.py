import pyworld as pw
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

def load_and_preprocess_audio(file_path, target_sr=22050):

    y, sr = librosa.load(file_path, sr=target_sr, mono=True)
    
    y = y.astype(np.float64)
    
    return y, sr

def analyze_speech(y, sr, frame_period=5.0):
    """
    Analyzes the speech signal to extract WORLD vocoder parameters.

    Args:
        y (np.ndarray): The input waveform.
        sr (int): The sampling rate.
        frame_period (float): The frame period in milliseconds.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The fundamental frequency (F0) contour.
            - np.ndarray: The spectral envelope (SP).
            - np.ndarray: The aperiodicity (AP).
    """
    # F0 estimation using DIO algorithm
    f0, t = pw.dio(y, sr, frame_period=frame_period)

    # Refine F0 estimation using Stonemask algorithm
    f0 = pw.stonemask(y, f0, t, sr)

    # Spectral envelope estimation using CheapTrick algorithm
    sp = pw.cheaptrick(y, f0, t, sr)

    # Aperiodicity estimation using D4C algorithm
    ap = pw.d4c(y, f0, t, sr)

    return f0, sp, ap


def transfer_prosody(f0_source, sp_target, ap_target):
    """
    Prepares the parameters for synthesis by aligning their lengths.

    Args:
        f0_source (np.ndarray): The F0 contour from the source (emotional) speech.
        sp_target (np.ndarray): The spectral envelope from the target (neutral) speech.
        ap_target (np.ndarray): The aperiodicity from the target (neutral) speech.

    Returns:
        tuple: A tuple of aligned (f0, sp, ap) ready for synthesis.
    """
    # Get the number of frames from the spectral envelope
    num_frames_target = sp_target.shape[0]
    
    # Align the F0 contour length to the spectral envelope length
    f0_aligned = np.resize(f0_source, num_frames_target)


    return f0_aligned, sp_target, ap_target


def synthesize_speech(f0, sp, ap, sr, frame_period=5.0):
    """
    Synthesizes a waveform from WORLD vocoder parameters.

    Args:
        f0 (np.ndarray): The fundamental frequency contour.
        sp (np.ndarray): The spectral envelope.
        ap (np.ndarray): The aperiodicity.
        sr (int): The sampling rate.
        frame_period (float): The frame period in milliseconds.

    Returns:
        np.ndarray: The synthesized waveform.
    """
    # Synthesize the waveform using the WORLD synthesizer
    y_synthesized = pw.synthesize(f0, sp, ap, sr, frame_period)
    return y_synthesized


def plot_f0_contours(f0_emo, f0_neu, f0_conv, sr, frame_period, save_path):
    """
    Plots the F0 contours for comparison.
    
    Args:
        f0_emo (np.ndarray): F0 of the emotional source.
        f0_neu (np.ndarray): F0 of the neutral target.
        f0_conv (np.ndarray): F0 of the converted output.
        sr (int): Sampling rate.
        frame_period (float): Frame period in ms.
        save_path (str): Path to save the plot image.
    """
    plt.figure(figsize=(12, 6))
    
    # Generate time axes for each F0 contour
    time_axis_emo = np.arange(len(f0_emo)) * frame_period / 1000.0
    time_axis_neu = np.arange(len(f0_neu)) * frame_period / 1000.0
    time_axis_conv = np.arange(len(f0_conv)) * frame_period / 1000.0
    
    # Plot F0 contours, masking out unvoiced (zero) parts
    plt.plot(time_axis_neu, np.ma.masked_where(f0_neu == 0, f0_neu), label='Original Neutral F0', color='blue', linewidth=2)
    plt.plot(time_axis_emo, np.ma.masked_where(f0_emo == 0, f0_emo), label='Original Emotional F0', color='red', linestyle='--', linewidth=2)
    plt.plot(time_axis_conv, np.ma.masked_where(f0_conv == 0, f0_conv), label='Converted F0', color='green', linestyle=':', linewidth=3)
    
    plt.title('Fundamental Frequency ($F_0$) Contour Comparison')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"F0 plot saved to {save_path}")

import argparse

def main(emotional_file, neutral_file, output_file, plot_file):
    """
    Main function to run the emotional prosody transfer process.
    """
    # --- 1. Load and Preprocess Audio ---
    print("Loading and preprocessing audio files...")
    y_emo, sr_emo = load_and_preprocess_audio(emotional_file)
    y_neu, sr_neu = load_and_preprocess_audio(neutral_file)

    # Ensure sampling rates are the same
    if sr_emo!= sr_neu:
        raise ValueError("Sampling rates of the two files must be the same.")
    sr = sr_emo
    frame_period = 5.0

    # --- 2. Analyze Speech Signals ---
    print("Analyzing speech signals using WORLD vocoder...")
    f0_emo, sp_emo, ap_emo = analyze_speech(y_emo, sr, frame_period)
    f0_neu, sp_neu, ap_neu = analyze_speech(y_neu, sr, frame_period)

    # --- 3. Transfer Prosody ---
    print("Transferring prosody...")
    f0_conv, sp_conv, ap_conv = transfer_prosody(f0_emo, sp_neu, ap_neu)

    # --- 4. Synthesize New Speech ---
    print("Synthesizing converted speech...")
    y_conv = synthesize_speech(f0_conv, sp_conv, ap_conv, sr, frame_period)

    # --- 5. Save Output ---
    print(f"Saving converted audio to {output_file}...")
    sf.write(output_file, y_conv, sr)

    # --- 6. Visualize for Validation ---
    if plot_file:
        print("Generating F0 contour plot for validation...")
        plot_f0_contours(f0_emo, f0_neu, f0_conv, sr, frame_period, plot_file)

    print("Process completed successfully.")


def main(emotional_file, neutral_file, output_file, plot_file):
    """
    Main function to run the emotional prosody transfer process.
    """
    # --- 1. Load and Preprocess Audio ---
    print("Loading and preprocessing audio files...")
    y_emo, sr_emo = load_and_preprocess_audio(emotional_file)
    y_neu, sr_neu = load_and_preprocess_audio(neutral_file)

    # Ensure sampling rates are the same
    if sr_emo!= sr_neu:
        raise ValueError("Sampling rates of the two files must be the same.")
    sr = sr_emo
    frame_period = 5.0

    # --- 2. Analyze Speech Signals ---
    print("Analyzing speech signals using WORLD vocoder...")
    f0_emo, sp_emo, ap_emo = analyze_speech(y_emo, sr, frame_period)
    f0_neu, sp_neu, ap_neu = analyze_speech(y_neu, sr, frame_period)

    # --- 3. Transfer Prosody ---
    print("Transferring prosody...")
    f0_conv, sp_conv, ap_conv = transfer_prosody(f0_emo, sp_neu, ap_neu)

    # --- 4. Synthesize New Speech ---
    print("Synthesizing converted speech...")
    y_conv = synthesize_speech(f0_conv, sp_conv, ap_conv, sr, frame_period)

    # --- 5. Save Output ---
    print(f"Saving converted audio to {output_file}...")
    sf.write(output_file, y_conv, sr)

    # --- 6. Visualize for Validation ---
    if plot_file:
        print("Generating F0 contour plot for validation...")
        plot_f0_contours(f0_emo, f0_neu, f0_conv, sr, frame_period, plot_file)

    print("Process completed successfully.")
    
    
    
    

if __name__ == '__main__':
    emotional_file = "emotion.wav"
    neutral_file = "neutral.wav"
    output_file = "converted.wav"
    plot_file = "f0_comparison.png"

    main(emotional_file, neutral_file, output_file, plot_file)
    
    


