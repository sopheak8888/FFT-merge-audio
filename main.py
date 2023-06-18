import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt


def merge_audio(file1, file2, output_file):
    # Load audio files
    data1, samplerate1 = sf.read(file1)
    data2, samplerate2 = sf.read(file2)

    # Resample the audio files to a common sample rate if necessary
    if samplerate1 != samplerate2:
        # Choose the higher sample rate
        target_samplerate = max(samplerate1, samplerate2)
        data1 = sf.resample(data1, target_samplerate)
        data2 = sf.resample(data2, target_samplerate)
    else:
        target_samplerate = samplerate1

    # Ensure the first audio file is longer or equal in length to the second audio file
    if len(data1) < len(data2):
        data1, data2 = data2, data1

    # Pad the second audio file with zeros to match the length of the first audio file
    data2_padded = np.pad(data2, (0, len(data1) - len(data2)))

    # Perform FFT on the audio data
    fft_data1 = np.fft.fft(data1)
    fft_data2 = np.fft.fft(data2_padded)

    # Merge the audio data in the frequency domain
    merged_fft = fft_data1 + fft_data2

    # Perform inverse FFT to get the merged audio data
    merged_data = np.fft.ifft(merged_fft).real

    # Normalize the merged audio data
    merged_data /= np.max(np.abs(merged_data))

    # Save the merged audio to a new file
    sf.write(output_file, merged_data, target_samplerate)

    # Plot the audio files
    plt.figure(figsize=(12, 8))

    # Plot file1
    plt.subplot(311)
    plt.plot(data1)
    plt.title('File 1')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')

    # Plot file2
    plt.subplot(312)
    plt.plot(data2)
    plt.title('File 2')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')

    # Plot merged audio
    plt.subplot(313)
    plt.plot(merged_data)
    plt.title('Merged Audio')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()


# Merge two audio files and save the merged audio to a new file
merge_audio('test1.wav', 'test2.wav', 'merged.wav')
