import librosa
import numpy as np

def extract_mfcc(file_path):
    # librosa.load returns a tuple, captured by 2 vars
    # y = numpy array of audio time series: amplitude
    # sr = sampling rate: hz
    y, sr = librosa.load(file_path)

    # mfcc = Mel-frequency cepstral coefficients
    mfcc = librosa.feature.mfcc(y = y, sr = sr, n_mfcc = 13)

    mfcc_mean = np.mean(mfcc.T, axis = 0)

    return mfcc_mean

audio_file = "AudioFiles/TOOL-ThePatient.wav"
mfcc_features = extract_mfcc(audio_file)

print(f"Extracted MFCCs: {mfcc_features}")
