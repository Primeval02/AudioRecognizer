import os
import json
import librosa
import numpy as np
from scipy.spatial.distance import euclidean

def extract_mfcc(file_path):
    # librosa.load returns a tuple, captured by 2 vars
    # y = numpy array of audio time series: amplitude
    # sr = sampling rate: hz
    y, sr = librosa.load(file_path)

    # mfcc = Mel-frequency cepstral coefficients
    mfcc = librosa.feature.mfcc(y = y, sr = sr, n_mfcc = 13)

    mfcc_mean = np.mean(mfcc.T, axis = 0)

    print(f"Extracted MFCCs: {mfcc_mean}")
    return mfcc_mean

def find_song(query_mfcc, song_database):
    closest_song = None
    smallest_distance = float('inf')

    for song_title, mfcc_values in song_database.items():
        distance = euclidean(query_mfcc, mfcc_values)

        if distance < smallest_distance:
            closest_song = song_title
            smallest_distance = distance

    return closest_song, smallest_distance

def load_from_json(filename = 'song_database.json'):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {};

def save_to_json(database, filename = 'song_database.json'):
    with open(filename, 'w') as f:
        json.dump(database, f, indent = 4)

def main():
    song_database = load_from_json()

    audio_file_path = input("Enter the path to an audio file: ").strip()

    while True:
        user_input = input("Enter 'a' to add the song or 'c' to compare with the database: ").strip().lower()

        if user_input == 'a':
            song_title = input("Enter the song title: ")
            mfcc_features = extract_mfcc(audio_file_path)
            song_database[song_title] = mfcc_features.tolist()
            save_to_json(song_database)
            print(f"{song_title} added to the database.")


        if user_input == 'c':
            query_mfcc = extract_mfcc(audio_file_path)
            closest_song, distance = find_song(query_mfcc, song_database)
            if closest_song:
                print(f"Closest match: {closest_song} with distance {distance:.2f}")
            else:
                print("No match found.")

        elif user_input == "e":
            break

        else:
            print("Invalid input. Please enter 'a', 'c', or 'e'.")
if __name__ == "__main__":
    main()
