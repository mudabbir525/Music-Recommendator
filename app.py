from flask import Flask, request, jsonify
import librosa
import numpy as np
from scipy import stats
import pandas as pd
import speech_recognition as sr
import pickle
import glob
from pathlib import Path
from pydub import AudioSegment
from pydub.silence import split_on_silence
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Load the lyric vector
dbfile = open('tag_vector', 'rb')
lyr = pickle.load(dbfile)
dbfile.close()
lyr = lyr['top']

# Load the MFCC features index table
index_file = open('query_vector', 'rb')
features_index = pickle.load(index_file)
index_file.close()

token = RegexpTokenizer('\s+|\-+|\.+|\@+|\t+|\n+|[0-9]+|\"+|\>+|\,+|\?+|\:+|\{+|\(+|\[+|\)+|\}+|\]+|\<+|\_+|\!+|\/+|\|+|\\+|\*+|\=+|\^+|\$+|\&+|\#+|\*+|\++|;+', gaps=True)
lem = WordNetLemmatizer()


def compute_mfcc(song_dir):
    name = 'mfcc'
    size = 20
    moments = ('mean', 'std', 'skew', 'kurtosis', 'median', 'min', 'max')

    columns = []
    for m in moments:
        col = ((name, m, '{:02d}'.format(i + 1)) for i in range(size))
        columns.extend(col)

    names = ('feature', 'statistics', 'number')
    columns = pd.MultiIndex.from_tuples(columns, names=names)

    features = pd.Series(index=columns, dtype=np.float32)

    y, sr = librosa.load(song_dir)
    stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    assert stft.shape[0] == 1 + 2048 // 2
    assert np.ceil(len(y) / 512) <= stft.shape[1] <= np.ceil(len(y) / 512) + 1

    mel = librosa.feature.melspectrogram(sr=sr, S=stft ** 2)
    f = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)

    features["mfcc", 'mean'] = np.mean(f, axis=1)
    features["mfcc", 'std'] = np.std(f, axis=1)
    features["mfcc", 'skew'] = stats.skew(f, axis=1)
    features["mfcc", 'kurtosis'] = stats.kurtosis(f, axis=1)
    features["mfcc", 'median'] = np.median(f, axis=1)
    features["mfcc", 'min'] = np.min(f, axis=1)
    features["mfcc", 'max'] = np.max(f, axis=1)
    return features


@app.route('/compute_mfcc', methods=['POST'])
def get_mfcc():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file.'}), 400

    try:
        audio_features = compute_mfcc(file)
        return jsonify(audio_features.to_dict()), 200
    except Exception as e:
        return jsonify({'error': 'An error occurred while computing MFCC features.', 'details': str(e)}), 500


@app.route('/process_lyrics', methods=['POST'])
def process_lyrics():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file.'}), 400

    try:
        lyric = ''
        with sr.AudioFile(file) as source:
            audio_data = sr.record(source)
            text = sr.recognize_google(audio_data)
            lyric += text.lower()

        lyric = token.tokenize(lyric)
        lyric = [lem.lemmatize(k) for k in lyric]

        arr = [0 for i in range(5000)]

        for i in lyric:
            for j in range(1, len(lyr)):
                if i == lyr[j]:
                    arr[i] += 1

        return jsonify({'lyrics_vector': arr}), 200

    except Exception as e:
        return jsonify({'error': 'An error occurred while processing lyrics.', 'details': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
