import pandas as pd
import mlflow
import librosa
import numpy as np


def get_dataframe(df, min_count, max_count, max_filter=None):
    df_temp = df.copy()

    freq = df_temp.source_id.value_counts()
    if max_filter:
        freq = freq[(min_count <= freq) & (freq <= max_filter)]
    else:
        freq = freq[min_count <= freq]
    df_temp = df_temp[df.source_id.isin(freq.index)]

    df_temp = (
        df_temp.groupby("source_id")
        .apply(lambda x: x.nlargest(max_count, "duration"))
        .reset_index(drop=True)
        .drop(
            columns=[
                "duration",
                "hash_id",
                "annotator_emo",
                "golden_emo",
                "annotator_id",
                "speaker_text",
                "speaker_emo",
            ]
        )
    )

    df_temp.dropna(inplace=True)

    print(df_temp.source_id.nunique())

    return df_temp


def get_model_and_params(experiment_name: str, model_name: str):
    experiment_id = dict(mlflow.get_experiment_by_name(experiment_name))['experiment_id']

    cls_102_5_df = mlflow.search_runs([experiment_id], order_by=['metrics.f1_weighted'])
    svc_df = cls_102_5_df[cls_102_5_df['tags.mlflow.runName'] == model_name]
    model_dict = dict(eval(svc_df['tags.mlflow.log-model.history'].item().replace('null', 'None'))[0])
    logged_model = '/'.join(['runs:', model_dict['run_id'], model_dict['artifact_path']])

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    params = svc_df[svc_df.columns[svc_df.columns.str.startswith('params.')]].dropna(axis=1).to_dict()
    params = {k.split('.')[-1]: list(v.values())[-1] for k, v in params.items()}
    for k, v in params.items():
        if v.isdigit():
            v = int(v)
        elif v.replace('.', '').isdigit():
            v = float(v)
        params[k] = v
    
    return loaded_model, params


def get_mfcc(wav_file_path):
    y, sr = librosa.load(wav_file_path, offset=0, duration=30)
    mfcc = np.array(librosa.feature.mfcc(y=y, sr=sr))
    return mfcc


def get_melspectrogram(wav_file_path):
    y, sr = librosa.load(wav_file_path, offset=0, duration=30)
    melspectrogram = np.array(librosa.feature.melspectrogram(y=y, sr=sr))
    return melspectrogram


def get_chroma_vector(wav_file_path):
    y, sr = librosa.load(wav_file_path)
    chroma = np.array(librosa.feature.chroma_stft(y=y, sr=sr))
    return chroma


def get_tonnetz(wav_file_path):
    y, sr = librosa.load(wav_file_path)
    tonnetz = np.array(librosa.feature.tonnetz(y=y, sr=sr))
    return tonnetz


def get_feature(file_path):
    mfcc = get_mfcc(file_path)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_min = mfcc.min(axis=1)
    mfcc_max = mfcc.max(axis=1)
    mfcc_feature = np.concatenate( (mfcc_mean, mfcc_min, mfcc_max) )

    melspectrogram = get_melspectrogram(file_path)
    melspectrogram_mean = melspectrogram.mean(axis=1)
    melspectrogram_min = melspectrogram.min(axis=1)
    melspectrogram_max = melspectrogram.max(axis=1)
    melspectrogram_feature = np.concatenate( (melspectrogram_mean, melspectrogram_min, melspectrogram_max) )

    chroma = get_chroma_vector(file_path)
    chroma_mean = chroma.mean(axis=1)
    chroma_min = chroma.min(axis=1)
    chroma_max = chroma.max(axis=1)
    chroma_feature = np.concatenate( (chroma_mean, chroma_min, chroma_max) )

    tntz = get_tonnetz(file_path)
    tntz_mean = tntz.mean(axis=1)
    tntz_min = tntz.min(axis=1)
    tntz_max = tntz.max(axis=1)
    tntz_feature = np.concatenate( (tntz_mean, tntz_min, tntz_max) ) 

    feature = np.concatenate((chroma_feature, melspectrogram_feature, mfcc_feature, tntz_feature) )
    return feature
