import code
import glob
import sys
import os
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import queue
import tensorflow.keras as keras
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import np_utils
from moviepy.editor import VideoFileClip, concatenate_videoclips
import youtube_dl
import time
import datetime
from duration import to_seconds, to_timedelta

SAVED_MODEL_PATH = 'nn_model.h5'
SAVED_LABEL_PATH = 'model_classes.npy'

class _extractor:

    model = None
    labels = None
    _instance = None

    def upload(self):
        vid_type = int(input('Enter 1 or 2\n1 - YouTube video, 2 - Local video\n'))
        if vid_type == 1:
            #URL = input('Enter Youtube Link:')
            #path = yt_video(URL)
            print('Sorry! This option will be enabled in upcoming updates.')
            self.upload()
        elif vid_type == 2:
            path = input('Enter video path:\n')    
        else:
            print('You entered an incorrect number!')
            self.upload()
            #sys.exit()
        return path

    def extract_feature(self,file_name=None, sr = None):
        #if file_name: 
        #    print('Extracting', file_name)
        #X, sample_rate = sf.read(file_name, dtype='float32')
        X, sample_rate = file_name, sr

        if X.ndim > 1: X = X[:,0]
        X = X.T

        # short term fourier transform
        stft = np.abs(librosa.stft(X))

        # mfcc (mel-frequency cepstrum)
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)

        # chroma
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)

        # melspectrogram
        mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)

        # spectral contrast
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)

        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
        return mfccs,chroma,mel,contrast,tonnetz

    def classifier(self, path):
        cl_time = time.time()
        #videofile = self.upload()
        videofile = path
        video = VideoFileClip(videofile)
        audio = video.audio 
        audio.write_audiofile('aud.wav')
        audiofile = 'aud.wav' 
        x, sr = sf.read(audiofile, dtype='float32')
        max_slice = 12
        window_length = max_slice * sr
        itrLen = int(librosa.get_duration(filename = audiofile)/max_slice)
        df = pd.DataFrame(columns=['start_time','end_time','category'])
        #model = 'nn_ad_trained - 2.h5'
        #model = keras.models.load_model(model)
        ft_time = time.time()

        for i in range(itrLen):
            features = np.empty((0,193))
            a = x[i*window_length:(i+1)*window_length]
            startTime = i * max_slice
            endTime = (i+1) * max_slice
            mfccs, chroma, mel, contrast,tonnetz = self.extract_feature(file_name = a, sr = sr)
            ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
            features = np.array(np.vstack([features,ext_features]))
            #features = np.expand_dims(features, axis=2)
            pred = np.argmax(self.model.predict(features), axis = -1)
            lb = LabelEncoder()
            lb.classes_ = np.load(self.labels)
            pred = (lb.inverse_transform((pred)))
            df.loc[i] = [startTime,endTime,pred]
            #print(startTime,endTime,pred)
        print('Feature extraction time: %d minutes' % int((time.time() - ft_time)/60))
        df.to_csv('cnn_ad_Result.csv', index = False)
        df = pd.read_csv('cnn_ad_Result.csv')
        df = df[df.category != "['non-highlight']"]
        df = df[df.category != "['advertisement']"]
        df = df.reset_index( drop = True)
        df.to_csv('cnn_ad_Highlights.csv', index = False)
        print('Classification done!')
        os.remove(audiofile)
        print('Time taken for classification: %d minutes' % int((time.time() - cl_time)/60))
        return df, videofile

    def generate_highlights(self, path):
        sg_time = time.time()
        df, videofile = self.classifier(path)

        temp = []
        i = 0
        j = 0
        n = len(df) - 2
        m = len(df) - 1
        while (i <= n):
            j = i + 1
            while (j <= m):
                if (df['end_time'][i] == df['start_time'][j]):
                    df.loc[i, 'end_time'] = df.loc[j, 'end_time']
                    temp.append(j)
                    j = j + 1
                else:
                    i = j
                    break
        df.drop(temp, axis=0, inplace=True)
        df = df.reset_index( drop = True)
        print(df)
        print('length of df ', len(df))
        input_video_path = videofile
        videoclip = VideoFileClip(input_video_path)
        video_duration = videoclip.duration
        print('video_duration in seconds - ',video_duration)
        start = np.array(df['start_time'])
        end = np.array(df['end_time'])
        for i in range(len(df)):
            if int(start[i]) != 0:
                start_lim = start[i] - 7
            else:
                start_lim = start[i]
            #if i+1 != len(df):
            #    end_lim = end[i] + 5
            if int(end[i]) != int(video_duration):
                end_lim = end[i] + 5
            #elif end[i] > (video_duration - 5):
            #    end_lim = end[i]
            else:
                end_lim = end[i]
            print('start', start_lim )
            print('end', end_lim )
            
            output_video_path = f'{str(i + 1)}.mp4'
            
            with VideoFileClip(input_video_path) as video:
                new = video.subclip(start_lim, end_lim)
                new.write_videofile(output_video_path, audio_codec='aac')
        print('Videos segmentation: %d minutes' % int((time.time() - sg_time)/60))  

        TARGET_VIDEO = "Highlights.mp4"
        gn_time = time.time()
        
        if len(df) == 1:
            os.rename('1.mp4', f'{TARGET_VIDEO}')
        else:
            clip = []
            clipname = []
            for i in range(len(df)):
                i += 1
                temp = f'{i}.mp4'
                if os.path.isfile(temp):
                    clipname.append(temp)
                    temp = VideoFileClip(f'{i}.mp4')
                    clip.append(temp)
            final_clip = concatenate_videoclips([clip[i] for i in range(len(df))])
            final_clip.write_videofile(TARGET_VIDEO, threads = 8, fps=24)
            for name in clipname:
                os.remove(name)
        print('Videos concatenation: %d minutes' % int((time.time() - gn_time)/60))
        print('Total time: %d minutes' % int((time.time() - sg_time)/60))
        return TARGET_VIDEO

def Extractor():
    

    
    if _extractor._instance is None:
        _extractor._instance = _extractor()
        _extractor.model = keras.models.load_model(SAVED_MODEL_PATH)
        _extractor.labels = SAVED_LABEL_PATH 
    return _extractor._instance

if __name__ == "__main__":
    pass

    # create instance
    #ext = Extractor()

    
    #path = ext.upload()
    #path = ext.upload()
    #filename = ext.generate_highlights(path)

