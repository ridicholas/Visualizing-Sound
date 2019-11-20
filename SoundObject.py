import glob
import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import IPython.display as ipd
import pandas as pd
import time
from matplotlib.lines import Line2D
from sklearn import preprocessing
from sklearn.decomposition import PCA
import seaborn

class SoundObject():
    def __init__(self, path, metaFrame, startsecond, endsecond):
        name = path.rsplit('/',1)[1]
        self.path = path
        self.id = name.replace('.wav', '')
        self.seriesValues, self.sr = librosa.load(path)
        self.seriesValues = self.seriesValues[int(startsecond*self.sr) : int(endsecond*self.sr)]
        self.timeLength = startsecond - endsecond
        self.className = metaFrame.loc[name,'class']
        self.classID = metaFrame.loc[name, 'classID']
        self.arrayLength = self.seriesValues.size
        self.salience = metaFrame.loc[name, 'salience']
        self.chroma = librosa.feature.chroma_stft(self.seriesValues).flatten()
        self.mfcc = librosa.feature.mfcc(self.seriesValues).flatten()
        self.rms = librosa.feature.rms(self.seriesValues).transpose().flatten()
        self.centroid = librosa.feature.spectral_centroid(self.seriesValues).transpose().flatten()
        
        


    def wavePlot(self):
        plt.figure()
        librosa.display.waveplot(self.seriesValues, self.sr)
        plt.title('{} - Class: {}'.format(self.id, self.className))
        plt.show()


    def specGram(self):
        plt.figure()
        specgram(self.seriesValues, Fs = self.sr)
        plt.title('{} - Class: {}'.format(self.id, self.className))
        plt.show()
        
    def play(self):
        ipd.Audio(self.path)
        
        
def makePathList(path):
    #create a list of all filepaths in directory (including subdirectories)
    filepaths = []
    for subdir, dirs, files in os.walk(path):
        for file in files:

            if file != '.DS_Store':
                filepath = subdir + os.sep + file
                filepaths.append(filepath)
    return filepaths


        
def makeSoundFrame(sr, length, pathlist, salience, meta, startsecond, endsecond):
    # create data frame of soundObjects of desired sampling rate, length, and salience
    frame = {}
    t0 = time.time()

    for file in pathlist:
        file = file.replace('//', '/')
        name = file.rsplit('/',1)[1]
        if meta.loc[name,'end'] - meta.loc[name, 'start'] == length and meta.loc[name,'salience'] == salience:
            sound = SoundObject(file, meta, startsecond, endsecond)
            if sound.sr == sr:
                frame[name]=sound
                print('appending...' + file)
    frame = pd.DataFrame.from_dict(frame, orient='index', columns=['Sound'])
    frame['Class'] = frame.apply(lambda row: row.Sound.className, axis = 1)
    frame.to_pickle('soundsFrame.pkl')

    print('that took {} seconds'.format(time.time()-t0))

    return frame

def makeValuesFrame(soundFrame):
    valuesFrame = pd.DataFrame(data = soundFrame['Sound'][0].seriesValues)
    valuesFrame = valuesFrame.transpose()
    for i in range(1, len(soundFrame['Sound'])):
        valuesFrame = valuesFrame.append(pd.Series(soundFrame['Sound'][i].seriesValues),
                                                       ignore_index = True)
    valuesFrame.index = soundFrame.index
    
    return valuesFrame

def makeChromaFrame(soundFrame):
    valuesFrame = pd.DataFrame(data = soundFrame['Sound'][0].chroma)
    valuesFrame = valuesFrame.transpose()
    for i in range(1, len(soundFrame['Sound'])):
        valuesFrame = valuesFrame.append(pd.Series(soundFrame['Sound'][i].chroma),
                                                       ignore_index = True)
    valuesFrame.index = soundFrame.index
    
    return valuesFrame

def makeMFCCFrame(soundFrame):
    valuesFrame = pd.DataFrame(data = soundFrame['Sound'][0].mfcc)
    valuesFrame = valuesFrame.transpose()
    for i in range(1, len(soundFrame['Sound'])):
        valuesFrame = valuesFrame.append(pd.Series(soundFrame['Sound'][i].mfcc),
                                                       ignore_index = True)
    valuesFrame.index = soundFrame.index
    
    return valuesFrame

def makeScaledMFCCFrame(mfccFrame):
    scaledMFCCFrame = pd.DataFrame(preprocessing.scale(mfccFrame), index = mfccFrame.index)
    return scaledMFCCFrame

def makeRMSFrame(soundFrame):
    valuesFrame = pd.DataFrame(data = soundFrame['Sound'][0].rms)
    valuesFrame = valuesFrame.transpose()
    for i in range(1, len(soundFrame['Sound'])):
        valuesFrame = valuesFrame.append(pd.Series(soundFrame['Sound'][i].rms),
                                                       ignore_index = True)
    valuesFrame.index = soundFrame.index
    
    return valuesFrame

def makeCentroidFrame(soundFrame):
    valuesFrame = pd.DataFrame(data = soundFrame['Sound'][0].centroid)
    valuesFrame = valuesFrame.transpose()
    for i in range(1, len(soundFrame['Sound'])):
        valuesFrame = valuesFrame.append(pd.Series(soundFrame['Sound'][i].centroid),
                                                       ignore_index = True)
    valuesFrame.index = soundFrame.index
    
    return valuesFrame



