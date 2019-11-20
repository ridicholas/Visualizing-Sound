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

class SoundPCA():
    def __init__(self, soundFrame, typ = 'wave'):
        pca = PCA(n_components = 10)

        #convert soundFrame to table of series values
        if typ == 'wave':
            self.valuesFrame = makeValuesFrame(soundFrame)
        if typ == 'scaleMFCC':
            self.valuesFrame = makeScaledMFCCFrame(makeMFCCFrame(soundFrame))
        if typ == 'chroma':
            self.valuesFrame = makeChromaFrame(soundFrame)
        if typ == 'mfcc':
            self.valuesFrame = makeMFCCFrame(soundFrame)
        if typ == 'rms':
            self.valuesFrame = makeRMSFrame(soundFrame)
        if typ == 'centroid':
            self.valuesFrame = makeCentroidFrame(soundFrame)
        self.classFrame = soundFrame['Class']
        pcaMod = pca.fit_transform(self.valuesFrame)
        self.pcScores = pd.DataFrame(data = pcaMod,
                                columns = ['pc1', 'pc2', 'pc3', 'pc4', 'pc5', 'pc6', 'pc7', 'pc8', 'pc9', 'pc10'],
                                index = soundFrame.index)
        self.pcScores['Class'] = self.classFrame
        self.pcLoadings = pd.DataFrame(data= pca.components_, index = ['pc1', 'pc2', 'pc3', 'pc4', 'pc5', 'pc6', 'pc7', 'pc8', 'pc9', 'pc10'])
        self.projection1 = self.projection(1)
        self.projection2 = self.projection(2)
        self.projection3 = self.projection(3)
        self.scree = np.cumsum(pca.explained_variance_ratio_)
        
    def plotScoreByScore(self, score1, score2, legend = False, title = False):
        groups = self.pcScores.groupby('Class')
        #fig, ax = plt.subplots(figsize = (10,8))
        #ax.margins(0.05)
        for name, group in groups:
            plt.plot(group[f'pc{score1}'], group[f'pc{score2}'],
                    marker='o', linestyle='', ms=2, label=name, color = colormap[name])
        plt.xlabel(f'pc{score1}')
        plt.ylabel(f'pc{score2}')
        if title:
            plt.title(f'PC{score1} x PC{score2}')
        if legend:
            plt.legend()
        
        
    
    def projection(self, score):
        projectionFrame = pd.DataFrame(data = (self.pcLoadings.iloc[score-1] * np.dot(self.valuesFrame.iloc[0],
                                    self.pcLoadings.iloc[score-1])/np.dot(self.pcLoadings.iloc[score-1],
                                                                     self.pcLoadings.iloc[score-1]))).transpose()
        for i in range(1,self.valuesFrame.shape[0]):
            projectionFrame = projectionFrame.append(pd.Series(self.pcLoadings.iloc[score-1] * np.dot(self.valuesFrame.iloc[i],
                                    self.pcLoadings.iloc[score-1])/np.dot(self.pcLoadings.iloc[score-1],
                                                                     self.pcLoadings.iloc[score-1])), ignore_index = True)
            
        projectionFrame.index = self.valuesFrame.index
        return projectionFrame
    
    def projectionPlot(self, score, classes, colormap, alpha, start = 0, end = 0, legend = False):
        if end == 0:
            end = self.valuesFrame.shape[1]
        if score > 3:
            projectionFrame = self.projection(score)
        if score == 1:
            projectionFrame = self.projection1
        if score == 2:
            projectionFrame = self.projection2
        if score == 3:
            projectionFrame = self.projection3
                
        projectionFrame['Class'] = self.classFrame
        groups = projectionFrame.groupby('Class')
        #fig, ax = plt.subplots(figsize = (45,15))
        #ax.margins(0.05)
        for name, group in groups:
            if name in classes:
                for i in range(0, group.shape[0]):
                    plt.plot(group.iloc[i][start:end], label=name, color = colormap[name], alpha = alpha)
        custom_lines = []
        for c in classes:
            custom_lines.append(Line2D([0], [0], color=colormap[c], lw=4, alpha = alpha))
        if legend:
            plt.legend(custom_lines, classes)
            
    def marronScorePlot(self):
        plt.figure(figsize = (30,20))
        plt.margin = 0.5
        plt.subplot(3,3,1)

        plotFrame = pd.DataFrame({'pc1': self.pcScores.pc1,
                                  'pc2': self.pcScores.pc2,
                                  'pc3': self.pcScores.pc3, 
                                  'y': np.repeat(0.5, len(self.pcScores.pc1))})
        seaborn.distplot(self.pcScores.pc1, label = sounds['Class'])
        plt.scatter(plotFrame.pc1, plotFrame.y, c = np.vectorize(colormap.get)(self.classFrame), s = 2)
        plt.subplot(3,3,4)
        self.plotScoreByScore(1,2)
        plt.subplot(3,3,2)
        self.plotScoreByScore(2,1)
        plt.subplot(3,3,5)
        seaborn.distplot(soundPca.pcScores.pc2, label = sounds['Class'])
        plt.scatter(plotFrame.pc2, plotFrame.y, c = np.vectorize(colormap.get)(self.classFrame), s = 2)
        plt.subplot(3,3,3)
        self.plotScoreByScore(3,1)
        plt.subplot(3,3,6)
        self.plotScoreByScore(3,2)
        plt.subplot(3,3,7)
        self.plotScoreByScore(1,3)
        plt.subplot(3,3,8)
        self.plotScoreByScore(2,3)
        plt.subplot(3,3,9)
        seaborn.distplot(soundPca.pcScores.pc3, label = sounds['Class'])
        plt.scatter(plotFrame.pc3, plotFrame.y, c = np.vectorize(colormap.get)(self.classFrame), s = 2)