
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import pandas as pd
from matplotlib.lines import Line2D


def plotRangeSeries(start, end, vF, classFrame, classes, colormap, alpha):
    plt.figure(figsize=(45,15))
    valuesFrame = vF.iloc[ :,start:end]
    valuesFrame['Class'] = classFrame
    groups = valuesFrame.groupby('Class')
    
    for name, group in groups:
        if name in classes:
            for i in range(0, group.shape[0]):
                plt.plot(group.iloc[i][:-1], label=name, color = colormap[name], alpha = alpha)
    custom_lines = []
    for c in classes:
        custom_lines.append(Line2D([0], [0], color=colormap[c], lw=4, alpha = alpha))
    plt.legend(custom_lines, classes)

