import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

def temporalAnalysis(TP_list, total_foreground, fscore):
    x = np.linspace(0,200,200)
    plt.figure(1)
    plt.plot(x, TP_list, 'b', label='True Positives')
    plt.plot(x, total_foreground, 'r', label='Total Foreground pixels')
    plt.xlabel("Frames")
    plt.ylabel("Number of pixels")
    plt.axis([0, 200, 0, max(total_foreground)])
    plt.show()

    plt.figure(2)
    plt.plot(x, fscore, 'g', label='F1-score')
    plt.xlabel("Frames")
    plt.ylabel("F1-score")
    plt.axis([0, 200, 0, max(fscore)])
    plt.show()
    
def temporalAnalysisDelay(TP_listlist, total_foreground_listlist, fscore_listlist):
    x = np.linspace(0,200,200)
    sizelist = len(TP_listlist)
    plt.figure(2)
    green_patch = mpatches.Patch(color='green', label='0 Frames Desynch')
    red_patch = mpatches.Patch(color='red', label='5 Frames Desynch')
    blue_patch = mpatches.Patch(color='blue', label='10 Frames Desynch')
    pink_patch = mpatches.Patch(color='black', label='20 Frames Desynch')
    yellow_patch = mpatches.Patch(color='yellow', label='30 Frames Desynch')
    color_array = ['g','r','b','k','y']
    plt.legend(handles=[green_patch,red_patch,blue_patch,pink_patch,yellow_patch],framealpha=0.5,loc='best')
    for i in range(0,sizelist):
        fscoreonly = fscore_listlist[i]
        plt.plot(x, fscoreonly, color_array[i], label='F1-score')
        plt.xlabel("Frames")
        plt.ylabel("F1-score")
        plt.axis([0, 200, 0, 1])
    plt.show()
    