# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 02:21:39 2023

@author: Dell
"""
import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores , mean_scores):
    display.clear_output(wait = True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Traning...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ylim=0)
    plt.text(len(scores)-1 , scores[-1] , str(scores[-1]))
    plt.text(len(mean_scores)-1 , mean_scores[-1] , str(mean_scores[-1]))