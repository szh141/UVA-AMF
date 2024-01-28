#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 21:00:19 2023

@author: chongmingwang
"""

import glob
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def get_stats(group):
    
    mean = []
    error = []
    
    file_list = glob.glob(f'{group}/**/*.csv')
    for file_name in file_list:
        df = pd.read_csv(file_name)
        stats = df.describe()

        #a = np.arange(4.5,25.5,3) #plot bin center
        b = stats.loc['mean']
        c = stats.loc['std']
        
        mean.append(b)
        error.append(c)
    
    return mean,error

mean_KI,error_KI = get_stats('KI-1')
mean_WT,error_WT = get_stats('WT-1')

a = np.arange(4.5,25.5,3)

"""
for i in range(len(mean_KI)):
    plt.errorbar(a,mean_KI[i], yerr=error_KI[i],fmt='o')
    plt.ylim(0,0.5)
    plt.xticks(np.arange(6,24,3))
"""

plt.figure(figsize=(24, 10))
for Yi, Ei, delta,i in zip(mean_KI, error_KI, np.linspace(-0.5, 0.5, len(mean_KI)),range(4)):
    plt.errorbar(a + delta, Yi, yerr=Ei, ls='none', marker='o',markersize=16,color=[1,0.8-0.2*i,0.7-0.2*i],label=f'KI-{i+1}')
    plt.legend()


for Yi, Ei, delta,i in zip(mean_WT, error_WT, np.linspace(-0.5, 0.5, len(mean_WT)),range(4)):
    plt.errorbar(a + delta, Yi, yerr=Ei, ls='none', marker='v',markersize=16,color=[0.25*i,0.25*i,0.25*i],label=f'WT-{i+1}')
    plt.legend()

plt.legend(ncol=2,fontsize=32)

plt.ylim(0, 0.55) 

"""
for Xi in X[:-1]:  # draw a vertical line between each group
    plt.axvline(Xi + 0.5, ls=':', lw=0.5, color='grey')
"""

plt.xticks(np.arange(6,24,3),fontsize=20)
plt.yticks(fontsize=20)

plt.xlabel('Diameter in micron',fontsize=32) 
plt.ylabel('Relative Frequency',fontsize=32) 

plt.savefig('combined.png')
