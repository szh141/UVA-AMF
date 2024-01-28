#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 09:46:17 2024

@author: chongmingwang
"""
import pandas as pd
import os

def export(results):
    file_name = f'{sample}.csv'
    counts.to_csv(file_name, index=False)  

cwd = os.getcwd()
sample= os.path.basename(cwd)
    
GFP = pd.read_csv(f"{sample} GFP.csv")
DAPI = pd.read_csv(f"{sample} DAPI.csv")


counts = pd.merge(DAPI,GFP)
counts['GFP/DAPI'] = counts['GFP_count']/counts['DAPI_count']
export(counts)