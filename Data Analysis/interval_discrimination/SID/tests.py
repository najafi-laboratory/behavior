# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 14:04:33 2025

@author: timst
"""
import os

SESSION_DATA_PATH = r'C://behavior//session_data//LCHR_TS01'


filename = 'test.mat'

filepath = os.path.join(SESSION_DATA_PATH, filename)

with open(filepath, "wb") as f:
    f.write(b"This is not a valid .mat file at all!")