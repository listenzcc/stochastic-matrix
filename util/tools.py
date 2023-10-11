"""
File: tools.py
Author: Chuncheng Zhang
Date: 2023-10-11
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Amazing things

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2023-10-11 ------------------------
# Requirements and constants
import time


# %% ---- 2023-10-11 ------------------------
# Function and class

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds")
        return result
    return wrapper


# %% ---- 2023-10-11 ------------------------
# Play ground


# %% ---- 2023-10-11 ------------------------
# Pending


# %% ---- 2023-10-11 ------------------------
# Pending
