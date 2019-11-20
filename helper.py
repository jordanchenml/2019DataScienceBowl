import numpy as np
import pandas as pd
import os
import json
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook

def read_data():
    print(f'Read data')
    train_df = pd.read_csv('../input/data-science-bowl-2019/train.csv')
    test_df = pd.read_csv('../input/data-science-bowl-2019/test.csv')
    train_labels_df = pd.read_csv('../input/data-science-bowl-2019/train_labels.csv')
    specs_df = pd.read_csv('../input/data-science-bowl-2019/specs.csv')
    sample_submission_df = pd.read_csv('../input/data-science-bowl-2019/sample_submission.csv')
    print(f"train shape: {train_df.shape}")
    print(f"test shape: {test_df.shape}")
    print(f"train labels shape: {train_labels_df.shape}")
    print(f"specs shape: {specs_df.shape}")
    print(f"sample submission shape: {sample_submission_df.shape}")
    return train_df, test_df, train_labels_df, specs_df, sample_submission_df

train_df, test_df, train_labels_df, specs_df, sample_submission_df = read_data()
sample_train_df = train_df.sample(100000)
sample_test_df = test_df.sample(10000)
sample_train_df.to_csv('sample_train.csv')
sample_test_df.to_csv('sample_test.csv')