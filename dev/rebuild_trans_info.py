import os
import time
import pandas as pd


df_train = pd.read_csv("~/gender-pronoun/input/dataset/trans-gap-development.csv")
df_val = pd.read_csv("~/gender-pronoun/input/dataset/trans-gap-validation.csv")
df_test = pd.read_csv("~/gender-pronoun/input/dataset/trans-gap-test.csv")
sample_sub = pd.read_csv("~/gender-pronoun/input/dataset/sample_submission_stage_1.csv")

df_train = df_train[df_train['Text_bt'].str.contains(' << \[\[A\]\] >> & << \[\[B\]\] >> & << \[\[P\]\] >> ',regex=True)]

import ipdb; ipdb.set_trace();
