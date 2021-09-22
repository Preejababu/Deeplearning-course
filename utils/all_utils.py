import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib # FOR SAVING MY MODEL AS A BINARY FILE
from matplotlib.colors import ListedColormap
import os


def prepare_data(df):
  X = df.drop("y", axis=1)

  y = df["y"]

  return X, y