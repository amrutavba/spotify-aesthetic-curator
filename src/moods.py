"""Mood / aesthetic labels: transparent rules (first match wins) + K-Means as a data-driven alternative."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from .config import *


def label_moods(df):
    r = [(df.danceability >= .7) & (df.energy >= .6), (df.energy >= .7), (df.valence >= .6) & (df.energy >= .5),
         (df.valence < .35) & (df.energy < .6), (df.acousticness >= .6), (df.energy < .45)]
    names = ["Dance-oriented", "Energetic", "Upbeat", "Melancholic", "Acoustic", "Relaxed"]
    return pd.Series(np.select(r, names, default="Balanced"), index=df.index)


def cluster(df, k=N_CLUSTERS):
    """K-Means on standardised audio features. Returns labels and a profile table (mean raw feature per cluster)."""
    X = StandardScaler().fit_transform(df[AUDIO])
    lab = KMeans(k, n_init=10, random_state=SEED).fit_predict(X)
    return lab, df.assign(cluster=lab).groupby("cluster")[AUDIO].mean().round(2)
