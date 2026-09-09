import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df=sns.load_dataset("titanic")
print(df.head())
df.columns
df.duplicated()