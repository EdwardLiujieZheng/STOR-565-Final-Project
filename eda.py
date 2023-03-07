import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt

csv_2019 = "TennisData/wta_matches_qual_itf_2019.csv"
# csv_2020 = "TennisData/wta_matches_qual_itf_2020.csv"
# csv_2021 = "TennisData/wta_matches_qual_itf_2021.csv"
# csv_2022 = "TennisData/wta_matches_qual_itf_2022.csv"

df_2019 = pd.read_csv(csv_2019)
cols = list(df_2019.columns)

print(df_2019.head())
print(cols)