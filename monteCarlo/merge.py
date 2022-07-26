import pandas as pd
import glob
import os
import numpy as np
import re

# list of merged files returned
files = os.path.join("gridSearchResultFinal_from*.csv")
files = glob.glob(files)

# Sorting files by the first index number
file_ids = [int(re.search('from_(.*)_to*', file).group(1)) for file in files]
files = [x for _, x in sorted(zip(file_ids, files))]

# joining files with concat and read_csv
df = pd.concat(map(pd.read_csv, files), ignore_index=True)
df.drop_duplicates(subset=["Unnamed: 0"], inplace=True)
df.set_index("Unnamed: 0", inplace=True)
df.index.names = ["index"]
matrix = df["hvac_average_power"].values

df.to_csv(f"gridSearchResultFinal_merged.csv")

# save as a np matrix
np.save("mergedGridSearchResultFinal.npy", matrix)
