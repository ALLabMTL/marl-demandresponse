import pandas as pd
import glob
import os
import numpy as np

# list of merged files returned
files = os.path.join("./monteCarlo/", "gridSearchResultFinal*.csv")
files = glob.glob(files)

# joining files with concat and read_csv
df = pd.concat(map(pd.read_csv, files), ignore_index=True)
df.drop_duplicates(subset=["Unnamed: 0"], inplace=True)
df.set_index("Unnamed: 0", inplace=True)
df.index.names = ["index"]
matrix = df["hvac_average_power"].values

# save as a np matrix
np.save("mergedGridSearchResultFinal_from_0_to_3061800.npy", matrix)
