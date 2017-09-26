
import os
from glob import glob
import json

import pandas as pd
import matplotlib.pyplot as plt

key_solver = 'ns2d'

dicts = []

for path in glob('results_bench/*'):
    with open(path) as f:
        dicts.append(json.load(f))

df = pd.DataFrame(dicts)
df1 = df.loc[df['nb_proc'] == 1]
t_elapsed1 = df1['t_elapsed'].mean()
times = df['t_elapsed']

fig = plt.figure()
ax = plt.subplot()
ax.plot(df['nb_proc'], t_elapsed1/times, 'or')
ax.set_title('speed up')


fig = plt.figure()
ax = plt.subplot()
ax.plot(df['nb_proc'], t_elapsed1/times/df['nb_proc'], 'or')
ax.set_title('normalized speed up')


plt.show()
