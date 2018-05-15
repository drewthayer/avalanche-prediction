import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if __name__=='__main__':
    avy_df = pd.read_csv('../data/data-clean/avy_data.csv')

    ''' columns:
    ['datetime', 'Obs ID', 'Date', 'Date Known', 'Time', 'Time Known',
       'BC Zone', 'HW Zone', 'Operation', 'Landmark', 'First Name',
       'Last Name', 'HW Path', '#', 'Elev', 'Asp', 'Type', 'Trigger',
       'Trigger_sub', 'Rsize', 'Dsize', 'Incident', 'Area Description',
       'Comments', 'Avg Slope Angle', 'Start Zone Elev',
       'Start Zone Elev units', 'Sliding Sfc', 'Weak Layer', 'Weak Layer Type',
       'Avg Width', 'Max Width', 'Width units', 'Avg Vertical', 'Max Vertical',
       'Vertical units', 'Avg Crown Height', 'Max Crown Height',
       'Crown Height units', 'Terminus ', 'datetime.1'] '''

    # where in Aspen
    aspen_locs = avy_df[avy_df['BC Zone'] == 'Aspen']['Landmark']

    from collections import Counter
    c = Counter(aspen_locs)
    s = sorted(c.items(), key = lambda x: x[1], reverse=True)
    for pair in s:
        plt.bar(pair[0], pair[1])
    plt.xticks( rotation='vertical')
    #plt.xticks(range(len(labels)), labels, rotation='vertical')
    plt.tight_layout()
    plt.show()
