import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

if __name__=='__main__':
    # read data
    snow_df = pickle.load( open('pkl/snow_df.p', 'rb'))

    ''' eda
    columns: ['DEPTH', 'GRTR_60', 'SNOW_24', 'SNOW_4DAY', 'SWE_24', 'DENSE_24',
       'SETTLE', 'TMIN', 'TMIN_DELTA', 'TMAX', 'TMAX_SUM', 'STATION']

    '''
    path = '../figs/eda/'
    # snow depth by station
    fig, ax = plt.subplots(figsize=(10,6))
    for name, group in snow_df.groupby('STATION'):
        group.plot(x=group.index, y='DEPTH', ax=ax, label=name)
    plt.title('Aspen area SNOTEL stations: snow depth')
    plt.xlabel('year')
    plt.ylabel('snow depth [m]')
    plt.savefig(path + 'aspen_snow_depth.png', dpi = 350)
    plt.close()


    # >60 cm depth by station
    fig, ax = plt.subplots(figsize=(10,6))
    for name, group in snow_df.groupby('STATION'):
        group.plot(x=group.index, y='GRTR_60', ax=ax, label=name)
    plt.title('Aspen area SNOTEL stations: snow depth >60 cm')
    plt.xlabel('year')
    plt.ylabel('snow depth [m]')
    plt.savefig(path + 'aspen_snow_depth_grt60.png', dpi = 350)
    plt.close()

    # 24 hr snowfall by station
    fig, ax = plt.subplots(figsize=(10,6))
    for name, group in snow_df.groupby('STATION'):
        group.plot(x=group.index, y='SWE_24', ax=ax, label=name, marker='.', alpha=0.3)
    plt.title(r'Aspen area SNOTEL stations: 24 hr $\Delta$ SWE')
    plt.xlabel('year')
    plt.ylabel(r'$\Delta$ SWE [m]')
    plt.show()
    #plt.savefig(path + 'aspen_change_swe.png', dpi = 350)
    plt.close()
