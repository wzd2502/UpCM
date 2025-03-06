import numpy as np
import pandas as pd
import os, csv
from DataProcess import csv2generalgraph
from copy import deepcopy

parent_dir = os.getcwd()
# parent_dir = os.path.dirname(rootpath)

def splitDAGandwrite(filepath, networks, numofdatasets, methods, n_corcon, n_wrocon):

    # csv_file = filepath
    csv_file = 'D:\\code_other\\bayesys_system\\Bayesys_Release_v3.6\\Output\\AllDAGlearned.csv'
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)

        IsFirst = True
        repeat = 0

        # recored the sign-th method
        methodsign = 0
        repeat = 0
        numofdatasign = 0
        networksign = 0
        infos = pd.DataFrame(columns=['ID', 'Variable1', 'Dependency', 'Variable2'])
        for row in reader:
            if row[0] == 'ID':
                if not IsFirst:
                    method = methods[methodsign]
                    numofdata = numofdatasets[numofdatasign]
                    network = networks[networksign]
                    infonames = f'{method}con_{network}_N{numofdata}_C{n_corcon}_W{n_wrocon}_{repeat}.csv'
                    infos.to_csv(os.path.join(parent_dir, 'Networks', network, method, infonames), index=False)

                    methodsign += 1
                    if methodsign == len(methods):
                        repeat += 1
                        methodsign = 0
                        if repeat == 10:
                            numofdatasign += 1
                            repeat = 0
                            if numofdatasign == len(numofdatasets):
                                networksign += 1
                                numofdatasign = 0
                    infos.drop(infos.index, inplace=True)
                IsFirst = False
            else:
                newdata = {'ID': row[0], 'Variable1': row[1], 'Dependency': row[2], 'Variable2': row[3]}
                # infos = infos.append(newdata, ignore_index=True)
                infos.loc[len(infos)] = newdata

        # after read all, the last dag should be added.
        infonames = f'{methods[methodsign]}con_{network}_N{numofdata}_C{n_corcon}_W{n_wrocon}_{repeat}.csv'
        infos.to_csv(os.path.join(parent_dir, 'Networks', network,methods[methodsign], infonames), index=False)


networks = ['COVID19']
numofdatasets = [866]
methods = ['MAHC', 'GES', 'SaiyanH']
splitDAGandwrite(os.path.join(parent_dir, 'Networks', 'AllDAGlearned.csv'), networks, numofdatasets, methods,  0.2, 1)