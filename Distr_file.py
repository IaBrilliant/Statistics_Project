#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 20:06:50 2020

@author: brilliant
"""

# Finding chi-distr


import matplotlib.pyplot as plt
import numpy as np 
import STOM_higgs_tools as sht
from statistics import mean 
import pickle 


def findA(la, bin_edg, bin_heig): 
    count = 0
    dArea = 0
    for b in bin_heig:
        if count < 10 or count > 13:
            dArea += b*(bin_edg[count+1] - bin_edg[count]) #total area under histrogram
            count += 1    
        else: count+=1
    return dArea*la/(la**2*(np.exp(-104.0/la) - np.exp(-121.0/la) + np.exp(-127.8/la) - np.exp(-155.0/la))) #You find A by equating areas under exp. and histogram 
 
def chi_distribution(N): #where N - number of signal samples
    Chi_s = np.zeros(10000)
    for i in range(0, 10000):
        s = sht.generate_data(N)
        heights, edges, patches_ = plt.hist(s, range = [104, 155], bins = 30)
        Lambd = mean(s[N:])
        A_c = findA(Lambd, edges, heights)
        Chi_s[i] = (sht.get_B_chi(s, [104, 155], 30, A_c, Lambd))*28
    f = open('Chi_Set_background' + str(N), 'wb')
    pickle.dump(Chi_s, f)
    f.close()
    return print('P0 is saved')

'''
chi_distribution(0)
'''

f = open('Chi_Set_background0', 'rb')
Chi_53_set = pickle.load(f)
f.close()
chi_53_heights, chi_53_edges, patches_ = plt.hist(Chi_53_set, range = [10, 90], bins = 100, density = 1) #We can clearly see our value to be next to the peak
plt.show()
print(mean(Chi_53_set))