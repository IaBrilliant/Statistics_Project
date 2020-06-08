#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 00:33:03 2020

@author: brilliant
"""

import matplotlib.pyplot as plt
import numpy as np 
import STOM_higgs_tools as sht
from statistics import mean 
from scipy import stats
import pickle 
import scipy as sp
                                                                                                                                                      
params = {'figure.figsize': [5, 5]} 
plt.rcParams.update(params)

# Part 1 

#f = open('Values', 'rb')
#vals = pickle.load(f)
#f.close()



vals = sht.generate_data(400)
bin_heights, bin_edges, patches = plt.hist(vals, range = [104, 155], bins = 30) #104-155 - specified range
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
plt.errorbar(bin_centers, bin_heights, yerr = np.sqrt(bin_heights), ls = '', color = 'red', lw = 1, capsize = 2.5)
#Statistical uncertainty is yerr = sqrt(N) where N is the height of the bin - Possion error (Computer Session 2)
plt.ylim(0, 2000)
plt.xlim(104, 155)

# Part 2 

def delete(x):              #Previous algorithm didn't work
    if x > 122 and x < 127: #You can play around with these values and you will realise that lambda is the highest with these params. 
      return False          #This is due to the fact that the gaussian peak is approx. in this range
    else: 
      return True 
    

def findA(lamb, bin_edges, bin_heights): 
    count = 0
    dArea = 0
    for b in bin_heights:
        dArea += b*(bin_edges[count+1] - bin_edges[count]) #total area under histrogram
        count += 1    
    return dArea*lamb/(lamb**2*(np.exp(-104.0/lamb) - np.exp(-155.0/lamb))) #You find A by equating areas under exp. and histogram 
 

vals_truncated = filter(delete, vals) # Filter is fast and efficient in deleting
lamb = mean(vals_truncated) #There is a prewritten func to calculate mean 
a = findA(lamb, bin_edges, bin_heights) #finding normalisation factor
expectation = sht.get_B_expectation(bin_edges, a, lamb) 
plt.plot(bin_edges, expectation)
plt.show()

# Part 3
'''
Chi_sq_b = sht.get_B_chi_trunc(vals, [104, 121, 129.5, 155], [10, 15], a, lamb) #New function for finding the value of chi_sq 
'''

Chi_sq_b = sht.get_B_chi(vals[400:], [104,155], 30, a, lamb) #Or we can just ignore gaussian elements
Chi_sq_s = sht.get_B_chi(vals, [104, 155], 30, a, lamb) #New function for finding the value of chi_sq 
print("Normalised Chi-Squared (backgroung) value is", Chi_sq_b) #By normalised I mean divided by degrees of freedom

# Part 4 (a)

print("Normalised Chi-Squared (+signal) value is", Chi_sq_s) #By normalised I mean divided by degrees of freedom
print("P-value is:", stats.chi2.sf(Chi_sq_s * 28, 28))
#Typical significane level is 5% hence we reject H0

# Part 4 (b)

#Saving the Chi_set separately
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


# Chi_Set_background for N = 0 was calculated separatly and is stored in the file
f = open('Chi_Set_background', 'rb')
Chi_set = pickle.load(f)
f.close()
chi_heights, chi_edges, patches_ = plt.hist(Chi_set, range = [0, 90], bins = 90) #We can clearly see our value to be next to the peak
plt.show()

# Part 4(c)

Chi_sq_expected = stats.chi2.isf(0.05, 28) #returns chi-squared given p and N_dof
print('Expected value of Chi-squared for p = 0.05 is', Chi_sq_expected) 

#Previously I developed a function that finds N so that it's chi-sq value is the same as the one expected
#However, the signal is generated randomly and its chi-value is chosen with accordance to gaussian distribution 
#Therefore the test is inconclusive


#From preliminary runs of the code, N for the required chi was estimated to be in the range (190 - 200)
'''
chi_distribution(200)
'''
f = open('Chi_Set_background200', 'rb')
Chi_53_set = pickle.load(f)
f.close()
chi_53_heights, chi_53_edges, patches_ = plt.hist(Chi_53_set, range = [10, 90], bins = 100) #We can clearly see our value to be next to the peak
plt.show()
print(mean(Chi_53_set[:400]))

#Let's take N = 200

print('P-value is approximately 0.05 when N =', 200)
signal_distribution = sht.generate_signal(200, 125., 1.5)
heights_check, edges_check, patches = plt.hist(signal_distribution, range = [104, 155], bins = 30)
plt.show()
print('From the plot it is clear that the signal has an amplitude of', heights_check[12]) 
#If an amplitude is a sacling factor for exponential as in signal_gaus, then:
print('Amplitude as a scaling factor:', heights_check[12]*np.sqrt(2*sp.pi)) 

'''
def delete_0(x):              
    if x == 0: 
      return False          
    else: 
      return True 

def find_prob(set_, heights, edges): #set should be normalised
    prob = 0
    s = filter(delete_0, set_)
    m = mean(s)
    for i in range(round(m), 91):
        if heigts[i] != 0:
            prob += heights[i]*(edges[i+1] - edges[i]) #total area under region where p < 0.05
    return prob

print('Probability of finding a hint is:', find_prob(xxx, heigthsxxx, edgesxxx))
'''

# Part 5 (a)

Chi_sig_back = sht.get_B_chi_H1(vals, [104, 155], 30, a, lamb, 125.0, 1.5, 700)
print('For the background + signal hypothesis chi-squared is =', Chi_sig_back)
print('Corresponding p-value is =', stats.chi2.sf(Chi_sig_back * 25, 25))
#The p-value is too big to reject the hypothesis

# Part 5 (b)

# Technically because it is not asked which techniques we are supposed to use specifically, we can use
# optimisator as a background parameterisation tech. 

''' 
Three possible ways of doing it:
- Taking the gaussian set directly from vals[:400] and using optimiser on it
- Taking the residuals: residuals_hist = bin_heights - resudials at certain points
- Creating a 2D matrix as in 2(d) to scan through Amp and sigma
'''

# Part 5(c)
''' 
You assume that the peak is at some m[i]
For this particular case you calculate chi-sq
Then by doing it over some range, you will find that Chi-sq is minimum at 125 GeV
'''





