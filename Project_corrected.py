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
import seaborn as sns
                                                                                                                                                      
params = {'figure.figsize': [5, 5]} 
plt.rcParams.update(params)

#%% Generates plot of raw data (including signal) w/ error bars
# Part 1 

f = open('Values', 'rb')
vals = pickle.load(f)#Loads the data from preset values rather than generating anything new
f.close()

bin_heights, bin_edges, patches = plt.hist(vals, range = [104, 155], bins = 30) #104-155 - specified range
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
plt.errorbar(bin_centers, bin_heights, yerr = np.sqrt(bin_heights), ls = '', color = 'red', lw = 1, capsize = 2.5)
#Statistical uncertainty is yerr = sqrt(N) where N is the height of the bin - Possion error (Computer Session 2)
plt.xlabel('$m_{\gamma\gamma} (GeV)$')
plt.ylabel('Number of Entries', fontstyle = 'italic')
plt.ylim(0, 2000)
plt.xlim(104, 155)

#%% Parameterising the exponential decay
# Part 2 

def delete(x):              #Defines a function to label values we want to remove as 'true'
    if x > 122 and x < 127:  
      return False          #This is due to the fact that the gaussian peak is approx. in this range
    else: 
      return True 
    

def findA(la, bin_edg, bin_heig): 
    count = 0
    dArea = 0
    for b in bin_heig:
        if count < 10 or count > 13:
            dArea += b*(bin_edg[count+1] - bin_edg[count]) #Total area under histrogram
            count += 1    
        else: count+=1
    return dArea*la/(la**2*(np.exp(-104.0/la) - np.exp(-121.0/la) + np.exp(-127.8/la) - np.exp(-155.0/la))) #You find A by equating areas under exp. and histogram 
 

vals_truncated = filter(delete, vals) #Filter is fast and efficient in deleting, so gives us a set with only exponential data
lamb = mean(vals_truncated) #Gives the lambda estimator
a = findA(lamb, bin_edges, bin_heights) #Finding normalisation factor
expectation = sht.get_B_expectation(bin_edges, a, lamb) 
plt.plot(bin_edges, expectation)
#plt.savefig('Entries_set.png', dpi = 250)
plt.show()

#%%
# Part 3
'''
Chi_sq_b = sht.get_B_chi_trunc(vals, [104, 121, 129.5, 155], [10, 15], a, lamb) 
'''

Chi_sq_b = sht.get_B_chi(vals[400:], [104,155], 30, a, lamb) 
print("Reduced Chi-Squared (background) value is", Chi_sq_b) #By reduced I mean divided by degrees of freedom
print("P-value is:", stats.chi2.sf(Chi_sq_b * 28, 28))

#%%
# Part 4 (a)

Chi_sq_s = sht.get_B_chi(vals, [104, 155], 30, a, lamb) #Same as part 3 but includes signal region
print("Reduced Chi-Squared (+signal) value is", Chi_sq_s) #By reduced I mean divided by degrees of freedom
print("P-value is:", stats.chi2.sf(Chi_sq_s * 28, 28))
#Typical significane level is 5% hence we reject H0

#%%
# Part 4 (b) and (c)

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
'''
chi_distribution(0)
'''
f = open('Chi_Set_background', 'rb')
Chi_set = pickle.load(f)
f.close()
chi_heights, chi_edges, patches_ = plt.hist(Chi_set, range = [0, 90], bins = 90, density = 1) #We can clearly see our value to be next to the peak
plt.show()
print(mean(Chi_set))

#From preliminary runs of the code, N for the required chi was estimated to be in the range (180 - 190)
Chi_sq_expected = stats.chi2.isf(0.05, 28) #returns chi-squared given p and N_dof
print('Expected value of Chi-squared for p = 0.05 is', Chi_sq_expected)
'''
chi_distribution(185)
'''
f = open('Chi_Set_background185', 'rb')
Chi_53_set = pickle.load(f)
f.close()
chi_53_heights, chi_53_edges, patches_ = plt.hist(Chi_53_set, range = [10, 90], bins = 100, density = 1) #We can clearly see our value to be next to the peak
plt.show()
print(mean(Chi_53_set))
print('P-value is approximately 0.05 when N =', 185)


amplitudes = np.zeros(50)
for i in range(0, 50):
    signal_distribution = sht.generate_signal(185, 125., 1.5)
    heights_check, edges_check, patches = plt.hist(signal_distribution, range = [104, 155], bins = 30)
    amplitudes[i] = heights_check[12]  

expected_amp = mean(amplitudes)
print('The expected value for amplitude: N =', int(round(expected_amp)))
#If an amplitude is a sacling factor for exponential as in signal_gaus, then:
print('Amplitude as a scaling factor:', round(expected_amp)*np.sqrt(2*sp.pi)) 


def delete_0(x):              
    if x == 0: 
      return False          
    else: 
      return True 

def find_prob(set_, heights, edges): #set should be normalised
    prob = 0
    s = filter(delete_0, set_)
    m = mean(s)
    for i in range(int(round(m)), 100):
        if heights[i] != 0:
            prob += heights[i]*(edges[i+1] - edges[i]) #total area under region where p < 0.05
    return prob

print('Probability of finding a hint is:', find_prob(Chi_53_set, chi_53_heights, chi_53_edges))

#%%
# Part 5 (a)

Chi_sig_back = sht.get_B_chi_H1(vals, [104, 155], 30, a, lamb, 125.0, 1.5, 700)
print('For the background + signal hypothesis chi-squared is =', Chi_sig_back)
print('Corresponding p-value is =', stats.chi2.sf(Chi_sig_back * 25, 25))
#The p-value is too big to reject the hypothesis

#%% Paramterising the gaussian (bump) in data
# Part 5 (b)
import scipy.optimize as opt
      
bin_less_exp = []
for i in range(0,30):
    bin_less_exp.append(bin_heights[i]-a*np.exp(-bin_centers[i]/lamb))
plt.plot(bin_centers,bin_less_exp,'o')

def gauss(x,mu,sigma,amp):
   y = amp*np.exp((x-mu)**2/-2*sigma**2)
   return y
p0 = [125,1.5,200]
length = sp.linspace(104,155,200)
gauss_curve,gauss_curve_err = opt.curve_fit(gauss,bin_centers,bin_less_exp,p0)
plt.plot(length,gauss(length,gauss_curve[0],gauss_curve[1],gauss_curve[2]))
plt.show()
print(gauss_curve[0],gauss_curve[1],gauss_curve[2])

#%%
# Part 5(c)
mass = range(104,155)
Chi_sig_back_mass = []
for i in mass:
    Chi_sig_back_mass.append(sht.get_B_chi_H1(vals, [104, 155], 30, a, lamb, i, 1.5, 700))
plt.plot(mass,Chi_sig_back_mass)
plt.show()

#%%

f = open('Chi_Set_background', 'rb')
x1= pickle.load(f)
f.close()

f = open('Chi_Set_background185', 'rb')
x2 = pickle.load(f)
f.close()

params = {'figure.figsize': [6, 5]} 
plt.rcParams.update(params)

sns.distplot(x1, color="royalblue", label = "Signal amplitude = 0")
sns.distplot(x2, color="orangered", label = "Signal amplitude = 185")
plt.ylabel('Probability density function', fontstyle = 'italic')
plt.xlabel('$\chi^2$ value', fontstyle = 'italic', fontsize = 11)
plt.xlim(0, 95)
plt.ylim(0, 0.065)
plt.text(60, 0.035, r'$\mu(0)=29.0$')
plt.text(60, 0.031, r'$\mu(185)=41.1$')
plt.legend()
