import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)

N_b = 10e5 # Number of background events, used in generation and in fit.
b_tau = 30. # Spoiler.


def generate_signal(N, mu, sig):
    ''' 
    Generate N values according to a gaussian distribution.
    '''
    return np.random.normal(loc = mu, scale = sig, size = N).tolist()


def generate_background(N, tau):
    ''' 
    Generate N values according to an exp distribution.
    '''
    return np.random.exponential(scale = tau, size = int(N)).tolist()

def generate_data(n_signals = 400):
    ''' 
    Generate a set of values for signal and background. Input arguement sets 
    the number of signal events, and can be varied (default to higgs-like at 
    announcement). 
    
    The background amplitude is fixed to 9e5 events, and is modelled as an exponential, 
    hard coded width. The signal is modelled as a gaussian on top (again, hard 
    coded width and mu).
    '''
    vals = []
    vals += generate_signal( n_signals, 125., 1.5)
    vals += generate_background( N_b, b_tau)
    return vals


def get_B_chi_trunc(vals, mass_range, nbins, A, lamb): #mass range - list, nbins - list 

    bin_heights_1, bin_edges_1 = np.histogram(vals, range = [mass_range[0], mass_range[1]], bins = nbins[0])
    half_bin_width_1 = 0.5*(bin_edges_1[1] - bin_edges_1[0])
    ys_expected_1 = get_B_expectation(bin_edges_1 + half_bin_width_1, A, lamb)
    
    bin_heights_2, bin_edges_2 = np.histogram(vals, range = [mass_range[2], mass_range[3]], bins = nbins[1])
    half_bin_width_2 = 0.5*(bin_edges_2[1] - bin_edges_2[0])
    ys_expected_2 = get_B_expectation(bin_edges_2 + half_bin_width_2, A, lamb)
    chi = 0

    for i in range( len(bin_heights_1) ):
        chi_nominator = (bin_heights_1[i] - ys_expected_1[i])**2
        chi_denominator = ys_expected_1[i]
        chi += chi_nominator / chi_denominator

    for i in range( len(bin_heights_2) ):
        chi_nominator = (bin_heights_2[i] - ys_expected_2[i])**2
        chi_denominator = ys_expected_2[i]
        chi += chi_nominator / chi_denominator
    
    return chi/float(nbins[0]+nbins[1]-2) 

def get_B_chi(vals, mass_range, nbins, A, lamb):
    ''' 
    Calculates the chi-square value of the no-signal hypothesis (i.e background
    only) for the passed values. Need an expectation - use the analyic form, 
    using the hard coded scale of the exp. That depends on the binning, so pass 
    in as argument. The mass range must also be set - otherwise, its ignored.
    '''
    bin_heights_1, bin_edges_1 = np.histogram(vals, range = mass_range, bins = nbins)
    half_bin_width_1 = 0.5*(bin_edges_1[1] - bin_edges_1[0])
    ys_expected_1 = get_B_expectation(bin_edges_1 + half_bin_width_1, A, lamb)
    chi = 0 
    
    for i in range( len(bin_heights_1) ):
        chi_nominator = (bin_heights_1[i] - ys_expected_1[i])**2
        chi_denominator = ys_expected_1[i]
        chi += chi_nominator / chi_denominator

    return chi/float(nbins-2) # B has 2 parameters.


def get_B_expectation(xs, A, lamb):
    ''' 
    Return a set of expectation values for the background distribution for the 
    passed in x values. 
    '''
    return [A*np.exp(-x/lamb) for x in xs]


def signal_gaus(x, mu, sig, signal_amp):
    return signal_amp/(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)


def get_SB_expectation(xs, A, lamb, mu, sig, signal_amp):
    ys = []
    for x in xs:
        ys.append(A*np.exp(-x/lamb) + signal_gaus(x, mu, sig, signal_amp))
    return ys


