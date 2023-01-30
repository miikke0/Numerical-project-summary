# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 19:17:28 2022

@author: mjafs
"""

import numpy as np
import matplotlib.pyplot as plt
import numba
import timeit



"Good figure template example"
# niceFigure example with good reasonable size fonts and TeX fonts
def niceFigure(useLatex=True):
    from matplotlib import rcParams
    plt.rcParams.update({'font.size': 25})
    if useLatex is True:
        plt.rc('text', usetex=True)
        plt.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    rcParams['xtick.major.width'] = 1
    rcParams['ytick.major.width'] = 1
    return

"Spin position and energy graphics"
def plot_spins(iterArr, N, EnArr, spinAr, EnAn, finaliter):
    niceFigure(True)
    fig, (ax2, ax1) = plt.subplots((2),figsize =(8,7), sharex = True)
    ax1.plot(iterArr/N,EnArr, label = 'E')
    ax1.plot(iterArr/N, EnAn/N, color = 'r', label = '$ < E > _{an}$')
    ax1.set_xlim([0,finaliter/N])
    ax1.set_xlabel('Iterations/N')
    ax1.set_yticks([0, -0.5, -1.0])
    ax1.set_ylabel('Energy/N$\epsilon$')
    ax1.legend()

    ax2.set_title('Spin evolution with $k_BT$ = {}'.format(kbT))
    ax2.contourf(iterArr/N,range(N),spinAr)
    ax2.set_yticks([0, 15, 30, 45])
    ax2.set_ylabel('N Spins')
    # extent = ([1, 0.1, 0.8, 0.4])
    plt.savefig('metro_trial_kT={},p={}.png'.format(kbT,p), format = 'png', dpi = 300, bbox_inches = 'tight')


#%%

#return the analytical average energy given the initial conditions
def calcAnE(N, kbT):
    "returns a single value representing the average expected energy of the system"
    b = 1/kbT
    return -N*np.tanh(b)


"Next two defitions are for use with the non-numba-optimized routine"
# @jit(nopython = True)
def initial_params(N):
    "Initial funciton for creating our array of spin states accordin to the order parameter p"
    spin = np.ones(N)    #set up spin array (we'll vary these in a moment)
    E = 0; M = 0   #initial energy and magnetization
    for i in range(1,N):   #don't include the outer values
        if np.random.rand(1) < p:
            spin[i] = -1   #flip the spins to align with the order parameter
        
        E -= spin[i - 1]*spin[i]   #implicitely setting epsilon = 1
        M += spin[i]
        
    E = E - spin[N - 1]*spin[0]
    M = M + spin[0]
    return spin, E, M

# @jit(nopython = True)
def metropolisAlg(N, spin, kT, E, M):
    "The update algroithm which tries a spin flip and returns it if the energy difference is too high"
    
    #begin by picking a random spin and flipping it
    num = np.random.randint(0, N - 1)    #start by picking a random number between 0 and N-1 
    flip = 0 
    
    #we assume we have flipped the spin 
    dE = 2*spin[num]*(spin[num - 1] + spin[(num + 1)%N])  #calculate the energy differennce
    
    if dE < 0.0:      # if the energy difference is negative, flip the spin
        flip = 1
    else:                   #else, see if its been flipped with prabability less than p
        p = np.exp(-dE/kT)
        if np.random.rand(1) < p:
            flip = 1               #if so, keeep the flip
    if flip == 1:             #if we keep the flip,
        E += dE               #update E
        M -= 2*spin[num]      #and M
        spin[num] = -spin[num]   #actually flip the spin in the spin array
    return E,M, spin 


"Definition with algorithm for use with the numba-optimized routine"
@numba.jit(nopython = True)  #use numba
def iterate(kT, E, M, spin, itertot):
    "For iterating over the metropolis alg. for itertot # of times - designed to work with numba"
    En = np.zeros(itertot)
    # Mn = np.zeros(itertot)
    spin2 = np.zeros((N, itertot))
    for n in range(itertot):
        #begin by picking a random spin and flipping it
        num = np.random.randint(0, N - 1)    #start by picking a random number between 0 and N-1 
        flip = 0 
        
        #we assume we have flipped the spin 
        dE = 2*spin[num]*(spin[num - 1] + spin[(num + 1)%N])  #calculate the energy differennce
        
        if dE < 0.0:      # if the energy difference is negative, flip the spin
            flip = 1
        else:                   #else, see if its been flipped with prabability less than p
            p = np.exp(-dE/kT)
            if np.random.rand(1) < p:
                flip = 1               #if so, keeep the flip
        if flip == 1:             #if we keep the flip,
            E += dE               #update E
            M -= 2*spin[num]      #and M
            spin[num] = -spin[num]   #actually flip the spin in the spin array
        En[n] = E/N
        # Mn[n] = M/N

        spin2[:,n] = spin
    # print(En[itertot - 1])
        # Esam = En[-1]   #average after all iterations
        # Msam = Mn[-1]
    # return Esam, Msam
    return En, spin2

#%%

"QUESTION 1 (a)"



"Optimized with numba"

"Initial parameters"
N = 50 #length of sampling size
p = 0.2   #order parameter (defines how 'hot' the start is)
kbT = 0.6   #temperature of surrounding 'bath'
itertot = 400*N
iterr = np.linspace(0,itertot, itertot)
#create line at the average energy of the system
Ean = calcAnE(N, kbT)
vals = np.ones(itertot)
Ean1 = Ean*vals
#set some empty arrays that will store our variables later 
En = np.zeros(itertot); spin2 = np.zeros((N, itertot), float)

#fetch the initial spin set-up
spin_init, Einit, Minit = initial_params(N)    


#time the main iteration loop which loops over itertot # of iterations
start = timeit.default_timer()
Earr, Spinarr = iterate(kbT, Einit, Minit, spin_init, itertot)
stop = timeit.default_timer()
print("time to solve with numba is:", stop - start)



"Plot the graphics for spin evolution"

#create line at the average energy of the system
Ean = calcAnE(N, kbT)
vals = np.ones(itertot)
Ean1 = Ean*vals

#plot the spin and energy evolution
plot_spins(iterr, N, Earr, Spinarr, Ean1, itertot)


#%%
"QUESTION 1 (b): Compute the monte carlo averages of the energy and the magenetization"


"Initial parameters"
N = 50 #length of sampling size
p = 0.8   #order parameter (defines how 'hot' the start is)
# kbT = 0.1   #temperature of surrounding 'bath'
kbT_range = np.linspace(0.1, 6, 200)
itertot = 100*N    #total number of iterations
N_mc = 100*N      #number of monte carlo averages
# iterr = np.linspace(0,itertot, itertot)
#create line at the average energy of the system
# Ean = calcAnE(N, kbT)
# vals = np.ones(itertot)
# Ean1 = Ean*vals
#set some empty arrays that will store our variables later 
# En = np.zeros(itertot); spin2 = np.zeros((N, itertot), float)

#fetch the initial conditions
spin, E, M = initial_params(N)    

@numba.jit(nopython  = True)
def metropolisiter1(totaliter, spin, E, M, kBT):
    En = np.zeros(totaliter)
    Mn = np.zeros(totaliter)
    Et = 0
    Mt = 0
    # spin2 = np.zeros((N, itertot))
    for n in range(totaliter):
        #begin by picking a random spin and flipping it
        num = np.random.randint(0, N)    #start by picking a random number between 0 and N-1 
        flip = 0 
        
        #we assume we have flipped the spin 
        dE = 2*spin[num]*(spin[num - 1] + spin[(num + 1)%N])  #calculate the energy differennce
        
        if dE < 0.0:      # if the energy difference is negative, flip the spin
            flip = 1
            
        else:                   #else, see if its been flipped with prabability less than p
            p = np.exp(-dE/kBT)
            if np.random.rand(1) < p:
                flip = 1               #if so, keeep the flip
        if flip == 1:             #if we keep the flip,
            E += dE               #update E
            M -= 2*spin[num]      #and M
            spin[num] = -spin[num]   #actually flip the spin in the spin array
        En[n] = E/N    #Energy array across all iterations
        Mn[n] = M/N    #magnetization at each iteration
        Et += En[n]
        Mt += Mn[n]
    Et = Et/totaliter
    Mt = Mt/totaliter

    return Et, Mt


@numba.jit(nopython=True)  #use numba
def monte_carloiter(kBT, E, M, spin, Nmc, totaliter):
    Eiter = np.zeros(Nmc)  
    Miter = np.zeros(Nmc)
    Eiter[:], Miter[:] = metropolisiter1(totaliter, spin, E, M, kBT)
    return Eiter, Miter



"Compute the average over a number of 'bath' temperatures kBT"
Eavg = np.zeros(len(kbT_range))
Mavg = np.zeros(len(kbT_range))
start = timeit.default_timer()
for i, T in enumerate(kbT_range):
    spin, E, M = initial_params(N)    
    # spin_init, Einit, Minit = initial_params(N)
    Eavgarr, Mavgarr = monte_carloiter(T, E, M, spin, N_mc, itertot)
    # Eavgarr, Mavgarr = monte_carloiter(T, Einit, Minit, spin_init, N_mc, itertot)
    Eavgarr1 = np.sum(Eavgarr)/N_mc
    Mavgarr1 = np.sum(Mavgarr)/N_mc
    # Eavg.append(Eavgarr1)
    Eavg[i] = Eavgarr1
    Mavg[i] = Mavgarr1
stop = timeit.default_timer()
print("The time to average is:", stop - start)

#%%
"Plot the Energy and Magnetization vs. the Temp. of the surrounding bath"
Esum_an = calcAnE(50,kbT_range)

niceFigure(True)
plt.plot(kbT_range, Eavg, '.', label = '$E$')
plt.plot(kbT_range, Esum_an/N, label = '$E_{an}$')
plt.xlabel('kT/$\epsilon$')
plt.ylabel('$< E >/N \epsilon$')
plt.show()

plt.plot(kbT_range, Mavg, '.')
plt.xlabel('kT/$\epsilon$')
plt.ylabel('$< M >/N$')
plt.show()




