# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 22:59:14 2022

@author: mjafs
"""

import numpy as np
import matplotlib.pyplot as plt

def niceFigure(useLatex=True):
    from matplotlib import rcParams
    plt.rcParams.update({'font.size':30})
    if useLatex is True:
        plt.rc('text', usetex=True)
        plt.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    rcParams['xtick.major.width'] = 1
    rcParams['ytick.major.width'] = 1
    return



#creating randomly distributed velocities: normally distributed with mu = 0, sigma = 1
vx = np.random.normal(0,1,10000)
vy = np.random.normal(0,1,10000)
vz = np.random.normal(0,1,10000)

#average 3D velocity
v = np.sqrt(vx**2 + vy**2 + vz**2)

#histogram of the simulated counts for average v
n, bins, patches = plt.hist(v, 100, density = True, edgecolor = 'black')

#now we define the theoretical shape of the distribution (maxwell boltzmann v distribution)
xvals = np.linspace(0,6)
def f(x):
    const = (4*np.pi)/((2*np.pi)**(3/2))
    return const*x**2*np.e**(-x**2/2)

#another set of randomly generated velocities (same paramaters as above)
vx2 = np.random.normal(0,1,10000)
vy2 = np.random.normal(0,1,10000)
vz2 = np.random.normal(0,1,10000)

#average speed of the randomly sampled relative velocity distribution (vrel = |v1 - v2|)
vrel = np.sqrt((vx - vx2)**2 + (vy - vy2)**2 + (vz - vz2)**2)

#histogram of the relative velocity counts
n, bins, patches = plt.hist(vrel, 100, density = True, edgecolor = 'black', alpha = 0.6)

#theoretical relative velocity function
def frel(x):
    const = (4*np.pi)/((2*2*np.pi)**(3/2))
    return const*x**2*np.e**(-x**2/4)

#plotting both theoretical curves
niceFigure(True)
plt.plot(xvals,f(xvals), color = 'red', label = '$\\vec{v}$ distribution')
plt.plot(xvals,frel(xvals), label = '$\\vec{v_{rel}}$ distribution')
plt.xlabel('Velocity (arb. units)')
plt.ylabel('Counts')
plt.legend()
plt.savefig('relvelex.png', format = 'png', dpi = 200, bbox_inches = 'tight')
plt.show()