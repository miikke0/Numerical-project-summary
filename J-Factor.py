# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 10:35:00 2022

@author: mjafs
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import dblquad,quad,tplquad,nquad,quadrature
from scipy.special import erf
import timeit

#%%

"Good figure template example"
# niceFigure example with good reasonable size fonts and TeX fonts
def niceFigure(useLatex=True):
    from matplotlib import rcParams
    plt.rcParams.update({'font.size': 20})
    if useLatex is True:
        plt.rc('text', usetex=True)
        plt.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    rcParams['xtick.major.width'] = 1
    rcParams['ytick.major.width'] = 1
    return


#%%

#Defining constants needed for integration
#Constants are based on values given in page 4 of ;/https://arxiv.org/pdf/1912.09486.pdf
rho0=0.4 #GeV cm^-3
R0=8.127*(3.0857*10**21)  #Converting all units in kpc to cm (Conversion factor is 3.0857*10**21 cm/kpc)
rs=20*(3.0857*10**21)
Rhalo=30*(3.0857*10**21)
xmax=Rhalo+R0             #This is based on eqn. 4 in above paper
gamma=1.2
rhos=rho0/((2**(3-gamma))/(((R0/rs)**gamma)*(1+(R0/rs))**(3-gamma)))  #scale radius equation (rearranged eqn. 5)


G_N = (((6.67e-8))) #Universal gravitational constant?? units of (cm^3 g^-1 s^-2)
#DM mass gravitational potential

#%%

"J-FACTOR CALCULATIONS IN DIFFERENT COORDINATES"

"Conversion between coordinates"

#Define funcion that lets the user convert freely between galactic and polar coords
#Essentially compute \vec{r} from any starting coordinate system
def coord_trans(sys,d,theta,phi,b,l):
    if sys == "polar":
        r = np.sqrt(R0**2-2*d*R0*np.cos(theta)+d**2)
    elif sys == "galactic":
        r = np.sqrt(d**2 + R0**2 - 2*R0*d*np.cos(b)*np.cos(l))
    return r
#don't use this right now... Need to figure out how to integrate this def in to 
#the integrand

#instead: for now, define each coordinate system 'r' seperately


"J-factor integrand in polar coordinates using dblquad"
def rhointpol(theta, h):
    r=np.sqrt(R0**2-2*h*R0*np.cos(theta)+h**2)      #distance from galactic center
    rho_x=((rhos*2**(3-gamma))/(((r/rs)**gamma)*(1+(r/rs))**(3-gamma)))**2 #we need the density squared to compute J
    full_int=np.sin(theta)*rho_x    #Multiplying by sintheta for solid angle
    return full_int


"Compute all-sky J-factor in polar"
start = timeit.default_timer()
dJ=dblquad(rhointpol,0,1e26, 0, np.pi)    #integrate over the line of sight and theta angle
allskyJ = dJ[0]*2*np.pi                 #integrand is not phi dependent so we multiply by 2pi here
print('All-sky j-factor is:',allskyJ,"GeV^-2 cm^-5 sr")
stop = timeit.default_timer()
print("Calculation time is: ", stop - start, "seconds")

#%%
"J-factor in polar coords using trapezoidal integration technique"


#write the integrand so that it contains all coordinates
def rhointpol1(h, theta, phi):
    r=np.sqrt(R0**2-2*h*R0*np.cos(theta)+h**2)      #distance from galactic center
    rho_x=((rhos*2**(3-gamma))/(((r/rs)**gamma)*(1+(r/rs))**(3-gamma)))**2 #we need the dencity squared
    full_int=np.sin(theta)*rho_x    #Multiplying by sintheta for solid angle
    return full_int

polarr = []   #empty array
theta = np.linspace(0,np.pi,100)   #theta and phi values 
phi = np.linspace(0,2*np.pi,100)

for i in theta:
    for j in phi:
        dJpol = quad(rhointpol1, 0,1e26, args = (i,j,)) #integrate over line of sight for every value of theta and phi
        polarr.append(dJpol[0])

polgrid = np.reshape(polarr,(len(theta),len(phi)))   #create 2D grid representing the theta and phi coordinates
theta_int = np.trapz(polgrid,x = theta, axis = 0)    #integrate over the columns of the grid (ie theta)
phi_int = np.trapz(theta_int,x = phi)            #integrate over the remaining array
print(phi_int)   #print the J-factor



#%%
"Calculate the J-factor in galactic coordinates"

#j-factor integrand in galactic coordinates
def rhointgal(d,b,l):
    """
    The integrand in the J-factor expression using galactic coordinates
    d: the line of sigh distance
    b: galactic longitude
    l: galactic latitude
    """
    r = np.sqrt(d**2 + R0**2 - 2*R0*d*np.cos(b)*np.cos(l))   #distance from glactic center  
    # r = np.sqrt(d**2*np.sin(b)**2 + d**2*np.cos(b)**2*np.sin(l)**2 + (d*np.cos(b)*np.cos(l) - R0)**2)
    rho_x=((rhos*2**(3-gamma))/(((r/rs)**gamma)*(1+(r/rs))**(3-gamma)))**2 #the density profile squared
    
    full_int=np.cos(b)*rho_x    #calculate solid angle in the sky
    # full_int = np.sin(b)*rho_x
    return full_int


#need to integrate using numpy.trapz to speed up integration over d,b, and l
galarr = []   #empty array
ell = np.linspace(-np.pi,np.pi,10)     #galactic latitudes and longitudes
b = np.linspace(-np.pi/2.1,np.pi/2.1,10)

for i in b:
    for j in ell:
        dJgal = quad(rhointgal, 0,1e26, args = (i,j,)) #integrate over line of sight for every value of theta and phi
        galarr.append(dJgal[0])

galgrid = np.reshape(galarr,(len(b),len(ell)))   #create 2D grid representing the theta and phi coordinates
b_int = np.trapz(galgrid,x = ell, axis = 0)    #integrate over the columns of the grid (ie theta)
ell_int = np.trapz(b_int,x = b)            #integrate over the remaining array
print(ell_int)   #print the J-factor


#%%

"Plot the intensity sky map. This is done in galactic coordinates"

#galactic latitudes and longtitudes
# b = np.linspace(-np.pi,np.pi,50)
# ell = np.linspace(-np.pi/1.1,np.pi/1.1,50)
# b = np.linspace(-np.pi/2.1,np.pi/2.1,50)
ell = np.linspace(-np.pi,np.pi,10)
b = np.linspace(-np.pi/2.1,np.pi/2.1,10)
cntrarr = []


for i in b:
    for j in ell:
        dJgal_cntr = quad(rhointgal,0, 1e26, args = (i,j,))
        cntrarr.append(dJgal_cntr[0])     #create 1D array for line of sight integral at various b and l

log_cntrarr = np.log10(cntrarr)
galcntr_grid = np.reshape(log_cntrarr,(len(b),len(ell)))   #convert 1D array to grid  

#plot the contour map
niceFigure(True)
plt.contourf(ell, b,galcntr_grid, cmap = 'inferno')
plt.xlabel('$\ell$')
plt.ylabel('$b$')
# plt.ylim(-np.pi/2.1,np.pi/2.1)
# plt.xlim(-np.pi/1.1,np.pi/1.1)
# plt.xticks([-3,-2,-1,0,1,2,3])
plt.colorbar()
plt.savefig("j-factor intensity map.pdf", format = 'pdf', dpi = 500, bbox_inches = 'tight')


