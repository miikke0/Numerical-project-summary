# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 13:50:01 2022

@author: mjafs
"""


import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import dblquad,quad,tplquad,nquad,quadrature, quad_vec
from scipy.interpolate import interp1d, interp2d
from scipy.special import erf
import timeit
from numba import jit
from scipy.optimize import curve_fit

"Good figure template example"
# niceFigure example with good reasonable size fonts and TeX fonts
def niceFigure(useLatex=True):
    from matplotlib import rcParams
    plt.rcParams.update({'font.size': 13})
    if useLatex is True:
        plt.rc('text', usetex=True)
        plt.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    rcParams['xtick.major.width'] = 1
    rcParams['ytick.major.width'] = 1
    return


"Load in the dispersion velocities"
Dist, Vel = np.loadtxt('disp_vel.csv',unpack = True, skiprows = 1 ,delimiter = ',') #skip the first row of data

dist = Dist*(3.0857*10**21)
vel = Vel*100  #scaled by factor of 100 since the original data set had values in m/s not cm/s
vel1 = vel*2

vel_km = vel/(100*1000)
print(dist)
sigma = interp1d(dist, vel, kind = 'cubic', bounds_error = False, fill_value = 70000)
# sigma = interp1d(dist, vel1, kind = 'cubic', bounds_error = False, fill_value = 70000)

print("{:.3}".format(vel[30]))
print(sigma(0.0120))
print(max(sigma(dist)/(100*1000)))
xvals = np.logspace(0,20,50)

niceFigure(True)
fig, ax = plt.subplots()
ax.loglog(Dist,sigma(dist)/(100*1000),'--', color = "purple")
ax.set_yticks([60,100,150,200])
ax.tick_params(axis='x', labelsize=25)
ax.tick_params(axis='y', labelsize=25)

plt.xlabel("Galactic Radius (kpc)",fontsize = 30)
plt.ylabel("Dispersion velocity (km/s)", fontsize = 30)
plt.show()

#%%

#values needed to determine the threshold velocity
#need to first convert the mass of a given WIMP from and energy scale to a mass scale

"Tidy this section up!!!!"

c = 3e8                   #speed of light (m/s)
J = 1.6022e-13            #conversion factor from 1 MeV to 1 Joule: 
delta = 1.22*J          #energy separation in Joules (separation is quoted as 1.22MeV (ie the mass of two electrons))
# M = 6000                   #mass of WIMP (GeV)
#(^this seems to need to be roughly greater than 1600 GeV to not return a nonsense answer for flux)
# M_list = [1600,10000,20000,40000]
# M_chi = (M*1000*J)/(c**2)       #mass of WIMP (kg) 
kpc = 3.086e21

# v_thresh = np.sqrt(4*(delta/M_chi))     #threshold velocity (m/s)
# vth_cgs = v_thresh*100   #converting threshold velocity to cgs units (cm/s)

# def Mass(M):
#     M_chi = (M*1000*J)/(c**2)       #mass of WIMP (kg) 
#     v_thresh = np.sqrt(4*(delta/M_chi))     #threshold velocity (m/s)
#     vth_cgs = v_thresh*100   #converting threshold velocity to cgs units (cm/s)
#     sigC = (8.2e-26)*M**2      #Scaled by the mass of the wimp as per the units listed in table 1 in the paper above
#     return sigC, M, vth_cgs

# sigC, M, vth_cgs = Mass(2000)
#<sigmav>_chi constant. This is the DM cross-section we are assuming is constant 
#This piece goes into the thermally averaged annihilation rate coefficient
#taken from https://arxiv.org/pdf/1201.0997.pdf, page 4 and using the value for NFW + Disk

# sigC = (8.2e-26)*M**2      #Scaled by the mass of the wimp as per the units listed in table 1 in the paper above

# print(sigC)
# print("{:.3}".format(vth_cgs))
# print(vth_cgs/1000)



#%%
#Defining constants needed for integration
#Constants are based on values given in page 4 of ;/https://arxiv.org/pdf/1912.09486.pdf
f_p = 0.967 #positronium fraction
rho0=0.4 #GeV cm^-3
R0=8.5*(3.0857e21)  #Converting all units in kpc to cm (Conversion factor is 3.0857*10**21 cm/kpc)
rs=26*(3.0857e21)
Rhalo=30*(3.0857e21)
xmax=Rhalo+R0             #This is based on eqn. 4 in above paper
gamma=1.2
rhos=rho0/((2**(3-gamma))/(((R0/rs)**gamma)*(1+(R0/rs))**(3-gamma)))  #scale radius equation (rearranged eqn. 5)
print(xmax)

G_N = (((6.67e-8))) #Universal gravitational constant?? units of (cm^3 g^-1 s^-2)
#DM mass gravitational potential

#%%


#CURRENTLY THE POTENTIAL SECTION THAT I AM USING
#Baryonic bulge mass gravitational potential:
def bulgepot(r):
    M_b = ((1.5e10*(1.989e33)))    #bulge mass in grams
    c_o = 0.6*(3.0857*10**21)              #bulge scale radius converted to cm
    return -(G_N*M_b)/(r + c_o)

#Baryonic disk mass gravitational potential:
def diskpot(r):
    M_d = ((7e10*(1.989e33)))       #disk mass converted from solar masses to GeV (this is in grams right now)
    b_d = 4*(3.0857*10**21)                 #disk scale radius converted to cm from kpc
    return -((G_N*M_d)/r)*(1 - np.exp(-r/b_d))

def dm_mass_integrand(r):
            rho_s=(rho0*1.78e-24)/((2**(3-gamma))/(((R0/rs)**gamma)*(1+(R0/rs))**(3-gamma)))
            rhoNFW =(rho_s*(2**(3-gamma)))/(((r/rs)**gamma)*(1+(r/rs))**(3-gamma))
            M_integrand = 4*np.pi*r**2*rhoNFW
            return M_integrand

Menc = lambda r: quad(dm_mass_integrand,0,r)[0]
Mass_enc = np.vectorize(Menc)        
        

def DMintegrand(r):        
        integrand = G_N*Mass_enc(r)/r**2
        return integrand        
        

DMpotential = lambda r: quad(DMintegrand, 1e24, r)[0]

rVals = np.linspace(0,1e24,1000)
DMarr = []
for i in rVals:
    DM = DMpotential(i)
    DMarr.append(DM)
# print(DMarr)


plt.plot(rVals,DMarr)
# DMpotent = np.vectorize(DMpotential)

#interpolated expression for the integral over enclosed mass of DM potential
DMpotent = interp1d(rVals,DMarr, kind = 'cubic', bounds_error = False, fill_value = '70000')

#check shape of the dm potential
plt.plot(rVals,DMpotent(rVals))

# testr = 2e20
# print(Mass_enc(testr))
# print(dm_mass_integrand(testr))
# print(DMintegrand(testr))
# print(DMpotential(testr))
# print(DMpotent(testr))
# print()


#escape velocity as defined from the potentials above
v_esc = lambda r: np.sqrt(2*(-(DMpotent(r) + bulgepot(r) + diskpot(r))))

#%%
"Chi squared statistic for observed flux (algorithm)"

#galactic longitudes and latitudes that match the SPI observations
# ell = np.linspace(-0.3354069037,0.4158437184,44)  #data points from digitized plot in this range 
# b = np.linspace(-7*np.pi/120,7*np.pi/120,10)    #bins had a height of 21 degrees = 7pi/60 rads

#for data set 2
# ell = np.linspace(-0.3368380737,0.413771838,44)  #data points from digitized plot in this range 
# b = np.linspace(-7*np.pi/120,7*np.pi/120,20)    #bins had a height of 21 degrees = 7pi/60 rads
#if skipping first 10 rows
# ell = np.linspace(-0.3368380737,0.2400499673,34)  #data points from digitized plot in this range 
# b = np.linspace(-7*np.pi/120,7*np.pi/120,20)    #bins had a height of 21 degrees = 7pi/60 rads
#if starting at 0
# ell = np.linspace(0,0.413771838,24)  #data points from digitized plot in this range 
# b = np.linspace(-7*np.pi/120,7*np.pi/120,20)    #bins had a height of 21 degrees = 7pi/60 rads
#try with arange for evenly spacing theoretical points
# dx = 1*np.pi/180
# ell = np.arange(-0.3368380737,0.413771838,dx)  #data points from digitized plot in this range 
# b = np.linspace(-7*np.pi/120,7*np.pi/120,10)    #bins had a height of 21 degrees = 7pi/60 rads

#for data set 4 (the 'full' set)
ell = np.linspace(-0.539524905,0.5837143902,66)
b = np.linspace(-7*np.pi/120,7*np.pi/120,10)


#for vincent data
# ell = np.linspace(0,0.4305557968,39)  #data points from digitized plot in this range 
# b = np.linspace(-np.pi/12,np.pi/12,30)    #bins had a height of 21 degrees = 7pi/60 rads


#define grid values (galactic coords)
# ell = np.linspace(-np.pi,np.pi,50)     #galactic longitude
# b = np.linspace(-np.pi/2.1,np.pi/2.1,50)   #latitude
L, B = np.meshgrid(ell,b)
print(np.shape(ell))
# print(b)

#screened flux integrand
intmin, intmax = 0, 1e26 #integrate from min to max
def flux_integ(d, sigC, vth_cgs, M):   
    #vectorized integrand to compute DM flux
    r = np.sqrt(d**2 + R0**2 - 2*R0*d*np.cos(B.ravel())*np.cos(L.ravel()))   #distance from glactic center  
    rho_x=((rhos*2**(3-gamma))/(((r/rs)**gamma)*(1+(r/rs))**(3-gamma)))**2 #Squaring eqn.5

    #this function defines the escape velocity of a test particle as a result of both dark and baryonic matter
    v_esc = lambda r: np.sqrt(2*(-(DMpotent(r) + bulgepot(r) + diskpot(r))))
    
    #the next two lines make of the second half on the analytical expression for the integral in 3 space over the velocity distribution    
    f_v3 = lambda r:(2*v_esc(r)*np.exp(-(v_esc(r)**2)/(4*sigma(r)**2))*sigma(r)*2) 
    f_v4 = lambda r: (2*sigma(r)**3*np.sqrt(np.pi)*erf(v_esc(r)/(2*sigma(r))))
    
    #Normalization factor: normalizes the velocity integral to 1 when v_thresh = 0
    N = lambda r: 1/(-f_v3(r) + f_v4(r))
    
    #the next two lines make of the first half on the analytical expression for the integral in 3 space over the velocity distribution
    f_v1 = lambda r: (2*vth_cgs*np.exp(-(vth_cgs**2)/(4*sigma(r)**2))*sigma(r)*2) 
    f_v2 = lambda r:(2*sigma(r)**3*np.sqrt(np.pi)*erf(vth_cgs/(2*sigma(r))))
            
    F = lambda r: (f_v1(r) - f_v2(r) - f_v3(r) + f_v4(r))
    
    
    vel_int = lambda r: N(r)*F(r)                          #with normalization
    velocity_CS = lambda r: sigC*vel_int(r)
    # velocity_CS = lambda r: sigC
    
    #the full integrand is divided by 4pi steradians to achieve units of flux scaled by the mass
    full_int=2*(1 - 0.75*f_p)*(1/(4*np.pi))*np.cos(B.ravel())*(1/2)*rho_x*velocity_CS(r)/(M**2)    #Multiplying by sintheta. Note that rho_x is actually rho**2. We do this in the 3rd line above

    full_int_positive = np.where(full_int < 0, 0, full_int)
    return full_int_positive
    # return full_int


def theor_data(arg,M,sig_v):
    """
    Definition which retrieves the array of the theoretical observed flux 
    per galactic longitude
    """
    # barr, ellarr = np.reshape(arg, (len(b),len(ell))) 
    ellarr = arg
    #retrieve threshold velocity from the mass
    M_chi = (M*1000*J)/(c**2)       #mass of WIMP (kg) 
    v_thresh = np.sqrt(4*(delta/M_chi))     #threshold velocity (m/s)
    vth_cgs = v_thresh*100  #converting threshold velocity to cgs units (cm/s)
    # sigC = (9.5e-27)*M**2  
    sigC = sig_v*M**2    #this lets us vary the DM crosssection in our fit
    
    #compute the observed flux (in 1D array)
    cntrarr1, error = quad_vec(flux_integ, intmin, intmax, args = (sigC, vth_cgs, M))
    cnt1 = np.reshape(cntrarr1,(len(b),len(ellarr)))   #convert to 2D (rows,cols) = (b,\ell)

    fluxperlong = np.trapz(cnt1, x = b, axis = 0)   #integrate along x (collapse b vals)
    # fluxperlong = np.trapz(fluxperlong, x = ell)   #leave this in if want full flux over solid angle range
    normflux = fluxperlong/0.01919

    return normflux
    # return fluxperlong
    # return cnt1



#%%

"Run this cell for total flux in inner 30 degrees calculation"
start = timeit.default_timer()

theor_flux = theor_data(ell, popt[0]   , popt[1])
print(theor_flux)
stop = timeit.default_timer()
print('The time to solve is',stop - start)
#%%
"Contour Graphics"
log_cntrarr =np.log(theor_flux)
galcntr_grid = np.reshape(log_cntrarr,(len(b),len(ell)))   #convert 1D array to grid  

#plot the contour map
plt.figure()
niceFigure(True)
# plt.subplot(111, projection="aitoff")
plt.contourf(ell, b,galcntr_grid, cmap = 'inferno')
plt.grid(True)
plt.xlabel('$\ell$')
plt.ylabel('$b$')
# plt.ylim(-np.pi/2.1,np.pi/2.1)
# plt.xlim(-np.pi/1.1,np.pi/1.1)
# plt.xticks([-3,-2,-1,0,1,2,3])
plt.colorbar()
# plt.savefig("DM-screened-intensity map_mass={}.pdf".format(M), format = 'pdf', dpi = 200, bbox_inches = 'tight')


#%%
# Oarr, Tarr = np.array([1,2,3]), np.array([3,4,5])
def chi_squared(Obs_arr, Theor_arr): #(experimental, observed/theoretical)
    """
    Compute the chi squared statistic given an array of theoretical and
    observational data
    """
    Obs_arr[:] = np.where(Obs_arr[:] < 0, 0.000001, Obs_arr[:])

    
    chisqr = ((Obs_arr[:] - Theor_arr[:])**2)/Obs_arr[:]
    chisum = np.cumsum(chisqr)
    return chisum[-1]
   
# chi = chi_squared(Tarr, Oarr)
# print(chi)

# matrix = np.array([[1,2],[3,4]])
# # print(matrix) 
# exp = np.array([1,2])
# chi = np.zeros(2)
# for i, val in enumerate(matrix):
#     chi[i] = chi_squared(exp, matrix[i])
    
# print(chi)
#%%
"Load in exp data and calculate theoretical values"

#digitized data
# exp_long, exp_flux = np.loadtxt('dataset2.csv', unpack = True, delimiter = ',')
# exp_long, exp_flux = np.loadtxt('dataset2.csv', unpack = True, delimiter = ',', skiprows = 10)
# exp_long, exp_flux = np.loadtxt('dataset3.csv', unpack = True, delimiter = ',')

#for 'full' dataset
exp_long, exp_flux = np.loadtxt('dataset4(fullset).csv', unpack = True, delimiter = ',')
# exp_flux = np.where(exp_flux < 0, 0.0001, exp_flux)

# exp_long, exp_flux = np.loadtxt('vincentflux.csv', unpack = True, delimiter = ',')

# print(exp_flux)

#convert exp. longitudes to units of radians
exp_long_rads = exp_long*np.pi/180
#%%
"Try to fit with scipy opt"

#initial guesses
M_init, sig_v_init = 4263, 9e-26
init = [M_init, sig_v_init]
#run the curve fit
start = timeit.default_timer()
popt1, pcov1 = curve_fit(theor_data, exp_long_rads, exp_flux, p0 = init)
stop = timeit.default_timer()
print('The time to solve is',stop - start)
# popt2, pcov2 = curve_fit(theor_data, exp_long_rads, exp_flux, p0 = init)



#%%

# from sklearn import preprocessing

# import scipy.stats as st

"Print fit params and cacluate the chi-sqaured statistic"
#print optimized parameters
print("The optimal mass is:," ,popt1[0], "GeV. The optimal DM cross-section is:,",popt1[1])
theor_flux1 = theor_data(ell,popt1[0], popt1[1])
# theor_flux2 = theor_data(ell,popt2[0], popt2[1])

# theor_flux = theor_data(ell,5000, 4e-26)
# print(pcov)
# eigen = np.linalg.eig(pcov)
# print(eigen)
# print(np.shape(theor_flux))
# print(np.shape(exp_long))

#%%
# from sklearn.preprocessing import StandardScaler
# from scipy.stats import chi2_contingency
# from scipy.stats import power_divergence

# # N1 = preprocessing.normalize([theor_flux])
# # N2 = preprocessing.normalize([exp_flux])
# # print(N1)
# # print(N2)
# exp_flux = np.where(exp_flux < 0, 0.0001, exp_flux)
# theor_flux = np.where(theor_flux < 0, 0.0001, theor_flux)

# print(exp_flux)
# # N1 = (theor_flux - np.min(theor_flux)) / (np.max(theor_flux) - np.min(theor_flux))
# # print(N1)
# # N2 = (exp_flux - np.min(exp_flux)) / (np.max(exp_flux) - np.min(exp_flux))
# # print(N2)
# # # print(np.cumsum(N2))
# # # print(np.trapz(N))
# dataset1 = theor_flux.reshape(-1, 1)
# scaler1 = StandardScaler()
# scaler1.fit(dataset1)
# standardized_dataset1 = scaler1.transform(dataset1)
# print(standardized_dataset1)
# print(np.mean(standardized_dataset1))
# print(np.std(standardized_dataset1))

# dataset2 = exp_flux.reshape(-1, 1)
# scaler2 = StandardScaler()
# scaler2.fit(dataset2)
# standardized_dataset2 = scaler2.transform(dataset2)
# print(standardized_dataset2)
# print(np.mean(standardized_dataset2))
# print(np.std(standardized_dataset2))
# sq1 = np.reshape(standardized_dataset1,(66))
# sq2 = np.reshape(standardized_dataset2,(66))

# print(sq1)

# print(np.shape(standardized_dataset1))
#%%

# def chi_sq2(expd, thd):
#     expdN = preprocessing.normalize([expd])
#     thdN = preprocessing.normalize([thd])
#     chi2 = st.chisquare(thdN,expdN)
#     return chi2


# c = chi_sq2(standardized_dataset1[0], standardized_dataset2[0])
# print(c)

#%%
    
# chisq = st.chisquare(standardized_dataset1[0], standardized_dataset2[0])
# chisq = power_divergence(sq1, sq2, lambda_ = 1)
# print(chisq)
# print(chisq)
# print(chisq)

"Find the chi squared statistic"
chisq1 = chi_squared(exp_flux, theor_flux1)
print(chisq1)

# cum = np.cumsum(theor_flux)
# print(cum[-1])
# cum2 = np.cumsum(exp_long)
# print(cum2[-1])
# print(standardized_dataset2)

#%%

#plot the best fit solution with the SPI data
niceFigure(True)
#check for shape of theoretical matrix
# print(theor_flux[0])
# print(exp_flux)
#plot some trial values to see if things make sense
# plt.plot(ell, theor_flux[47,:])
# plt.plot(ell, theor_flux[5], label = "theoretical")
plt.plot(ell*180/np.pi,theor_flux1, '--',color = 'blue', label = '$\gamma = 1.2$')
# plt.plot(ell*180/np.pi,theor_flux2, '--',color = 'red', label = '$\gamma = 0.8$')
plt.plot(exp_long_rads*180/np.pi, exp_flux,'.',color = 'magenta', label = "SPI data")
plt.errorbar(exp_long_rads*180/np.pi, exp_flux,color = 'black',alpha = 0.4, yerr = 0.0018149)
plt.xticks([-30, -20, -10,0,10,20,30])
plt.yticks([0.000, 0.005, 0.010, 0.015, 0.020])
plt.xlabel("Galactic Longitude (degrees)")
plt.ylabel("Flux (ph cm$^{-2}$ s$^{-1}$ sr$^{-1}$)")
plt.legend()
plt.savefig('theorscipysiegcomp.png', format = 'png', dpi = 400, bbox_inches = 'tight')


















