# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 11:59:36 2022

@author: mjafs
"""

"""
Routine which manually computes the chi-squared function for a given set of 
mass values (in GeV) and finds the minimum value. Note that this is a manually
assumbled routine. See 'least squares fit.py' for the routine that 
accomplishes this same tast using scipy.optimize
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import dblquad,quad,tplquad,nquad,quadrature, quad_vec
from scipy.interpolate import interp1d, interp2d
from scipy.special import erf
import timeit
from numba import jit

"Good figure template example"
# niceFigure example with good reasonable size fonts and TeX fonts
def niceFigure(useLatex=True):
    from matplotlib import rcParams
    plt.rcParams.update({'font.size': 30})
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

# ell = np.arange(-np.pi,np.pi,0.5)
# b = np.arange(-np.pi/2.1,np.pi/2.1,0.5)


#for data set 4 (the 'full' set)
ell = np.linspace(-0.539524905,0.5837143902,66)
b = np.linspace(-7*np.pi/120,7*np.pi/120,10)


#for vincent data
# ell = np.linspace(0,0.4305557968,39)  #data points from digitized plot in this range 
# b = np.linspace(-np.pi/12,np.pi/12,20)    #bins had a height of 21 degrees = 7pi/60 rads


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

    
    # full_int_positive = np.where(full_int < 0, 0, full_int)  #this was causing all the fit problems so I have it removed at the moment
    return full_int  


def theor_data(M, dmin, dmax):
    """
    Definition which retrieves the array of the theoretical observed flux 
    per galactic longitude
    """
    
    #retrieve threshold velocity from the mass
    M_chi = (M*1000*J)/(c**2)       #mass of WIMP (kg) 
    v_thresh = np.sqrt(4*(delta/M_chi))     #threshold velocity (m/s)
    vth_cgs = v_thresh*100   #converting threshold velocity to cgs units (cm/s)
    sigC = (9e-26)*M**2  
    
    #compute the observed flux (in 1D array)
    cntrarr1, error = quad_vec(flux_integ, intmin, intmax, args = (sigC, vth_cgs, M))
    cnt1 = np.reshape(cntrarr1,(len(b),len(ell)))   #convert to 2D (rows,cols) = (b,\ell)
    
    """
    NOTE: I have purposely neglected to include an 'x =' in trapz
    because it gives a value closer to the exp data... Still need 
    to figure out why this is/ if this is ok to do.
    """
    fluxperlong = np.trapz(cnt1, x = b, axis = 0)   #integrate along x (collapse b vals)
    # fluxperlong = np.trapz(fluxperlong, x = ell)   #leave this in if want full flux over solid angle range
    normflux = fluxperlong/0.019
    # fluxperlong = quad(cnt1, -np.pi, np.pi)   #integrate along x (collapse b vals)

    return normflux
    # return fluxperlong


#%%
# Oarr, Tarr = np.array([1,2,3]), np.array([3,4,5])
def chi_squared(Obs_arr, Theor_arr): #(experimental, observed/theoretical)
    """
    Compute the chi squared statistic given an array of theoretical and
    observational data
    """
    
    Obs_arr[:] = np.where(Obs_arr[:] < 0, 0.0001, Obs_arr[:])

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
exp_long, exp_flux = np.loadtxt('dataset4(fullset).csv', unpack = True, delimiter = ',')

# exp_long, exp_flux = np.loadtxt('vincentflux.csv', unpack = True, delimiter = ',')

# print(exp_flux)

#convert exp. longitudes to units of radians
exp_long_rads = exp_long*np.pi/180

print(exp_flux)
#%%
#theoretical data with bins corresponding to experimental data
Marr = np.linspace(3000,5000,20)   #array of possible WIMP masses
theor_flux = np.zeros((len(Marr),len(ell)))

start = timeit.default_timer()
#calculate the theoretical flux array for each mass 
for i,val in enumerate(Marr):
    theor_flux[i,:] = theor_data(Marr[i], intmin, intmax)
stop = timeit.default_timer()
print("The time to solve is", stop - start, "seconds")

#%%
theor_flux = theor_data(100000, intmin, intmax)
# theor_flux = theor_flux*2
#%%
niceFigure(True)
#check for shape of theoretical matrix
# print(theor_flux[0])
# print(exp_flux)
#plot some trial values to see if things make sense
# plt.plot(ell, theor_flux[47,:])
plt.plot(ell*180/np.pi, theor_flux[12],'--',color = "blue", label = "NFW only")
# plt.plot(ell,theor_flux)
plt.plot(exp_long_rads*180/np.pi, exp_flux,'.',color = 'magenta', label = "SPI data")   #for use with vincent data
# plt.plot(exp_long_rads, exp_flux,'.', label = "SPI data")
plt.errorbar(exp_long_rads*180/np.pi, exp_flux, yerr = 0.0018149, color = 'black', alpha = 0.4)   #size of error bar estimated from digitized data 
plt.xlabel("$\ell$ (degrees)")
plt.ylabel("$\Phi$ (ph cm$^{-2}$ s$^{-1}$ sr$^{-1}$)")
# plt.ylabel("$\Phi$ (ph cm$^{-2}$ s$^{-1}$ rad$^{-1}$)")
# plt.xticks([0,5,10,15,20,25])
plt.legend(fontsize = 25)
plt.savefig('theorsiegm={}.png'.format(Marr[12]), format = 'png', dpi = 200, bbox_inches = 'tight')
# plt.savefig('theorvincent.png', format = 'png', dpi = 200, bbox_inches = 'tight')
#%%%
"Calculate chi-squared using SPI data"

chisqr = np.zeros(len(Marr))
for i, val in enumerate(theor_flux):
    # print(i,theor_flux[i])
    # print(i,val)
    chisqr[i] = chi_squared(exp_flux, theor_flux[i])
    print(chisqr)
#this didn't work
# for i in theor_flux[i,:]:
    # chisqr[i] = chi_squared(exp_flux,theor_flux[i])


#for if just one mass value is used and not an array
# chisqr = chi_squared(exp_flux, theor_flux)
# print(chisqr)
#%%

"Chi-squared graphics and final results"

#plot the chi squaredvalues as a funciton of the WIMP mass
plt.plot(Marr,chisqr)
plt.ylabel("$\chi^2$")
plt.xlabel("WIMP mass (GeV)")

#print the answer for the 'best' mass our model gives
# print(chisqr)
min_chi_squared = min(chisqr)
index = np.where(chisqr == min_chi_squared)
plt.savefig('chi_sieg.png', format = 'png', dpi = 200, bbox_inches = 'tight')
print("The minimum chisquared value is:", min_chi_squared, "with an index of", index, "in the 'mass' array")
print()
print("Therefore the mass value of interest is:", Marr[index], "GeV")

#%%
"graphics for intensity mapping"
# print(cntrarr)
# cntrarr[cntrarr < 0] = 0   #force any negative intensities to 0
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
plt.savefig("DM-screened-intensity map_mass={}.pdf".format(M), format = 'pdf', dpi = 200, bbox_inches = 'tight')
