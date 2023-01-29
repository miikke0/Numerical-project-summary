# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 12:26:59 2022

@author: mjafs
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import dblquad,quad,tplquad,nquad,quadrature, quad_vec
from scipy.interpolate import interp1d, interp2d
from scipy.special import erf
import timeit
# from numba import jit

#%%
"Good figure template example"
# niceFigure example with good reasonable size fonts and TeX fonts
def niceFigure(useLatex=True):
    from matplotlib import rcParams
    plt.rcParams.update({'font.size':20})
    if useLatex is True:
        plt.rc('text', usetex=True)
        plt.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    rcParams['xtick.major.width'] = 1
    rcParams['ytick.major.width'] = 1
    return

#%%

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
print(max(sigma(dist)/(1000*100)))
xvals = np.logspace(0,20,50)

niceFigure(True)
fig, ax = plt.subplots()
ax.loglog(Dist,sigma(dist)/(100*1000),'--', color = "purple")
ax.set_yticks([60,100,150,200])
ax.set_xticks([1e-2,1e-1,1e0, 1e1, 1e2])
# ax.tick_params(axis='x', labelsize=25)
# ax.tick_params(axis='y', labelsize=25)
plt.xlabel("R (kpc)")
plt.ylabel("$\sigma_v$ (km/s)")
plt.savefig('dispersionrel.png', format = 'png', dpi = 200, bbox_inches = 'tight')
plt.show()

#%%

print(sigma(dist[0]))
# print(sigma(1e20))

#%%


#values needed to determine the threshold velocity
#need to first convert the mass of a given WIMP from and energy scale to a mass scale

"Tidy this section up!!!!"

c = 3e8                   #speed of light (m/s)
J = 1.6022e-13            #conversion factor from 1 MeV to 1 Joule: 
delta = 1.22*J          #energy separation in Joules (separation is quoted as 1.22MeV (ie the mass of two electrons))
M1 = 10000                 #mass of WIMP (GeV)
M2 = 20000                 #mass of WIMP (GeV)
M3 = 40000                 #mass of WIMP (GeV)
M = 900 
# M = 2861.259432339565
# M = 2882.9090560104414
M = 2861.135450803292                #mass of WIMP (GeV)
#(^this seems to need to be roughly greater than 1600 GeV to not return a nonsense answer for flux)
M_list = [1600,10000,20000,40000]
M_chi = (M*1000*J)/(c**2)       #mass of WIMP (kg) 
kpc = 3.086e21

v_thresh = np.sqrt(4*(delta/M_chi))     #threshold velocity (m/s)
vth_cgs = v_thresh*100   #converting threshold velocity to cgs units (cm/s)



#<sigmav>_chi constant. This is the DM cross-section we are assuming is constant 
#This piece goes into the thermally averaged annihilation rate coefficient
#taken from https://arxiv.org/pdf/1201.0997.pdf, page 4 and using the value for NFW + Disk

sigC = (8.2e-26)*M**2      #Scaled by the mass of the wimp as per the units listed in table 1 in the paper above
# sigC = 4.95e-24
# sigC = 1.8058030004078898e-25
# sigC = 1.9197016376490095e-25
print(sigC)
print("{:.3}".format(vth_cgs))
print(vth_cgs/(100))

#%%


#Defining constants needed for integration
#Constants are based on values given in page 4 of ;/https://arxiv.org/pdf/1912.09486.pdf
f_p = 0.967 #positronium fraction
rho0=0.4 #GeV cm^-3
R0=8.5*(3.0857e21)  #Converting all units in kpc to cm (Conversion factor is 3.0857*10**21 cm/kpc)
rs=26*(3.0857e21)  #this is 12 from the 'results' section of paper by vincent et al
Rhalo=30*(3.0857e21)
xmax=Rhalo+R0             #This is based on eqn. 4 in above paper
gamma=1.32
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

DMpotent = interp1d(rVals,DMarr, kind = 'cubic', bounds_error = False, fill_value = '70000')

plt.plot(rVals,DMpotent(rVals))

testr = 2e20
print(Mass_enc(testr))
print(dm_mass_integrand(testr))
print(DMintegrand(testr))
print(DMpotential(testr))
print(DMpotent(testr))
print()

#%%

v_esc = lambda r: np.sqrt(2*(-(DMpotent(r) + bulgepot(r) + diskpot(r))))
# rv = np.logspace(16,24)
rv = np.logspace(16,25)

print(max(v_esc(rv)))
niceFigure(True)
kpc1 = 3.24078e-22 #amount of cm in a kpc
fig, ax = plt.subplots()
# ax.tick_params(axis='x', labelsize=15)
# ax.tick_params(axis='y', labelsize=15)
ax.plot(rv*kpc1,v_esc(rv)/(100*1000), label = "$v_{esc}$")
ax.loglog(rv*kpc1,np.sqrt(-2*DMpotent(rv))/(100*1000),'-.', label = "$v_{esc}$ from $\Phi_{DM}$")
ax.loglog(rv*kpc1,np.sqrt(-2*bulgepot(rv))/(100*1000),':', label = "$v_{esc}$ from $\Phi_{bulge}$" )
ax.loglog(rv*kpc1,np.sqrt(-2*diskpot(rv))/(100*1000),'--',label = "$v_{esc}$ from $\Phi_{disk}$")
ax.set_yticks([1e2, 1e3, 2e3])
ax.set_xticks([1e-5,1e-3,1e-1,1e1,1e3])
plt.xlabel('R (kpc)')
plt.ylabel('$v_{esc}$ (km/s)')
plt.legend(fontsize = 16)

plt.show()

#%%
# rv = np.logspace(1, 24)
rv = np.linspace(1e6,1e24)
niceFigure(True)
fig, ax = plt.subplots()
# ax.tick_params(axis='x', labelsize=25)
# ax.tick_params(axis='y', labelsize=25)
ax.plot(rv*kpc1,DMpotent(rv)/(100*1000),'--', label = '$\Phi_{DM}$')
ax.plot(rv*kpc1,bulgepot(rv)/(100*1000),':', label = '$\Phi_{bulge}$', color='magenta')
ax.plot(rv*3.24078e-22,diskpot(rv)/(100*1000), '-.',label = '$\Phi_{disk}$')
# ax.loglog(rv,DMpotent(rv),'--', label = 'Dark Matter Potential')
# ax.loglog(rv,bulgepot(rv),'x', label = 'Disk Potential', color='magenta')
# ax.loglog(rv,diskpot(rv), '>',label = 'Bulge Potential')
# plt.xscale('symlog')
# plt.yscale('symlog')
plt.xlabel('R (kpc)')
plt.ylabel('$\Phi(R)$ (km$^2$ s$^{-2}$)')
plt.legend(fontsize = 16)
plt.savefig('potentials.png', format = 'png', dpi = 200, bbox_inches = 'tight')


#%%


f_p = 0.967 #positronium fraction
# sigC = (8.2e-26)*M**2      #Scaled by the mass of the wimp as per the units listed in table 1 in the paper above


"Compute all-sky flux (most accurate way of getting the total flux)"
def integrand1(xArr,theta):
    r=np.sqrt(R0**2-2*xArr*R0*np.cos(theta)+xArr**2)                       #applying eqn 3
    rho_x=((rhos*2**(3-gamma))/(((r/rs)**gamma)*(1+(r/rs))**(3-gamma)))**2 #Squaring eqn.5

    # #this function defines the escape velocity of a test particle as a result of both dark and baryonic matter
    # v_esc = lambda r: np.sqrt(2*(-(DMpotent(r) + bulgepot(r) + diskpot(r))))
    
    # #the next two lines make of the second half on the analytical expression for the integral in 3 space over the velocity distribution    
    # f_v3 = lambda r:(2*v_esc(r)*np.exp(-(v_esc(r)**2)/(4*sigma(r)**2))*sigma(r)*2) 
    # f_v4 = lambda r: (2*sigma(r)**3*np.sqrt(np.pi)*erf(v_esc(r)/(2*sigma(r))))
    
    # #Normalization factor: normalizes the velocity integral to 1 when v_thresh = 0
    # N = lambda r: 1/(-f_v3(r) + f_v4(r))
    
    # #the next two lines make of the first half on the analytical expression for the integral in 3 space over the velocity distribution
    # f_v1 = lambda r: (2*vth_cgs*np.exp(-(vth_cgs**2)/(4*sigma(r)**2))*sigma(r)*2) 
    # f_v2 = lambda r:(2*sigma(r)**3*np.sqrt(np.pi)*erf(vth_cgs/(2*sigma(r))))
            
    # F = lambda r: (f_v1(r) - f_v2(r) - f_v3(r) + f_v4(r))
    
    
    # vel_int = lambda r: N(r)*F(r)                          #with normalization
    # velocity_CS = lambda r: sigC*vel_int(r)
    velocity_CS = lambda r: sigC    #simulates no splitting (faster than leacing in the line above)

    
    #the full integrand is divided by 4pi steradians to achieve units of flux scaled by the mass
    full_int=2*(1 - 0.75*f_p)*(1/(4*np.pi))*np.sin(theta)*(1/2)*rho_x*velocity_CS(r)/(M**2)    #Multiplying by sintheta. Note that rho_x is actually rho**2. We do this in the 3rd line above
    return full_int


#NOTE: 2pi term comes from integrating phi angle from [0,2pi] in the dOmega term (solid angle)
#NEED TO FIGURE OUT WHY IT PRINTS A NEGATIVE ANSWER... (update: happens is WIMP mass too low)
#I need to double check units here...

start = timeit.default_timer()

DthetaDx=dblquad(integrand1,0,np.pi,0,xmax)
flux = DthetaDx[0]*2*np.pi
print(flux,"photons cm^-2 s^-1")

stop = timeit.default_timer()

print("Calculation time is: ", stop - start)


#%%


"Compute intensity using loop structure and full quad integration (this is slow)"
def integrand2(d,b,l):
    r = np.sqrt(d**2 + R0**2 - 2*R0*d*np.cos(b)*np.cos(l))   #distance from glactic center  
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
    
    #the full integrand is divided by 4pi steradians to achieve units of flux scaled by the mass
    full_int=2*(1 - 0.75*f_p)*(1/(4*np.pi))*np.cos(b)*(1/2)*rho_x*velocity_CS(r)/(M**2)    #Multiplying by sintheta. Note that rho_x is actually rho**2. We do this in the 3rd line above
    # if full_int < 0:
    #     return 0
    # else:
    #     return full_int
    full_int_positive = np.where(full_int < 0, 0, full_int)
    return full_int_positive
    # return full_int



ell = np.linspace(-np.pi,np.pi,10)
b = np.linspace(-np.pi/2.1,np.pi/2.1,10)
cntrarr = []
start = timeit.default_timer()
for i in b:
    for j in ell:
        dJgal_cntr = quad(integrand2,0, 1e26, args = (i,j,))[0]
        cntrarr.append(dJgal_cntr)     #create 1D array for line of sight integral at various b and l
stop = timeit.default_timer()
print(stop-start)



#%%
print(cntrarr)

#%%
cnt = np.reshape(cntrarr,(len(b),len(ell)))
print(np.shape(cnt))
"Check full integral after using quad_vec"

b_int = np.trapz(cnt,x = b, axis = 0)    #integrate over the columns of the grid (ie theta)
ell_int = np.trapz(b_int,x = ell) 

print("The flux is:",ell_int)

#%%

"graphics for intensity mapping"
# print(cntrarr)
# cntrarr[cntrarr < 0] = 0   #force any negative intensities to 0
log_cntrarr = np.log(cntrarr)
galcntr_grid = np.reshape(log_cntrarr,(len(b),len(ell)))   #convert 1D array to grid  

#plot the contour map
niceFigure(True)
plt.figure()
# plt.subplot(111, projection="aitoff")
plt.contourf(ell, b,galcntr_grid, cmap = 'inferno')
plt.xlabel('$\ell$')
plt.ylabel('$b$')
# plt.ylim(-np.pi/2.1,np.pi/2.1)
# plt.xlim(-np.pi/1.1,np.pi/1.1)
# plt.xticks([-3,-2,-1,0,1,2,3])
plt.colorbar()
plt.savefig("DM-screened-intensity map_mass={}.pdf".format(M), format = 'pdf', dpi = 500, bbox_inches = 'tight')


#%%

"Compute intensity map using vectorized quad (slightly faaster)"
# ell = np.arange(-np.pi,np.pi,0.09)
# b = np.arange(-np.pi/2.1,np.pi/2.1,0.09)
ell = np.linspace(-np.pi,np.pi,300)
b = np.linspace(-np.pi/2.1,np.pi/2.1,300)
L, B = np.meshgrid(ell,b)

print(b)
#%%
# d = np.linspace(0,1e26,100)
intmin, intmax = 0, 1e26 
def integrand3(d):
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
    f_v1 = lambda r: (2*vth_cgs*np.exp(-((vth_cgs**2)/(4*sigma(r)**2)))*sigma(r)*2) 
    f_v2 = lambda r:(2*(sigma(r)**3)*np.sqrt(np.pi)*erf(vth_cgs/(2*sigma(r))))
            
    F = lambda r: (f_v1(r) - f_v2(r) - f_v3(r) + f_v4(r))
    
    
    vel_int = lambda r: N(r)*F(r)                          #with normalization
    velocity_CS = lambda r: sigC*vel_int(r)   
    # velocity_CS = lambda r: sigC   

    
    #the full integrand is divided by 4pi steradians to achieve units of flux scaled by the mass
    full_int=2*(1 - 0.75*f_p)*(1/(4*np.pi))*np.cos(B.ravel())*(1/2)*rho_x*velocity_CS(r)/(M**2)    #Multiplying by sintheta. Note that rho_x is actually rho**2. We do this in the 3rd line above
    # if full_int < 0:
    #     return 0
    # else:
        # return full_int
    full_int_positive = np.where(full_int < 0, 0, full_int)
    return full_int_positive
    # return full_int



# cntrarr = np.empty((len(b),len(ell)))

start = timeit.default_timer()
cntrarr1 = quad_vec(integrand3, intmin, intmax)[0]
# cntrarr[:,:] = quad(integrand2,0, 1e26, args = (b[:],ell[:],))      

stop = timeit.default_timer()
print(stop-start)

#%%

#determine total flux to compare to dblquad value
cnt1 = np.reshape(cntrarr1,(len(b),len(ell)))
print("the dimension is :",np.shape(cnt1))
print(cnt1)
"Check full integral after using quad_vec"
b_int1 = np.trapz(cnt1,x = ell, axis = 0)    #integrate over the columns of the grid (ie theta)
ell_int1 = np.trapz(b_int1,x = b) 
# b_int1 = np.trapz(cnt1, axis = 0)    #integrate over the columns of the grid (ie theta)
# ell_int1 = np.trapz(b_int1) 

print("The flux is:",ell_int1)

#%%

# import cartopy.crs as ccrs
"graphics for intensity mapping"
# print(cntrarr)
# cntrarr[cntrarr < 0] = 0   #force any negative intensities to 0
log_cntrarr =np.log(cntrarr1)
# log_cntrarr =(cntrarr1)


galcntr_grid = np.reshape(log_cntrarr,(len(b),len(ell)))   #convert 1D array to grid  
niceFigure(True)

#plot the contour map
plt.figure()
# plt.stock_img()
# ax = plt.axes(projection=ccrs.Mollweide(central_longitude=100))
plt.subplot(111, projection="hammer")
# ax = fig.add_subplot(111, projection=ccrs.Mollweide(central_longitude=0))
# ax.set_global()
# plt.contourf(ell*180/np.pi, b,galcntr_grid,cmap = 'inferno')
plt.contourf(ell, b,galcntr_grid,cmap = 'viridis', alpha = 1)

plt.grid(True)
plt.xlabel('Galactic Longitude (degrees)', labelpad = 10)
plt.ylabel('Galactic Latitude (degrees)')
plt.tick_params(axis='both', which='major', labelsize=12)

# plt.show()
# plt.ylim(-np.pi/2.1,np.pi/2.1)
# plt.xlim(-np.pi/1.1,np.pi/1.1)
# plt.xticks([-3,-2,-1,0,1,2,3])
plt.colorbar( shrink = 0.5, aspect = 16)
# cbar.xlabel('T (K)', labelpad=20)
plt.savefig("DM-screened-intensity map_massv2={}.png".format(M), format = 'png', dpi = 300, bbox_inches = 'tight')


  #%%

def F1(r):
    
    
    #Normalization factor: normalizes the velocity integral to 1 when v_thresh = 0
    # N = lambda r: 1/(-f_v3(r) + f_v4(r))
    
    #the next two lines make of the first half on the analytical expression for the integral in 3 space over the velocity distribution
    f_v11 = lambda r: (2*vth_cgs*np.exp(-(vth_cgs**2)/(4*sigma(r)**2))*sigma(r)*2) 
    f_v21 = lambda r:(2*sigma(r)**3*np.sqrt(np.pi)*erf(vth_cgs/(2*sigma(r))))
    f_v31 = lambda r:(2*v_esc(r)*np.exp(-(v_esc(r)**2)/(4*sigma(r)**2))*sigma(r)*2) 
    f_v41 = lambda r: (2*sigma(r)**3*np.sqrt(np.pi)*erf(v_esc(r)/(2*sigma(r))))
    F1 = (f_v11(r) - f_v21(r) - f_v31(r) + f_v41(r))
    # F = np.where(F1 < 0, 0, F1)
    return F1

rvals = np.linspace(0,1e24)
# F = F1(rvals)
# print(F)
# F[F<0] = 0
# print(F)
# Function1 = Function[Function < 0] = 0
plt.plot(rvals,F1(rvals))

#%%


def F2(r):
    f_v31 = lambda r:(v_esc(r)*np.exp(-(v_esc(r)**2)/(sigma(r)**2))*sigma(r)*2) 
    f_v41 = lambda r: (sigma(r)**3*np.sqrt(np.pi)*np.sqrt(2)*erf(np.sqrt(2)*v_esc(r)/(2*sigma(r))))/2
    
    #Normalization factor: normalizes the velocity integral to 1 when v_thresh = 0
    # N = lambda r: 1/(-f_v3(r) + f_v4(r))
    
    #the next two lines make of the first half on the analytical expression for the integral in 3 space over the velocity distribution
    f_v11 = lambda r: (vth_cgs*np.exp(-(vth_cgs**2)/(2*sigma(r)**2))*sigma(r)*2) 
    f_v21 = lambda r:(sigma(r)**3*np.sqrt(np.pi)*np.sqrt(2)*erf(np.sqrt(2)*vth_cgs/(2*sigma(r))))/2
            
    F2 = (f_v11(r) - f_v21(r) - f_v31(r) + f_v41(r))
    return F2

rvals = np.linspace(0,1e24)


plt.plot(rvals,F2(rvals))


#%%
def vel_dist(vrel,r):
    return np.exp(-vrel**2/(4*sigma(r)**2))
                       
rV = np.linspace(4e20,3e23,100)   #range of radii. sigma takes cm in the argument 
vV = np.linspace(0, 20000000,20)    #velocities in km/s


VV, RV = np.meshgrid(vV,rV)
Z = vel_dist(VV,RV)

niceFigure(True)
fig = plt.figure()   #create the figure outline
ax = plt.axes(projection = '3d')
# ax.contour3D(VV,RV,Z, 70, cmap = 'binary')
ax.plot_surface(VV/(100*1000),RV*kpc1,Z, rstride = 1, cstride = 1,cmap = 'viridis', edgecolor = 'none')
# ax.view_init(10,130)
ax.view_init(10,-30)
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.set_xlim(1e1, 1e22)
ax.set_xlabel('$v_{rel}$ (km/s)')
ax.set_ylabel('$R$ (kpc)')
ax.set_zlabel('Number of Particles')         
plt.savefig('3dvel_dist.png', format = 'png', dpi = 200, bbox_inches = 'tight')              
                       
#%%
def vel_dist_norm(v,r):
    phi = -v**2/(4*sigma(r)**2)
    factor = v**2*np.exp(phi)
    return factor

#%%

norm = lambda r: quad(vel_dist_norm, 0, 100, args = r)[0]
# print(norm(100))
print(norm(rv))
# plt.plot(rv, norm(rv))



#%%



def vel_dist(v_rel):
    
    integrand = 4*np.pi*N

#%%




f_p = 0.967 #positronium fraction
# sigC = (8.2e-26)*M**2      #Scaled by the mass of the wimp as per the units listed in table 1 in the paper above


"Compute all-sky flux (most accurate way of getting the total flux)"
def integrand4(xArr,theta):
    r=np.sqrt(R0**2-2*xArr*R0*np.cos(theta)+xArr**2)                       #applying eqn 3
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
    # velocity_CS = lambda r: sigC    #simulates no splitting (faster than leacing in the line above)

    
    #the full integrand is divided by 4pi steradians to achieve units of flux scaled by the mass
    full_int=2*(1 - 0.75*f_p)*(1/(4*np.pi))*np.sin(theta)*(1/2)*rho_x*velocity_CS(r)/(M**2)    #Multiplying by sintheta. Note that rho_x is actually rho**2. We do this in the 3rd line above
    return full_int


#NOTE: 2pi term comes from integrating phi angle from [0,2pi] in the dOmega term (solid angle)
#NEED TO FIGURE OUT WHY IT PRINTS A NEGATIVE ANSWER... (update: happens is WIMP mass too low)
#I need to double check units here...

start = timeit.default_timer()

DthetaDx=dblquad(integrand1,0,np.pi,0,xmax)
flux = DthetaDx[0]*2*np.pi
print(flux,"photons cm^-2 s^-1")

stop = timeit.default_timer()

print("Calculation time is: ", stop - start)






