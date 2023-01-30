# -*- coding: utf-8 -*-
"""
Late edited: April 2022

@author: mjafs
"""

"""
Start by running this first cell as it contain all necessary function etc. for later on. The questions are broken up into
their own cell and can each be run on their own. Only one ctrl + enter is needed per question. Question two allows for 
the user to select between which disperison model they want to use. See kw_f_mapper def below.

Our parallelized Q3 code is in another file  
"""
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import timeit
#from matplotlib import animation # not using, but you can also use this
import math
import scipy.constants as constants
import numba
from mpi4py import MPI
# comment these if any problems - sets graphics to auto
# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'auto')

"Good figure template example"
# niceFigure example with good reasonable size fonts and TeX fonts
def niceFigure(useLatex=True):
    from matplotlib import rcParams
    plt.rcParams.update({'font.size': 34})
    if useLatex is True:
        plt.rc('text', usetex=True)
        plt.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    rcParams['xtick.major.width'] = 1
    rcParams['ytick.major.width'] = 1
    return



"FFT function"
def calc_fft(Et_t, Er_t, Ein_t, fft_res, dt):
    "Takes Et(t), Er(t), Ein(t) (source field) and returns the reflection and transmission coefficients"
    
    N = np.size(Et_t)*fft_res

    omega = np.fft.fftfreq(N,dt)
    Einfreq = np.fft.fft(Ein_t,N)
    Etfreq = np.fft.fft(Et_t,N)
    Erfreq = np.fft.fft(Er_t,N)

    Ref = np.abs(Erfreq)**2/(np.abs(Einfreq)**2)
    Trans = np.abs(Etfreq)**2/(np.abs(Einfreq)**2)
    Sum = Ref + Trans
    return Ref, Trans, Sum, omega 


def dial_c(ncells, start, stop, eps, dx):
    """
    Returns an array defined by 1 everywhere, except the location of the thin film
    start: the grid point assigned to the start of the thin film
    stop: the grid point assigned to where the thin film ends
    ncells: the number of cells in z
    """
    cb = 0.5*np.ones(ncells)
    cb[start:stop] = 0.5/eps    #create the 'thin film'
    L = (stop - start)      #return the thickness of the thin film in SI units
    return cb, eps, L



def Epulse_init(t, t0, sigma, wL):
    init_pulse = -np.exp(-0.5*(t-t0)**2/sigma**2)*(np.cos(t*wL)) #initial Efield pulse
    return init_pulse

def Hpulse_init(t, t0, sigma, wL):
    #initial pulse in H with total field scattered field approach
    init_pulse = -np.exp(-0.5*((t + 1/2) - t0)**2/sigma**2)*(np.cos((t + 1/2)*wL))  #Hy pulse at \pm 0.5\Delta t relative to Ex
    return init_pulse


def clear_variables(tf):
    """
    Boolean. If True: Clear memory of specified variables from running prevous cells.
    """
    if tf == True:
        try:
            global time_pause, cycle, t_plot, graph_flag, Xmax, nsteps, ddx, dt, isource, filmstart \
                , filmstop, epsilon, cb, eps, L, spread, X1, t0, freq_in, w_scale, lam, ppw, xs, t, \
                    EinL, EtL, ErL, time, Ra, Ta, R, T, RT, omega, thickness, t_plot
                    
            del time_pause, cycle, t_plot, graph_flag, Xmax, nsteps, ddx, dt, isource, filmstart \
                , filmstop, epsilon, cb, eps, L, spread, X1, t0, freq_in, w_scale, lam, ppw, xs, t, \
                    EinL, EtL, ErL, time, Ra, Ta, R, T, RT, omega, thickness, t_plot
        except NameError:
            pass
    elif tf == False:
        pass
    
"Analytical functions for reflection and transmission coefficients"
def Ran(eps, w, l):  #dielectric constant, frequency, thin film thickness
    #analytical expression for the reflection coefficient
    l = l*ddx      #want to be able to enter the thickness in units of grid spacing function call
    n = np.sqrt(eps)
    r1 = (1 - n)/(1 + n)
    r2 = (n - 1)/(n + 1)
    k0 = w/(c)  
    exp = np.exp(2j*k0*l*n)
    
    r = (r1 + r2*exp)/(1 + r1*r2*exp)
    Ran = (abs(r))**2
    return Ran

def Tan(eps, w, l):  #dielectric constant, frequency, thin film thickness
    #analytical expression for the transmission coefficient
    l = l*ddx      #want to be able to enter the thickness in units of grid spacing function call
    n = np.sqrt(eps)
    r1 = (1 - n)/(1 + n)
    r2 = (n - 1)/(n + 1)
    k0 = w/c  
    exp1 = np.exp(1j*k0*l*n)
    exp2 = np.exp(2j*k0*l*n)

    t = ((1 + r1)*(1 + r2)*exp1)/(1 + r1*r2*exp2)
    Tan = (abs(t))**2
    return Tan

    
def Ran_drude(w, l, a):
    l = l*ddx      #want to be able to enter the thickness in units of grid spacing function call
    eps_w = 1 - (w_p**2)/(w**2 + 1j*w*a)
    n = np.sqrt(eps_w)
    r1 = (1 - n)/(1 + n)
    r2 = (n - 1)/(n + 1)
    k0 = w/(c)  
    exp = np.exp(2j*k0*l*n)
    
    r = (r1 + r2*exp)/(1 + r1*r2*exp)
    Ran_drude = (abs(r))**2
    return Ran_drude

def Tan_drude(w, l, a):  #frequency, thin film thickness (unit of grid spacing)
    l = l*ddx       #same as in Ran_drude function 
    #analytical expression for the transmission coefficient
    eps_w = 1 - (w_p**2)/(w**2 + 1j*w*a)
    n = np.sqrt(eps_w)
    r1 = (1 - n)/(1 + n)
    r2 = (n - 1)/(n + 1)
    k0 = w/c  
    exp1 = np.exp(1j*k0*l*n)
    exp2 = np.exp(2j*k0*l*n)

    t = ((1 + r1)*(1 + r2)*exp1)/(1 + r1*r2*exp2)
    Tan_drude = (abs(t))**2
    return Tan_drude

def Ran_Lorentz(w, l, a):
    l = l*ddx      #want to be able to enter the thickness in units of grid spacing function call
    eps_w = 1 + (f_0*w_0**2)/(w_0**2 - w**2 - 2j*w*a)
    n = np.sqrt(eps_w)
    r1 = (1 - n)/(1 + n)
    r2 = (n - 1)/(n + 1)
    k0 = w/(c)  
    exp = np.exp(2j*k0*l*n)
    
    r = (r1 + r2*exp)/(1 + r1*r2*exp)
    Ran_Lorentz = (abs(r))**2
    return Ran_Lorentz


def Tan_Lorentz(w, l, a):  #frequency, thin film thickness (unit of grid spacing)
    l = l*ddx       #same as in Ran_drude function 
    #analytical expression for the transmission coefficient
    eps_w = 1 + (f_0*w_0**2)/(w_0**2 - w**2 - 2j*w*a)
    n = np.sqrt(eps_w)
    r1 = (1 - n)/(1 + n)
    r2 = (n - 1)/(n + 1)
    k0 = w/c  
    exp1 = np.exp(1j*k0*l*n)
    exp2 = np.exp(2j*k0*l*n)

    t = ((1 + r1)*(1 + r2)*exp1)/(1 + r1*r2*exp2)
    Tan_Lorentz = (abs(t))**2
    return Tan_Lorentz


    

def plot_ampAndfft(Et_t,Er_t,Es_t,t,):
    niceFigure(True)
    fftres = 20
    R, T, Sum, omega = calc_fft(Et_t, Er_t, Es_t, fftres, dt)
    
    #set up the figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, figsize = ((11,8)),gridspec_kw={'height_ratios': [1,1]}) 

    #plot Ex amplitudes as a funciton of time on top
    ax1.plot(time, EtL, label = '$E_{t}$', color = 'r')
    ax1.plot(time, ErL, label = '$E_{r}$', color = 'b')
    ax1.plot(time, EinL, label = '$E_{in}$', color = 'g')
    ax1.set_xlabel('time $(\Delta t)$')
    ax1.set_ylabel('$E_x$ (arb. units)')
    # ax1.grid(True)
    # ax1.set_xticks([0,500,1000,1500,2000,2500,3000,3500,4000,4500])
    ax1.set_yticks([-0.5,0,0.5,1])
    ax1.legend(loc = 1)

    #numerical fft on the bottom subplot
    ax2.plot(omega*1e-12, R, label = 'R')
    ax2.plot(omega*1e-12, T, label = "T")
    ax2.plot(omega*1e-12, Sum, label = "R + T")
    ax2.set_xlim((100,300))
    ax2.set_ylim(0,1.2)
    ax2.set_yticks([0,0.2,0.4,0.6,0.8,1,1.2])
    # ax2.grid(True)
    ax2.set_xlabel('$\omega/2 \pi$ (THz)')
    ax2.set_ylabel('R, T (arb units)')
    ax2.legend(loc = 1)
    fig.tight_layout()
    
def plot_compareToAn(anid, freqstart,freqstop,Et_t, Er_t,Es_t,t,):
    niceFigure(True)
    fftres = 20
    R, T, Sum, omega = calc_fft(Et_t, Er_t, Es_t, fftres, dt)
    pts = 200
    freqs = np.linspace(freqstart,freqstop,pts)#analytical frequency range
    if anid == 'None':
        Ra = Ran(eps, freqs, thickness)   #analytical R
        Ta = Tan(eps, freqs, thickness)   #analytical T
    elif anid == 'Drude':
        Ra = Ran_drude(freqs, thickness, ab)
        Ta = Tan_drude(freqs, thickness, ab)
    elif anid == 'Lorentz':
        Ra = Ran_Lorentz(freqs, thickness, ac)
        Ta = Tan_Lorentz(freqs, thickness, ac)

    fig = plt.figure(figsize = (12,7))
    plt.plot(omega*1e-12, R,color = 'r', label = 'R')
    plt.plot(omega*1e-12, T, color = 'b', label = "T")
    plt.plot(omega*1e-12, R + T, color = 'magenta', linewidth = 2, label = "R + T")
    plt.plot(freqs*1e-12/(2*np.pi),Ra, '--',linewidth = 2, color = 'r',label = '$R_{an}$')
    plt.plot(freqs*1e-12/(2*np.pi),Ta, '--', linewidth = 2, color = 'b', label = '$T_{an}$')
    plt.plot(freqs*1e-12/(2*np.pi),Ra + Ta, label = '$R_{an} + T_{an}$')
    plt.xlim(100,300)
    plt.ylim(0,1.2)
    plt.yticks([0,0.2,0.4,0.6,0.8,1,1.2])
    # plt.grid(True)
    plt.xlabel('$\omega/2 \pi$ (THz)')
    plt.ylabel('R, T (arb units)')
    plt.legend(loc = 'best', fontsize = 25)
    fig.tight_layout()
    
def kw_f_mapper(modelchoice):
    """
    Choose which type of dispersion model use. L: for Lorentz, D: Drude, C: for constant \epsilon
    like in question one. 
    """
    if (modelchoice == 'L'):
        return 'Lorentz'
    elif (modelchoice == 'D'):
        return 'Drude'
    elif (modelchoice == 'C'):
        return 'None'



#useful constants
c = constants.c # speed of light in vacuum
fs = constants.femto # 1.e-15 - useful for pulses 
tera = constants.tera # 1.e12 - used for optical frequencues 

#%%

"QUESTION 1: ABC'S, TFSF AND IMPLEMENTATION OF A LINEAR DIELECTIC THIN FILM"

#start by clearing any earlier variables
clear_variables(True)

"Initialize the animation and parameters in this cell"

"For animation updates - will slow down the loop to see Ex frames better"
time_pause = 0.8
cycle = 100 # for graph updates
t_plot = [500,800,1200,2000]  #save snapshots at these timesteps

"create filename for saving"
graph_flag = 0 # 0 will not save graphics 
filename = "fdtd_1d.pdf" # save at final time slice (example)


"Basic geometry and dielectric parameters"
model = kw_f_mapper('C')   #decide which dispersion model to use
Xmax = 801  # no of FDTD cells in x
nsteps = 4000 # number of FDTD time steps
# nsteps = 1000 # number of FDTD time steps
ddx = 20.e-9 #  FDTD grid size in space, in SI Units
dt = ddx/(2.*c) # FDTD time step
isourceE = 200   # source position



#Add thin film with dielelectric constant \epsilon
filmstart, filmstop = 350, 400   #grid start and stop location for thin film
epsilon = 9    #dialectric constant 
cb, eps, thickness = dial_c(Xmax, filmstart, filmstop, epsilon, ddx) #call function for this simpler example



"Pulse parameters and points per wavelength"
spread = 2.* fs/dt # 2 fs for this example
X1 = int(Xmax/2) # center position
t0 = spread*6   
freq_in = 2*math.pi*200*tera # incident (angular) frequency
w_scale = freq_in*dt
lam = 5*math.pi*c/freq_in # near 1.5 microns
lam = lam/eps   #for wavelength decrease due to dialectric
ppw = int(lam/ddx) # will round down
print('points per wavelength',ppw, 'should be > 15')

# an array for spatial points (without first and last points)
xs = np.arange(1,Xmax-1)  
t = 0   # initial time


"Run animation and plot fft-related graphics"

#Empty lists for storing reflected and transmitted fields 
EinL = []
EtL = []
ErL = []
time = []
#E and H fields
Ex = np.zeros((Xmax),float) # E array  
Hy = np.zeros((Xmax),float) # H array  


" Main FDTD loop iterated iter_t times"
def FDTD_loop1(nsteps,cycle):

    #transmitted, reflected, and incident fields
    Et = np.zeros((nsteps + 1))
    Er = np.zeros(nsteps + 1)
    Ein = np.zeros(nsteps + 1)

    #initialize 'storage' variables for the ABC's
    Ebl, Ebr, Hyl, Hyr = 0,0,0,0


    # loop over all time steps
    for i in range (0,nsteps+1): # time loop, from 0 to nsteps  
       t=i-1 # iterative time dep pulse as source
       
       #initial E and H pulses
       Epulse = Epulse_init(t, t0, spread, w_scale)
       Hpulse = Hpulse_init(t, t0, spread, w_scale)
 
       
       #calculate the transmitted and reflected field amplitudes at eachlocation decided
       Et = Ex[450]   #calculate Et after the thin film
       Er = Ex[100]   #calculate Er before the source
       Ein[t] = Epulse


        #set absorbing bc - terminating the grid 
       Ex[0], Ex[-1] = Ebl, Ebr   #update once here and once below = 2timesteps
       Ebl, Ebr = Ex[1], Ex[-2]  
      
       # update E 
       Ex[1:-1] = Ex[1:-1] + cb[1:-1]*(Hy[0:-2] - Hy[1:-1])  #with dialectric medium
       # Ex[1:-1] = Ex[1:-1] + 0.5*(Hy[0:-2] - Hy[1:-1])  #without dialectric medium

       
       #initiate pulse with TFSF conditions
       Ex[isourceE] = Ex[isourceE] - Hpulse*0.5    #TFSF
       # Ex[isourceE] = Ex[isourceE] - Epulse*0.5   #choose this for full field from pulse

       
        #append transmission, reflection, and initial fields for fft later
       EtL.append(Et)    #Et(t) list
       ErL.append(Er)    #Er(t) list
       EinL.append(Ein[t])   #E_source(t) list
       time.append(t)

       #absorbing bc's for H
       Hy[0], Hy[-1] = Hyl, Hyr   #update once here and once below = 2timesteps
       Hyl, Hyr = Hy[1], Hy[-2]  

        # update H                   
       Hy[1:-1] = Hy[1:-1] + 0.5*(Ex[1:-1] - Ex[2:])  #with dialectric medium

       #initialize the H field using the E pulse
       Hy[isourceE-1] = Hy[isourceE-1] - 0.5*Epulse

        # update graph every cycle - very simple animation
       if (i % cycle == 0): 
           im.set_ydata(Ex[1:Xmax-1])
           ax.set_title("frame time {}".format(i))
           plt.show()
           plt.pause(time_pause) # sensible pause to watch animation 
           if i in t_plot:
               # plt.savefig('abc_tsteps(c)p1={}.png'.format(i), format = 'png', dpi = 200, bbox_inches = 'tight')
               plt.draw()


# initialize graph, fixed scaling for this first example
def init1(zoom):
    """
    zoom: Boolean. If True, zoom in to see backward injected field.
    """
    if zoom == True:    
        plt.ylim((-0.00001, 0.00001))   #note that our backward injected field has an amplitude of ~5e-6
        plt.xlim((20, 100))
    elif zoom == False:
        plt.ylim((-1, 1))
        plt.xlim((0, Xmax-1))  
    plt.axvline(x=X1,color='r') # Vert line separator
    plt.axvline(x=filmstart,color='r') # show the opposite side of the thin film on the grid
    ax.annotate('Thin Film', xy=(350,0.75),  xycoords='data',alpha=1,    
                xytext=(90, 0.87), textcoords='data',
                arrowprops=dict(facecolor='blue', shrink=0.01,headlength=5)
                ,fontsize = 25)
    plt.grid('on')
    ax.set_xlabel('Grid Cells ($z$)')
    ax.set_ylabel('$E_x$')
    plt.show()




"Define first (only in this simple example) graph for updating Ex at various times"
lw=2 # line thickness for graphs
niceFigure(True)
fig = plt.figure(figsize=(10,6))
ax = fig.add_axes([.18, .18, .7, .7])
[im] = ax.plot(xs,Ex[1:Xmax-1],linewidth=lw)   #initialize the line 
init1(False) # initialize, then we will just update the y data and title frame

"Main FDTD: time steps = nsteps, cycle for very simple animation"
FDTD_loop1(nsteps,cycle)

"Save last slice"
if graph_flag == 1:
    plt.savefig(filename, format='pdf', dpi=1200,bbox_inches = 'tight')

#close the animation when its done
plt.close("all")


"Numerical fft and Ex amplitude graphics"
plot_ampAndfft(EtL, ErL, EinL, time)  
# plt.savefig("Q1numFFT.png", format='png', dpi=200,bbox_inches = 'tight')
plt.pause(7)    #keep plot open for 7 seconds
plt.close('all')  #then close


"graphics for comparison between numerical and analytical fft"
plot_compareToAn(model, 100*tera*2*np.pi, 300*tera*2*np.pi, EtL, ErL, EinL, time)
# plt.savefig("Q1analFFT.png", format='png', dpi=200,bbox_inches = 'tight')
plt.pause(7)
plt.close('all')


#%%
"QUESTION 2: IMPLIMENTATION OF Dx AND LOSSY MEDIA"

#start by clearing earlier variables
clear_variables(True)

"Initialize the animation and parameters in this cell"

"For animation updates - will slow down the loop to see Ex frames better"
time_pause = 0.8
cycle = 100 # for graph updates
t_plot = [0,400,800,1200]  #save snapshots at these timesteps

"create filename for saving"
graph_flag = 0 # 0 will not save graphics 
filename = "fdtd_1d.pdf" # save at final time slice (example)


"Basic geometry and dielectric parameters"
model = kw_f_mapper('D')   #decide which dispersion model to use
Xmax = 801  # no of FDTD cells in x
nsteps = 2000 # number of FDTD time steps, put to ~7000 to get the best result for lorentz model. 2000 \
    #is good for Drude
# nsteps = 4000 # number of FDTD time steps
ddx = 20.e-9 #  FDTD grid size in space, in SI Units
dt = ddx/(2.*c) # FDTD time step
isourceE = 200   # source position
eps = 9


#for thin film thickness and parameters
thickness = 10  #desired film thickness in units of the grid spacing (ie. 10 = 200 nm)
ab = 1.4e14 #decay rate (rads/s)  for part (b)
ac = 4*np.pi*10**12  #decay rate for part (c)
w_p = 1.26e15  #plasma frequency (rads/s)
w_0 = 2*np.pi*200*10**12  #resonant oscillator frequency (for part (c))
f_0 = 0.05  #oscillator strength

"Pulse parameters and points per wavelength"
spread = 1.* fs/dt # 1 fs for question 2(b/c)
X1 = int(Xmax/2) # center position
t0 = spread*6   
freq_in = 2*math.pi*200*tera # incident (angular) frequency
w_scale = freq_in*dt
lam = 5*math.pi*c/freq_in # near 1.5 microns
lam = lam/eps   #for wavelength decrease due to dialectric
ppw = int(lam/ddx) # will round down
print('points per wavelength',ppw, 'should be > 15')

# an array for spatial points (without first and last points)
xs = np.arange(1,Xmax-1)  
t = 0   # initial time



"Run animation and plot fft-related graphics"

#Empty lists for storing reflected and transmitted fields 
EinL = []
EtL = []
ErL = []
time = []
#E and H fields
Ex = np.zeros((Xmax),float) # E array  
Hy = np.zeros((Xmax),float) # H array  
Dx = np.zeros((Xmax),float)  #D array
S = np.zeros((Xmax),float)  #S array's for 0, 1, and 2 time steps back
S1 = np.zeros((Xmax),float)  
S2 = np.zeros((Xmax),float)



" Main FDTD loop iterated iter_t times"
def FDTD_loop2(modelid, nsteps,cycle, thick):
    L = X1 + thick
    ga = np.zeros(Xmax)
    ga[X1:L] = 1   #start the thin film at the center of the grid always

    #transmitted, reflected, and incident fields
    Et = np.zeros((nsteps + 1))
    Er = np.zeros(nsteps + 1)
    Ein = np.zeros(nsteps + 1)

    #initialize 'storage' variables for the ABC's
    Ebl, Ebr, Hyl, Hyr = 0,0,0,0


    # loop over all time steps
    for i in range (0,nsteps+1): # time loop, from 0 to nsteps  
       t=i-1 # iterative time dep pulse as source
       
       #initial E and H pulses
       Epulse = Epulse_init(t, t0, spread, w_scale)
       Hpulse = Hpulse_init(t, t0, spread, w_scale)
 
       
       #calculate the transmitted and reflected field amplitudes at eachlocation decided
       Et = Ex[L + 50]   # be sure to calculate Et after the thin film
       Er = Ex[isourceE - 100]   # and calculate Er before the source
       Ein[t] = Epulse


        #set absorbing bc - terminating the grid 
       Ex[0], Ex[-1] = Ebl, Ebr   #update once here and once below = 2timesteps
       Ebl, Ebr = Ex[1], Ex[-2]  
           
       #update Dx (instead of Ex)
       Dx[1:-1] = Dx[1:-1] + 0.5*(Hy[0:-2] - Hy[1:-1])  

       #initiate Dx with Hpulse - TFSF conditions
       Dx[isourceE] = Dx[isourceE] - Hpulse*0.5    #TFSF
      
       #now update E from D with opiton of S for the lossy media
       Ex[1:-1] = Dx[1:-1] - ga[1:-1]*S[1:-1]
       
       #in sampled time-domain, frequency-dep. media is created by adding this line
       
       #depending on users choice of dispersion model
       if modelid == 'Drude':   #use a drude model
           S[1:-1] = (1 + np.exp(-ab*dt))*S1[1:-1] - np.exp(-ab*dt)*S2[1:-1] + \
            (dt*(w_p**2)/ab) * (1 - np.exp(-ab*dt))*Ex[1:-1]
       elif modelid == 'Lorentz':   #use a lorentz model
           b = w_0
           S[1:-1] = 2*np.exp(-ac*dt)*np.cos(b*dt)*S1[1:-1] - np.exp(-2*ac*dt)*S2[1:-1] + \
               np.exp(-ac*dt)*np.sin(b*dt)*dt*w_0*f_0*Ex[1:-1]
           
        #need to store values for the S arrays one and two time steps back   
       S2[1:-1] = S1[1:-1]
       S1[1:-1] = S[1:-1]
       
        
       #append transmission, reflection, and initial fields for fft later
       EtL.append(Et)    #Et(t) list
       ErL.append(Er)    #Er(t) list
       EinL.append(Ein[t])   #E_source(t) list
       time.append(t)

       #absorbing bc's for H
       Hy[0], Hy[-1] = Hyl, Hyr   #update once here and once below = 2timesteps
       Hyl, Hyr = Hy[1], Hy[-2]  

        # update H                   
       Hy[1:-1] = Hy[1:-1] + 0.5*(Ex[1:-1] - Ex[2:])  

       #initialize the H field using the E pulse
       Hy[isourceE-1] = Hy[isourceE-1] - 0.5*Epulse

        # update graph every cycle - very simple animation
       if (i % cycle == 0): 
           im.set_ydata(Ex[1:Xmax-1])
           ax.set_title("frame time {}".format(i))
           plt.show()
           plt.pause(time_pause) # sensible pause to watch animation 
           if i in t_plot:
               # plt.savefig('Q2c_l=800={}.png'.format(i), format = 'png', dpi = 200, bbox_inches = 'tight')
               plt.draw()


# initialize graph, fixed scaling for this first example
def init1(zoom):
    """
    zoom: Boolean. If True, zoom in to see backward injected field.
    """
    if zoom == True:    
        plt.ylim((-0.00001, 0.00001))   #note that our backward injected field has an amplitude of ~5e-6
        plt.xlim((20, 100))
    elif zoom == False:
        plt.ylim((-1, 1))
        plt.xlim((0, Xmax-1))  
    plt.axvline(x=X1,color='r') # Vert line separator
    plt.axvline(x=X1 + thickness,color='r') # Vert line separator
    ax.annotate('Thin Film', xy=(X1,0.75),  xycoords='data',alpha=1,
                xytext=(90, 0.87), textcoords='data',
                arrowprops=dict(facecolor='blue', shrink=0.01,headlength=5),fontsize = 26
                )
    plt.grid('on')
    ax.set_xlabel('Grid Cells ($z$)')
    ax.set_ylabel('$E_x$')
    plt.show()




"Define first (only in this simple example) graph for updating Ex at various times"
lw=2 # line thickness for graphs
niceFigure(True)
fig = plt.figure(figsize=(10,6))
ax = fig.add_axes([.18, .18, .7, .7])
[im] = ax.plot(xs,Ex[1:Xmax-1],linewidth=lw)   #initialize the line 
init1(False) # initialize, then we will just update the y data and title frame

"Main FDTD: time steps = nsteps, cycle for very simple animation"
FDTD_loop2(model, nsteps,cycle, thickness)

"Save last slice"
if graph_flag == 1:
    plt.savefig(filename, format='pdf', dpi=1200,bbox_inches = 'tight')

#close the animation when its done
plt.close("all")



"NUMERICAL AND ANALYTICAL FFT SOLVING AND GRAPHICS"

"Numerical fft and Ex amplitude graphics"
plot_ampAndfft(EtL, ErL, EinL, time)
plt.savefig('Q2b_l=200p1={}.png', format = 'png', dpi = 200, bbox_inches = 'tight')

plt.pause(7)    #keep plot open for 7 seconds
plt.close('all')  #then close

"graphics for comparison between numerical and analytical fft"
plot_compareToAn(model, 100*tera*2*np.pi, 300*tera*2*np.pi, EtL, ErL, EinL, time)
plt.savefig('Q2b_l=200p1b={}.png', format = 'png', dpi = 200, bbox_inches = 'tight')
plt.pause(7)
plt.close('all')








#%%

"QUESTION 3: 2D FDTD FOR TM MODES AND SPEED UP TECHNIQUES"
#full set-up plus animation with lines left in for quick comparison betwen base code and sped up versions



clear_variables(True)


"For animation updates - will slow down the loop to see Ex frames better"
time_pause = 0.01
t_plot1 = [80, 240, 480, 800, 960]   #for saving the plot during animation
# t_plot2 = [40, 120, 240, 440, 680, 820, 920, 960]   #for plots to save if running faster animation with smaller 'cycle' value
"Quick and dirty graph to save"
filename = "fdtd_2d_example1.pdf" # save at final time slice (example)

"Basic Geometry and Dielectric Parameters"
printgraph_flag = 0 # print graph to pdf (1)
livegraph_flag = 1 # update graphs on screen every cycle (1)
Xmax = 160  # no of FDTD cells in x
Ymax = 160  # no of FDTD cells in y
nsteps = 1000 # total number of FDTD time steps
cycle = 40 # for graph updates
Ex = np.zeros((Xmax),float) # E array  
Hy = np.zeros((Xmax),float) # H array  

"2d Arrays"
Ez = np.zeros([Xmax,Ymax],float); 
Hx = np.zeros([Xmax,Ymax],float); Hy = np.zeros([Xmax,Ymax],float) 
Dz = np.zeros([Xmax,Ymax],float); ga=np.ones([Xmax,Ymax],float)
cb = np.zeros([Xmax,Ymax],float) # for spatially varying dielectric constant
EzMonTime1=[]; PulseMonTime=[] # two time-dependent field monitors

ddx = 20.e-9 #  FDTD grid size in space, in SI Units
dt = ddx/(2.*c) # FDTD time step

# dipole source position, atr center just now
isource = int(Ymax/2)
jsource = int(Xmax/2)

"Pulse parameters and points per wavelength"
spread=1.* fs/dt # 2 fs for this example
t0=spread*6
freq_in = 2*np.pi*200*tera # incident (angular) frequency
w_scale = freq_in*dt
lam = 2*np.pi*c/freq_in # near 1.5 microns
eps2 = 1 # dielectric box (so 1 is just free space)
ppw = int(lam/ddx/eps2**0.5) #  rounded down
print('points per wavelength:',ppw, '(should be > 15)')

# simple fixed dielectric box coordinates
X1=isource+10; X2=X1+40
Y1=jsource+10; Y2=Y1+40
for i in range (0,Xmax):
    for j in range (0,Ymax): 
        if i>X1 and i<X2+1 and j>Y1 and j<Y2+1:   
            ga[i,j] = 1./eps2

# ga[X1:X2 + 1, Y1:Y2 + 1] = 1./eps2
# an array for x,y spatial points (with first and last points)
xs = np.arange(0,Xmax)  
ys = np.arange(0,Ymax)  





# "Update def's for main loop (seperate from main loop for use with numba)"
@numba.jit(nopython = True)   #use numba 
def UpdateE(Dz, Ez, Hy, Hx, ga):   
    for x in range (1,Xmax-1): 
        for y in range (1,Ymax-1):
            Dz[x,y] =  Dz[x,y] + 0.5*(Hy[x,y]-Hy[x-1,y]-Hx[x,y]+Hx[x,y-1])    #curl equation for D
            Ez[x,y] =  ga[x,y]*(Dz[x,y])       #obtain Ez from D
    # for vectorized upddates
    # Dz[1:-1,1:-1] = Dz[1:-1,1:-1] + 0.5*(Hy[1:-1,1:-1] - Hy[0:-2,1:-1] - Hx[1:-1,1:-1] + Hx[1:-1,0:-2])
    # Ez[1:-1,1:-1] = ga[1:-1,1:-1]*(Dz[1:-1,1:-1])
    return Dz, Ez, Hy, Hx, ga

@numba.jit(nopython = True)
def UpdateH(Hx, Hy, Ez):
    for x in range (0,Ymax-1): 
        for y in range (0,Xmax-1): 
            Hx[x,y] = Hx[x,y]+ 0.5*(Ez[x,y]-Ez[x,y+1])                       
            Hy[x,y] = Hy[x,y]+ 0.5*(Ez[x+1,y]-Ez[x,y])        
    #vectorized updates
    # Hx[0:-1,0:-1] = Hx[0:-1,0:-1] + 0.5*(Ez[0:-1,0:-1] - Ez[0:-1,1:])
    # Hy[0:-1,0:-1] = Hy[0:-1,0:-1] + 0.5*(Ez[1:,0:-1] - Ez[0:-1,0:-1])
    return Hx, Hy, Ez



" Main FDTD loop iterated over nsteps"
def FDTD_loop3(nsteps,cycle):
    # loop over all time steps
    for t in range (0,nsteps):
        #call pulse function (this is the same as the Epulse in 1D)
        # pulse = Epulse_init(t, t0, spread, w_scale)
        pulse = np.exp(-0.5*(t-t0)**2/spread**2)*(np.cos(t*freq_in*dt))


        "update D and obtain E" #3 approaches here; comment out whatever not using
        # calculate Dz (Hy is diff sign to before with Dz term from curl eqs)        
        #slowest: no numba or slicing
        # for y in range (1,Ymax-1):
            # for x in range (1,Xmax-1): 
                # Dz[x,y] =  Dz[x,y] + 0.5*(Hy[x,y]-Hy[x-1,y]-Hx[x,y]+Hx[x,y-1]) 
                # Ez[x,y] =  ga[x,y]*(Dz[x,y])
        #vectorized approach
        # Dz[1:-1,1:-1] = Dz[1:-1,1:-1] + 0.5*(Hy[1:-1,1:-1] - Hy[0:-2,1:-1] - Hx[1:-1,1:-1] + Hx[1:-1,0:-2])
        # Ez[1:-1,1:-1] = ga[1:-1,1:-1]*(Dz[1:-1,1:-1])
        #use numba
        UpdateE(Dz, Ez, Hy, Hx, ga)  

        
        #begin updating Dz and Ez
        Dz[isource,jsource] =  Dz[isource,jsource] + pulse # soft source in simulation center
        Ez[isource,jsource] =  ga[isource,jsource]*(Dz[isource,jsource])
        # save one point in time just to see the transient
        EzMonTime1.append(Ez[isource,jsource]) 
        PulseMonTime.append(pulse) 


        "Update Hx,y"   #3 approaches; comment out the lines for the aproach not using
        #slowest: no numba or slicing
        # for y in range (0,Xmax-1): 
            # for x in range (0,Ymax-1): 
                # Hx[x,y] = Hx[x,y]+ 0.5*(Ez[x,y]-Ez[x,y+1])                       
                # Hy[x,y] = Hy[x,y]+ 0.5*(Ez[x+1,y]-Ez[x,y])                 
        #vectorized approach
        # Hx[0:-1,0:-1] = Hx[0:-1,0:-1] + 0.5*(Ez[0:-1,0:-1] - Ez[0:-1,1:])
        # Hy[0:-1,0:-1] = Hy[0:-1,0:-1] + 0.5*(Ez[1:,0:-1] - Ez[0:-1,0:-1])
        #use numba
        UpdateH(Hx, Hy, Ez)      


                          
        # update graph every cycle 
        if (t % cycle == 0 and livegraph_flag == 1): # simple animation
            graph(t)
            if t in t_plot1:
                plt.savefig('q3_160_eps = {},time = {}.png'.format(eps2,t), format = 'png', dpi = 300, bbox_inches = 'tight')
                plt.draw()


def graph(t):

    # main graph is E(z,y, time snapshops), and a small graph of E(t) as center
    plt.clf() # close each time for new update graph/colormap
    ax = fig.add_axes([.25, .25, .6, .6])   
    ax2 = fig.add_axes([.015, .8, .15, .15])   

    # 2d plot - several options, two examples below
    #    img = ax.imshow(Ez)
    img = ax.contourf(Ez)
    cbar=plt.colorbar(img, ax=ax)
    cbar.set_label('$Ez$ (arb. units)')

    # add labels to axes
    ax.set_xlabel('Grid Cells ($x$)')
    ax.set_ylabel('Grid Cells ($y$)')
     
    # dielectric box - comment if not using of course (if eps2=1)
    # ax.vlines(X1,Y1,Y2,colors='r')
    # ax.vlines(X2,Y1,Y2,colors='r')
    # ax.hlines(Y1,X1,X2,colors='r')
    # ax.hlines(Y2,X1,X2,colors='r')

    # add title with current simulation time step
    ax.set_title("frame time {}".format(t))
    # plt.show()

    # Small graph to see time development as a single point
    PulseNorm = np.asarray(PulseMonTime)*0.2;
    ax2.plot(PulseNorm,'r',linewidth=1.6)
    ax2.plot(EzMonTime1,'b',linewidth=1.6)
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    ax2.set_title('$E_{(x,y)=0}(t),E_{s} (t)$')
    # plt.show()
    plt.pause(time_pause) # pause sensible value to watch what is happening


# set figure for graphics output
if livegraph_flag==1: 
    niceFigure(True)
    fig = plt.figure(figsize=(8,6))

"Main FDTD: time steps = nsteps, cycle for very simple animation"
start = timeit.default_timer()     #benchmark the main routine 
FDTD_loop3(nsteps,cycle)
stop = timeit.default_timer()
print ("Time for FDTD simulation", stop - start)

"Save Last Slice - adjust as you like"
if printgraph_flag == 1:
    plt.savefig(filename, format='pdf', dpi=500,bbox_inches = 'tight')

plt.close('all')












