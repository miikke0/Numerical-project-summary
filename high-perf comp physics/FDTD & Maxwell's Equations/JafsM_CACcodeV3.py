# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 21:26:16 2022

@author: mjafs
"""

"Question 3 (D) - Parallization of the fdtd 2d routine for use with CAC"

import numpy as np
from matplotlib import pyplot as plt
import timeit
#from matplotlib import animation # not using, but you can also use this
from mpi4py import MPI

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


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()



t_plot = [150, 450, 750, 950]  #save a high quality snap shot at these time-frames


#useful constants
c = 3e8 # speed of light in vacuum
fs = 1e-15 # 1.e-15 - useful for pulses 
tera = 1e12 # 1.e12 - used for optical frequencues 

#Basic Geometry and Dielectric Parameters"
Xmax = 504  # no of FDTD cells in x
Ymax = 504  # no of FDTD cells in y
nsteps = 1000 # total number of FDTD time steps


chunk_y = int(Ymax/size)   #only divide up the grid in the y-direction

#2d Arrays  of size, chunk by Xmax
Ez = np.zeros([chunk_y + 2,Ymax],float); 
Hx = np.zeros([chunk_y + 2,Ymax],float); Hy = np.zeros([chunk_y + 2,Ymax],float) 
Dz = np.zeros([chunk_y + 2,Ymax],float); ga=np.ones([chunk_y + 2,Ymax],float)
EzMonTime1=[]; PulseMonTime=[] # two time-dependent field monitors

ddx = 20.e-9 #  FDTD grid size in space, in SI Units
dt = ddx/(2.*c) # FDTD time step

# dipole source position, at center just now
jsource = int(Xmax/2)
isource = int(Ymax/2)




#Pulse parameters and points per wavelength
spread=1.* fs/dt # 2 fs for this example
t0=spread*6
freq_in = 2*np.pi*200*tera # incident (angular) frequency
w_scale = freq_in*dt
lam = 2*np.pi*c/freq_in # near 1.5 microns
eps2 = 9 # dielectric box (so 1 is just free space)
ppw = int(lam/ddx/eps2**0.5) #  rounded down
print('points per wavelength:',ppw, '(should be > 15)')



j_start = rank*chunk_y    #tell each process where to begin calculations
if rank == 0:
    j_start = rank*chunk_y + 1# implicit boundary condition

j_stop = j_start + chunk_y    #where should each process end its calculations
if rank == size - 1:
    j_stop = j_start + chunk_y - 1 # implicit boundary condition




# simple fixed dielectric box coordinates
X1=isource+10; X2=X1+40
Y1=jsource+10; Y2=Y1+40
for i in range (0,Xmax):
    for j in range (j_start,j_stop): 
        if i>X1 and i<X2+1 and j>Y1 and j<Y2+1:   
            ga[j - j_start,i] = 1./eps2
                
# an array for x,y spatial points (with first and last points)
xs = np.arange(0,Xmax)  
ys = np.arange(0,Ymax)  


# "Update def's for main loop (seperate from main loop for use with numba)"
# @numba.jit(nopython = True)   #use numba 
def UpdateE(D, E, Hyy, Hxx, gaa):   
    # for x in range (1,Xmax-1): 
        # for y in range (1,Ymax-1):
            # Dz[x,y] =  Dz[x,y] + 0.5*(Hy[x,y]-Hy[x-1,y]-Hx[x,y]+Hx[x,y-1])    #curl equation for D
            # Ez[x,y] =  ga[x,y]*(Dz[x,y])       #obtain Ez from D
    # for vectorized upddates
    D[1:-1,1:-1] = D[1:-1,1:-1] + 0.5*(Hyy[1:-1,1:-1] - Hyy[:-2,1:-1] - Hxx[1:-1,1:-1] + Hxx[1:-1,:-2])
    E[1:-1,1:-1] = gaa[1:-1,1:-1]*(D[1:-1,1:-1])
    # Dz[:-1,1:-1] = Dz[1:-1,1:-1] + 0.5*(Hy[1:-1,1:-1] - Hy[0:-2,1:-1] - Hx[1:-1,1:-1] + Hx[1:-1,0:-2])
    # Ez[1:-1,1:-1] = ga[1:-1,1:-1]*(Dz[1:-1,1:-1])
    return D, E, Hyy, Hxx, gaa


# @numba.jit(nopython = True)
def UpdateH(Hxx, Hyy, E):
    # for x in range (0,Ymax-1): 
        # for y in range (0,Xmax-1): 
            # Hx[x,y] = Hx[x,y]+ 0.5*(Ez[x,y]-Ez[x,y+1])                       
            # Hy[x,y] = Hy[x,y]+ 0.5*(Ez[x+1,y]-Ez[x,y])        
    #vectorized updates
    Hxx[:-1,:-1] = Hxx[:-1,:-1] + 0.5*(E[:-1,:-1] - E[:-1,1:])
    Hyy[:-1,:-1] = Hyy[:-1,:-1] + 0.5*(E[1:,:-1] - E[:-1,:-1])
    return Hxx, Hyy, E




#Main FDTD loop iterated over nsteps
def FDTD_loop_par(nsteps):
    # loop over all time steps
    for t in range (0,nsteps):
        


        #communication (we are creating a "buffer" array above and below each grid segment of main array)
        #only need to communicate the arrays that chanhge in the y-axis
        if rank != 0:
            comm.Send(Ez[1], dest=rank-1, tag=11)   #send second row to segment above
            comm.Recv(Ez[0], source=rank-1, tag=12)  #recieve array and place in first row position
            comm.Send(Hy[1], dest=rank-1, tag=5)   #send second row to segment above
            comm.Recv(Hy[0], source=rank-1, tag=6)  #recieve array and place in first row position
            
        # calculate Dz (Hy is diff sign to before with Dz term from curl eqs)        
        # UpdateE(Dz, Ez, Hy, Hx, ga)  #update Dz and Ez and use slicing

        if rank != size-1:
            comm.Recv(Ez[-1], source=rank+1, tag=11)   #place array in last position
            comm.Send(Ez[-2], dest=rank+1, tag=12)    #send second last row to segment below
            comm.Recv(Hy[-1], source=rank+1, tag=5)   #place array in last position
            comm.Send(Hy[-2], dest=rank+1, tag=6)    #send second last row to segment below

        #call pulse function (this is the same as the Epulse in 1D)
        pulse = np.exp(-0.5*(t-t0)**2/spread**2)*(np.cos(t*freq_in*dt))

        UpdateE(Dz, Ez, Hy, Hx, ga)  #update Dz and Ez and use slicing
            
         
        if rank == int(size/2 - 1):   
            isource = int(chunk_y/2)      #assign the source to be just above the middle of the grid
            Dz[isource,jsource] =  Dz[isource,jsource] + pulse # soft source in simulation center
            Ez[isource,jsource] =  ga[isource,jsource]*(Dz[isource,jsource])


         # save one point in time just to see the transient
        # PulseMonTime.append(pulse)   #leaving out for now
        
        UpdateH(Hx, Hy, Ez) #update H with slicing
        
                          
        # update graph every cycle 
        if t in t_plot:
            full_U = None   #our full grid
            partialgrid = Ez[1:-1]     #get rid of the buffer arrays
            
            if rank == 0:
                full_U = np.zeros((Ymax,Xmax))
                
            comm.Gather(partialgrid,full_U,root=0)
            if rank == 0:
                niceFigure(True)
                fig = plt.figure()
                # main graph is E(z,y, time snapshops), and a small graph of E(t) as center
                ax = fig.add_axes([.25, .25, .6, .6])   
                # ax2 = fig.add_axes([.012, .8, .15, .15])   

                # 2d plot - several options, two examples below
                #    img = ax.imshow(Ez)
                img = ax.contourf(full_U, cmap = 'viridis')
                cbar=plt.colorbar(img, ax=ax)
                cbar.set_label('$Ez$ (arb. units)')

                # add labels to axes
                ax.set_xlabel('Grid Cells ($x$)')
                ax.set_ylabel('Grid Cells ($y$)')
                # ax.set_xticks([0, 50, 100, 150])
                
                # dielectric box - comment if not using of course (if eps2=1)
                ax.vlines(X1,Y1,Y2,colors='r')
                ax.vlines(X2,Y1,Y2,colors='r')
                ax.hlines(Y1,X1,X2,colors='r')
                ax.hlines(Y2,X1,X2,colors='r')

                # add title with current simulation time step
                ax.set_title("frame time {}".format(t))
                # plt.show()
                
                #difficult to get this working with CAC, so we comment out for now
                # # Small graph to see time development as a single point
                # PulseNorm = np.asarray(PulseMonTime)*0.2;
                # # ax2.plot(PulseNorm,'r',linewidth=1.6)
                # # ax2.plot(EzMonTime1,'b',linewidth=1.6)
                # ax2.set_yticklabels([])
                # ax2.set_xticklabels([])
                # ax2.set_title('$E_{(x,y)=0}(t), E_{s}$')
                # # plt.show()
                plt.savefig('fdtd2d_code_Par_time = {}.png'.format(t), format = 'png', dpi = 300, bbox_inches = 'tight')
                plt.clf()            



start = timeit.default_timer()     #benchmark the main routine 
FDTD_loop_par(nsteps)   #call the loop (perform the calculation)
stop = timeit.default_timer()
print ("Time for FDTD simulation", stop - start)
