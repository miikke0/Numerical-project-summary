# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 17:51:53 2022

@author: mjafs
"""

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI 
import timeit



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


#SETUP INITIAL PARAMETERS:
# plate size, mm
w = h = 20.48
#print("plate size:",w,"x",h,"mm")

# grid size
nx = ny = int(2**10)    #(1024x1024)
#print("grid size:",nx,"x",ny)


# intervals in x-, y- directions, mm
dx, dy = w/nx, h/ny
dx2, dy2 = dx*dx, dy*dy

# Thermal diffusivity of steel, mm2/s
D = 4.2
#print("thermal diffusivity:",D)

# time
nsteps = 1001
dt = dx2 * dy2 / (2 * D * (dx2 + dy2))
#print("dt:",dt)

# time steps for plotting 
plot_ts = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

#circle params
Tcool, Thot = 300, 2000
r, cx, cy = 5.12, w/2, h/2   #radius, circumference (x & y)
r2 = r**2

#parallel params
chunk_y = int(ny/size)   #only divide up the grid in the y-direction

#define starting array (size that each process will work on)
u0 = np.zeros((chunk_y + 2, nx)) # (rows, columns)
u = u0.copy()



#vectorized 2d finite difference approximation routine for the diffusion equation 
def diffusion_2d(u0, u):
# Propagate with forward-difference in time, central-difference in space
    u[1:-1, 1:-1] = u0[1:-1, 1:-1] + D * dt * (
        (u0[2:, 1:-1] - 2*u0[1:-1, 1:-1] + u0[:-2, 1:-1])/dy2
        + (u0[1:-1, 2:] - 2*u0[1:-1, 1:-1] + u0[1:-1, :-2])/dx2 )

    u0 = u.copy()
    return u0, u



j_start = rank*chunk_y    #tell each process where to begin calculations
if rank == 0:
    j_start = rank*chunk_y + 1 # implicit boundary condition

j_stop = j_start + chunk_y    #where should each process end its calculations
if rank == size - 1:
    j_stop = j_start + chunk_y - 1 # implicit boundary condition



# Parallelized initialization - circle of radius r centred at (cx,cy) (mm)
for j in range(j_start,j_stop):
    for i in range(nx):
        p2 = (i*dx-cx)**2 + (j*dy-cy)**2   #only calculate inside of each grid of size chunk_y
        if p2 < r2:
            radius = np.sqrt(p2)
            u0[j-j_start + 1,i] = Thot*np.cos(4*radius)**4   #account for the "buffer" above each grid segment
        else:
            u0[j-j_start + 1,i] = Tcool
        
            
#set boundary conditions
u0[0,:], u0[-1,:], u0[:,0], u0[:,-1] = Tcool, Tcool, Tcool, Tcool           


#setup timer
if rank == 0:
    start = timeit.default_timer()
    

#MAIN CALCULATION
for t in range(nsteps):
    
    
    #communication (we are creating a "buffer" array above and below each grid segment of main array)
    if rank != 0:
        comm.Send(u0[1], dest=rank-1, tag=11)   #send second row to segment above
        comm.Recv(u0[0], source=rank-1, tag=12)  #recieve array and place in first row position
    if rank != size-1:
        comm.Recv(u0[-1], source=rank+1, tag=11)   #place array in last position
        comm.Send(u0[-2], dest=rank+1, tag=12)    #send second last row to segment below
        
    
    #step forward in time                            
    u0, u = diffusion_2d(u0, u)
    
    #stop the timer
if rank == 0:
    stop = timeit.default_timer()
    print("Time for solver:", stop - start)  #print the time to solve (calculation only)
        
        
    if t in plot_ts:   #plot at pre-determined time steps
        
        full_U = None   #our full grid
        partialgrid = u[1:-1]     #get rid of the buffer arrays
        if rank == 0:
            full_U = np.zeros((ny,nx))
        comm.Gather(partialgrid,full_U,root=0)
 
        if rank == 0:
            niceFigure(True)
            fig = plt.figure(1, figsize = (10,8))
            im = plt.imshow(full_U, cmap=plt.get_cmap('hot'), vmin=Tcool,vmax=Thot)
            plt.title('Time elapsed: {:.1f} ms'.format((t+1)*dt*1000))
            # plt.axis([0.01,0.4,0.2,0.9])
            plt.xticks([0,300,600,900])
            plt.yticks([0,300,600,900])
            plt.xlabel("Plate Width (mm)")
            plt.ylabel("Plate Height (mm)")
            cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
            cbar_ax.set_xlabel('T (K)', labelpad=20)
            fig.colorbar(im, cax=cbar_ax)
            plt.savefig("iter_{}.png".format(t), dpi=200, bbox_inches = 'tight')
            plt.clf()

