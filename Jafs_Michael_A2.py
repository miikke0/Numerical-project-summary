# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 16:15:45 2022

@author: mjafs
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton
from scipy.integrate import odeint


#%%
"Derivatives"

"EOM's for use with 'leap frog method"
def oscillator(id , x, v, t): # return dx/dt or dv/dt
    if (id==0): # calc velocity
        return v
    else: # calc acceleration
        return -x # $-\ omega0^2 x, \omega0=1$
    
def oscillatorE(x, v):    #oscillator energy
    return 0.5*x**2 + 0.5*v**2   #total energy of the system in natural units (m = 1)
    
"For use with RK4 solver"
def derivs(t,y): # my derivatives function
    dy=np.zeros((len(y)))
    #dy = [0] * len(y) - stick with numpy arrays!
    dy[0] = y[1]
    dy[1] = -y[0]
    return dy

"Derivatives for three body EOM's for use with the Leapfrog method"
def threeb_LF_derivs(id, r, v, t):
    dr, dv = np.zeros((3,2)), np.zeros((3,2))
    
    r12, r23, r31 = r[0] - r[1], r[1] - r[2], r[2] - r[0]
    s12, s23, s31 = np.sqrt(r12.dot(r12)), np.sqrt(r23.dot(r23)), np.sqrt(r31.dot(r31))
    
    if (id==0):
        dr[0] = v[0]  #dr1x/dt = dv1x
        dr[1] = v[1]
        dr[2] = v[2]
        return dr
    else:
        dv[0] = -m2*(r12/(s12**3)) - m3*((-r31)/(s31**3))
        dv[1] = -m3*(r23/(s23**3)) - m1*((-r12)/(s12**3))
        dv[2] = -m1*(r31/(s31**3)) - m2*((-r23)/(s23**3))
        return dv
    
"Three Body EOM's for RK4"
def threeb_RK4_derivs(t, y): # my derivatives function
    ri, vi = y[:6], y[6:]
    r, v = np.reshape(ri,(3,2)), np.reshape(vi, (3,2))
    dr, dv = np.zeros((3,2)), np.zeros((3,2))
        
    r12, r23, r31 = r[0] - r[1], r[1] - r[2], r[2] - r[0]
    s12, s23, s31 = np.sqrt(r12.dot(r12)), np.sqrt(r23.dot(r23)), np.sqrt(r31.dot(r31))
    
    #mass one x and y
    dr[0] = v[0]  #dr1x/dt = dv1x
    dv[0] = -m2*(r12/(s12**3)) - m3*((-r31)/(s31**3))
    
    #mass two x and y
    dr[1] = v[1]
    dv[1] = -m3*(r23/(s23**3)) - m1*((-r12)/(s12**3))
    
    #mass three x and y
    dr[2] = v[2]
    dv[2] = -m1*(r31/(s31**3)) - m2*((-r23)/(s23**3))
    diff = ([dr, dv])
    dy = np.reshape(diff,(12))
    return dy

"Total Energy for three bodies"
def threeb_Energy(r,v): # KE + PE, in units of G=1
    r12 , r13 , r23 = r[0]-r[1], r[0]-r[2], r[1]-r[2]
    s12v, s13v, s23v = np.array(r12), np.array(r13), np.array(r23)
    s12 , s13 , s23 = np.sqrt(s12v.dot(s12v)), np.sqrt(s13v.dot(s13v)), np.sqrt(s23v.dot(s23v))
    Energy=0.5*(m1*v[0].dot(v[0])+m2*v[1].dot(v[1])+m3*v[2].dot(v[2])) - m1*m2/s12 - m1*m3/s13- m2*m3/s23
    return Energy




    
#%%
"ODE ALGORITHMS"

"Leap frog algorithm"
def leapfrog(diffeq , r0 , v0 , t, h): # vectorized leapfrog
    hh = h/2.0
    r1 = r0 + hh*diffeq(0, r0, v0, t) # 1: r1 at h/2 using v0
    v1 = v0 + h*diffeq(1, r1, v0, t+hh) # 2: v1 using a(r) at h/2
    r1 = r1 + hh*diffeq(0, r0, v1, t+h) # 3: r1 at h using v1
    return r1, v1

"RK4 ODE algorithm"
def rk4solver(f,t,y,h):
    k1 = h*np.array(f(t,y))
    k2 = h*np.array(f(t + h/2., y + k1/2.))
    k3 = h*np.array(f(t + h/2., y + k2/2.))
    k4 = h*np.array(f(t + h, y + k3))
    y = y + (k1 + 2.*(k2 + k3) + k4)/6.
    return y

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

"QUESTION 1"
"(a)/(b)"

"Example One: h = 0.02 for 40 periods"

#animation function that allows user to pass step size 'h' and total
#animation time in units of T_0 (units of period since we let T_0 = 2*pi)
def go(h,time):
    cycle=15; ic=0
    niceFigure(False) #can delete false if you have TeX installed
    fig, axs = plt.subplots(2,2, figsize = (15,15))   #create four subplots displaying v-x phasespace and energy for both rk4 and leapfrog solver
#    plt.ion()

    #figure labels and size:
    axs[0,0].set_title('Leapfrog')
    axs[0,0].set_ylim(-1.2,1.2)
    axs[0,0].set_xlim(-1.2,1.2)
    axs[0,0].set_xlabel('$x$ (arb. units)')     # add labels
    axs[0,0].set_ylabel('$y$ (arb. units)')
    axs[0,1].set_title('Leapfrog')
    axs[0,1].set_ylim(0.0,1.0)
    #axs[0,1].set_xlim(-1.2,1.2)
    axs[0,1].set_xlabel('$t$ (arb. units)')     # add labels
    axs[0,1].set_ylabel('$E$ (arb. units)')
    axs[1,0].set_title('RK4')
    axs[1,0].set_ylim(-1.2,1.2)
    axs[1,0].set_xlim(-1.2,1.2)
    axs[1,0].set_xlabel('$x$ (arb. units)')     # add labels
    axs[1,0].set_ylabel('$y$ (arb. units)')
    axs[1,1].set_title('RK4')
    axs[1,1].set_ylim(0.0,1.0)
    #axs[1,1].set_xlim(-1.2,1.2)
    axs[1,1].set_xlabel('$t$ (arb. units)')     # add labels
    axs[1,1].set_ylabel('$E$ (arb. units)')
    
    
    T0 = 2.*np.pi   #initial period
    t_fin = time*T0
    t=0.
    tpause = 0.001 # delay within animation 
    plt.pause(2) # pause for 2 seconds before start
    
    
    xl, vl = 1.0, 0.0 #leapfrog initial values
    
    #rk4 initial conditions:
    dt = h   #step size
    tlist = np.arange(0.0, t_fin, dt)
    npts = len(tlist)
    y = np.zeros((npts,2))
    yinit = np.array([1.0,0.0])
    y1 = yinit
    y[0,:] = y1
    
    
    #create two plots for x and v vals:
    #once for the leapfrog solver
    line1, = axs[0,0].plot( xl, vl,'--', markersize=13) # Fetch the line object (use this for summary of phase space)
    line1a, = axs[0,0].plot( xl, vl,'o',color = 'cyan', markersize=13) # Fetch the line object (for trajectory or 'curser')
    #a second time for the RK4 solver
    line3, = axs[1,0].plot( y[:,0], y[:,1],'--', markersize=13) # Fetch the line object (use this for summary of phase space)
    line3a, = axs[1,0].plot( y[:,0], y[:,1],'o',color = 'cyan', markersize=13) # Fetch the line object (for trajectory or 'curser') 


    #empty lists to store values in for animations:
        
    t1 = []       #time
    
    #lists for leapfrog method:
    LF_xVals = []    #position
    LF_vVals = []    #velocity
    LF_Energy = []   #total energy (which should be conserved)
    
    #for RK4
    RK4_xVals = []
    RK4_vVals = []
    RK4_Energy = []
    
    
    while t<t_fin:        # loop until a final time in units of period
        
        xl, vl = leapfrog(oscillator, xl, vl, t, h) # solve using leapfrog
        
        y1 = rk4solver(derivs, t, y1, dt)   #solve using rk4
        y[int(t),:] = y1
        
        #append Leapfrog values
        E = oscillatorE(xl, vl)  #retrieve the total energy
        LF_Energy.append(E)         #store these values as we loop through
        t1.append(t)             #do the same for value of time
        LF_xVals.append(xl)         #and position values
        LF_vVals.append(vl)         #and velocity values
        
        #append RK4 values
        E1 = oscillatorE(y[int(t),0], y[int(t),1])
        RK4_Energy.append(E1)
        RK4_xVals.append(y[int(t),0])
        RK4_vVals.append(y[int(t),1])
        
        # you probably want to downsample if dt is too small (every cycle)
        if (ic % cycle == 0): # very simple animate (update data with pause)
            fig.suptitle("frame time {}".format(ic)) # show current time on graph

            line1.set_xdata(LF_xVals)    #plotting summary of phase space (dashed line)
            line1.set_ydata(LF_vVals)
            line1a.set_xdata(xl)      #line1a is just the curser (cyan dot)
            line1a.set_ydata(vl)
            
            line3.set_xdata(RK4_xVals)    #plotting summary of phase space (dashed line)
            line3.set_ydata(RK4_vVals)
            line3a.set_xdata(y[int(t),0])      #line3a is just the curser (cyan dot)
            line3a.set_ydata(y[int(t),1])
            
            axs[0,1].plot(t1,LF_Energy, color = "blue")   #plot the total energy as a function of time
            axs[1,1].plot(t1,RK4_Energy, color = "blue")
            
            plt.draw() # may not be needed (depends on your set up)
            plt.pause(tpause) # pause to see animation as code v. fast
           
        t  = t + h # loop time
        ic = ic + 1 # simple integer counter that might be useful 

go(0.02, 40)

#%%

"Example two: h = 0.2 for 40 periods"
"We sample less more updates per period to see the animation better"

def go(h,time):
    cycle=3; ic=0
    niceFigure(False) #can delete false if you have TeX installed
    fig, axs = plt.subplots(2,2, figsize = (15,15))   #create four subplots displaying v-x phasespace and energy for both rk4 and leapfrog solver
#    plt.ion()

    #figure labels and size:
    axs[0,0].set_title('Leapfrog')
    axs[0,0].set_ylim(-1.2,1.2)
    axs[0,0].set_xlim(-1.2,1.2)
    axs[0,0].set_xlabel('$x$ (arb. units)')     # add labels
    axs[0,0].set_ylabel('$y$ (arb. units)')
    axs[0,1].set_title('Leapfrog')
    axs[0,1].set_ylim(0.0,1.0)
    #axs[0,1].set_xlim(-1.2,1.2)
    axs[0,1].set_xlabel('$t$ (arb. units)')     # add labels
    axs[0,1].set_ylabel('$E$ (arb. units)')
    axs[1,0].set_title('RK4')
    axs[1,0].set_ylim(-1.2,1.2)
    axs[1,0].set_xlim(-1.2,1.2)
    axs[1,0].set_xlabel('$x$ (arb. units)')     # add labels
    axs[1,0].set_ylabel('$y$ (arb. units)')
    axs[1,1].set_title('RK4')
    axs[1,1].set_ylim(0.0,1.0)
    #axs[1,1].set_xlim(-1.2,1.2)
    axs[1,1].set_xlabel('$t$ (arb. units)')     # add labels
    axs[1,1].set_ylabel('$E$ (arb. units)')
    
    
    T0 = 2.*np.pi   #initial period
    t_fin = time*T0
    t=0.
    tpause = 0.001 # delay within animation 
    plt.pause(2) # pause for 2 seconds before start
    
    
    xl, vl = 1.0, 0.0 #leapfrog initial values
    
    #rk4 initial conditions:
    dt = h   #step size
    tlist = np.arange(0.0, t_fin, dt)
    npts = len(tlist)
    y = np.zeros((npts,2))
    yinit = np.array([1.0,0.0])
    y1 = yinit
    y[0,:] = y1
    
    
    #create two plots for x and v vals:
    #once for the leapfrog solver
    line1, = axs[0,0].plot( xl, vl,'--', markersize=13) # Fetch the line object (use this for summary of phase space)
    line1a, = axs[0,0].plot( xl, vl,'o',color = 'cyan', markersize=13) # Fetch the line object (for trajectory or 'curser')
    #a second time for the RK4 solver
    line3, = axs[1,0].plot( y[:,0], y[:,1],'--', markersize=13) # Fetch the line object (use this for summary of phase space)
    line3a, = axs[1,0].plot( y[:,0], y[:,1],'o',color = 'cyan', markersize=13) # Fetch the line object (for trajectory or 'curser') 


    #empty lists to store values in for animations:
        
    t1 = []       #time
    
    #lists for leapfrog method:
    LF_xVals = []    #position
    LF_vVals = []    #velocity
    LF_Energy = []   #total energy (which should be conserved)
    
    #for RK4
    RK4_xVals = []
    RK4_vVals = []
    RK4_Energy = []
    
    
    while t<t_fin:        # loop until a final time in units of period
        
        xl, vl = leapfrog(oscillator, xl, vl, t, h) # solve using leapfrog
        
        y1 = rk4solver(derivs, t, y1, dt)   #solve using rk4
        y[int(t),:] = y1
        
        #append Leapfrog values
        E = oscillatorE(xl, vl)  #retrieve the total energy
        LF_Energy.append(E)         #store these values as we loop through
        t1.append(t)             #do the same for value of time
        LF_xVals.append(xl)         #and position values
        LF_vVals.append(vl)         #and velocity values
        
        #append RK4 values
        E1 = oscillatorE(y[int(t),0], y[int(t),1])
        RK4_Energy.append(E1)
        RK4_xVals.append(y[int(t),0])
        RK4_vVals.append(y[int(t),1])
        
        # you probably want to downsample if dt is too small (every cycle)
        if (ic % cycle == 0): # very simple animate (update data with pause)
            fig.suptitle("frame time {}".format(ic)) # show current time on graph

            line1.set_xdata(LF_xVals)    #plotting summary of phase space (dashed line)
            line1.set_ydata(LF_vVals)
            line1a.set_xdata(xl)      #line1a is just the curser (cyan dot)
            line1a.set_ydata(vl)
            
            line3.set_xdata(RK4_xVals)    #plotting summary of phase space (dashed line)
            line3.set_ydata(RK4_vVals)
            line3a.set_xdata(y[int(t),0])      #line3a is just the curser (cyan dot)
            line3a.set_ydata(y[int(t),1])
            
            axs[0,1].plot(t1,LF_Energy, color = "blue")   #plot the total energy as a function of time
            axs[1,1].plot(t1,RK4_Energy, color = "blue")
            
            plt.draw() # may not be needed (depends on your set up)
            plt.pause(tpause) # pause to see animation as code v. fast
           
        t  = t + h # loop time
        ic = ic + 1 # simple integer counter that might be useful 


go(0.2, 40)

#%%
"QUESTION 2"

"""
I was able to make my animation function to include many user choices but couldn't quite 
figure out how to take snap shots to save as graphs in an automated way. The only way
I was aware of was to click the 'save' button inside the animation window. I would
appreciate a suggestion of many how to automate this in the future!
"""

"Finding root of interest of Euler's quintic equation"
#mass values:
m1 = 1
m2 = 2
m3 = 3

#subroutine found on: https://personal.math.ubc.ca/~pwalls/math-python/roots-optimization/newton/
#Newtons method for root finding of a polynomial with the form f(x) = 0:
def newtonSolve(f,Df,x0,epsilon,max_iter):  #takes function (f) and derivative (Df)
    xn = x0    #begin the iteration at an initial guess
    for n in range(0,max_iter):   #iterate a number of times equal to max_iter
        fxn = f(xn)    
        if abs(fxn) < epsilon: #if the function evaluated at xn is less than the tolerance, we have found the solution
            print('Found solution after',n,'iterations.')
            return xn
        Dfxn = Df(xn) #if the derivative is never equal to zero near our initial guess, 
        if Dfxn == 0:  #print there are no solutions 
            print('Zero derivative. No solution found.')
            return None
        xn = xn - fxn/Dfxn
    print('Exceeded maximum iterations. No solution found.') #if the max iteration is reached without a solution
    return None

#quintic polynomial (f) and its derivative function (Df):
f = lambda x: x**5*(m2 + m3) + x**4*(2*m2 + 3*m3) + x**3*(m2 + 3*m3) + x**2*(-3*m1 - m2) + x*(-3*m1 - 2*m2) + (-m1 - m2)
Df = lambda x: 5*x**4*(m2 + m3) + 4*x**3*(2*m2 + 3*m3) + 3*x**2*(m2 + 3*m3) + 2*x*(-3*m1 - m2) + (-3*m1 - 2*m2)


print("Solution to f(x) = 0 is:",newtonSolve(f,Df,0.5,1e-8,10))

#create the lambda variable
l = newtonSolve(f,Df,0.5,1e-8,10)
#%%
"Setup initial conditions beginning with w0 = 1 (no perturbation)"

w = 1
a = ((1/(w**2))*(m2 + m3 - (m1*(1 + 2*l))/(l**2*(1+l)**2)))**(1/3) #find the distance 'a'
x2 = (1/(w**2*a**2))*(m1/(l**2) - m3)  #position of mass 2
x1 = x2 - l*a    #of mass 1
x3 = -(m1*x1 + m2*x2)/m3   #of mass3
v1y = w*x1    #initial velocities
v2y = w*x2
v3y = w*x3

# Fixed initial condition (1)
def init_cond1():
    r, v = np.zeros((3,2)), np.zeros((3,2))
    # initial r and v - set 2
    r[0,0] = x1; r[1,0] = x2; r[2,0] = x3
    v[0,1] = w*x1; v[1,1] = w*x2; v[2,1] = w*x3; 
    return r , v


rinit, vinit = init_cond1()
#%%
"""
Run animation for w = 1
go function has choice for user inputs - outlined at end of function
"""
def go(time, id, h, cycle, r0, v0, Animate):
    ic=0
    niceFigure(False) #can delete false if you have TeX installed

    c1 = plt.Circle((0,0), 0.07, color = 'black', fill = False) #center of mass is at the origin
    c2 = plt.Circle((r0[0,0],0), 0.04, color = 'purple', fill = True) #starting position for mass 1
    c3 = plt.Circle((r0[1,0],0), 0.04, color = 'red', fill = True) #for mass 2
    c4 = plt.Circle((r0[2,0],0), 0.04, color = 'aqua', fill = True) #for mass 3
    
    fig, (ax1,ax2) = plt.subplots(2, figsize = (10,10), gridspec_kw={'height_ratios': [1,2]})
#    plt.ion()
    ax2.set_xlabel('$x$ (arb. units)')     # add labels
    ax2.set_ylabel('$y$ (arb. units)')
    ax1.set_xlabel('Time $(T_0)$')
    ax1.set_ylabel('E (arb. units)')
    
    #add the starting positions as filled circles
    ax2.add_patch(c1)    
    ax2.add_patch(c2)    
    ax2.add_patch(c3)    
    ax2.add_patch(c4)    

    #add temporary quivers
    Q1=plt.quiver(0,0,-1,width=0.008,color='b',headwidth=8.,headlength=3.,headaxislength=3,scale=40)    
    Q2=plt.quiver(0,0,-1,width=0.008,color='b',headwidth=8.,headlength=3.,headaxislength=3,scale=40)    
    Q3=plt.quiver(0,0,-1,width=0.008,color='b',headwidth=8.,headlength=3.,headaxislength=3,scale=40)    
    
    w_0 = w
    T0 = 2.*np.pi/w_0
    t_fin = time*T0
    t=0.
    tpause = 0.00001 # delay within animation 
    plt.pause(2) # pause for 2 seconds before start
    
    if (id==0):
        
        y_array = ([r0,v0])
        y_init = np.reshape(y_array,(12))
        yl = y_init  #rk4 initial conditions:

        E3body = [] #empty list to store energy values        
        tVals = []
        while t<t_fin:        # loop for 'time' number of periods
        
        # so your leapfrog or RK4 call could be here for example ...
        
            yl = rk4solver(threeb_RK4_derivs, t, yl, h)
            rr, vr = np.reshape(yl[:6],(3,2)), np.reshape(yl[6:], (3,2))
        
            EVals = threeb_Energy(rr, vr)
            E3body.append(EVals)
            tVals.append(t/T0)
        
            # you probably want to downsample if dt is too small (every cycle)
            if (ic % cycle == 0): # very simple animate (update data with pause)
                fig.suptitle("frame time {}".format(ic)) # show current time on graph

                Q1.remove()
                Q2.remove()
                Q3.remove()
                    
                ax2.plot(rr[0,0], rr[0,1], '.', color = 'purple')
                ax2.plot(rr[1,0], rr[1,1], '.', color = 'red')
                ax2.plot(rr[2,0], rr[2,1], '.', color = 'aqua')          
    
                ax1.plot(tVals,E3body, color = 'magenta')
            
                Q1 = ax2.quiver(rr[0,0],rr[0,1],vr[0,0],vr[0,1], width = 0.008, color = 'purple', headwidth=6.,headlength=3.,headaxislength=3,scale=40)
                Q2 = ax2.quiver(rr[1,0],rr[1,1],vr[1,0],vr[1,1], width = 0.008, color = 'red', headwidth=6.,headlength=3.,headaxislength=3,scale=40)
                Q3 = ax2.quiver(rr[2,0],rr[2,1],vr[2,0],vr[2,1], width = 0.008, color = 'aqua', headwidth=6.,headlength=3.,headaxislength=3,scale=40)
            
                plt.draw() # may not be needed (depends on your set up)
                if (Animate == 0):    
                    plt.pause(tpause) # pause to see animation as code v. fast
                else:
                    plt.show()
           
            t  = t + h # loop time
            ic = ic + 1 # simple integer counter that migth be useful 

    else:
        
        rl, vl = r0, v0      # initial values

        E3body = [] #empty list to store energy values        
        tVals = []
        while t<t_fin:        # loop for 'time' number of periods
            
            
            # so your leapfrog or RK4 call could be here for example ...
            rl, vl = leapfrog(threeb_LF_derivs, rl, vl, t, h) # solve using leapfrog
        
            EVals = threeb_Energy(rl, vl)
            E3body.append(EVals)
            tVals.append(t/(T0))
        
            # you probably want to downsample if dt is too small (every cycle)
            if (ic % cycle == 0): # very simple animate (update data with pause)
                fig.suptitle("frame time {}".format(ic)) # show current time on graph
                    
                Q1.remove()
                Q2.remove()
                Q3.remove()

                    
                ax2.plot(rl[0,0], rl[0,1], '.', color = 'purple')
                ax2.plot(rl[1,0], rl[1,1], '.', color = 'red')
                ax2.plot(rl[2,0], rl[2,1], '.', color = 'aqua')
            
                ax1.plot(tVals,E3body, color = 'magenta')
            
                Q1 = ax2.quiver(rl[0,0],rl[0,1],vl[0,0],vl[0,1], width = 0.008, color = 'purple', headwidth=6.,headlength=3.,headaxislength=3,scale=40)
                Q2 = ax2.quiver(rl[1,0],rl[1,1],vl[1,0],vl[1,1], width = 0.008, color = 'red', headwidth=6.,headlength=3.,headaxislength=3,scale=40)
                Q3 = ax2.quiver(rl[2,0],rl[2,1],vl[2,0],vl[2,1], width = 0.008, color = 'aqua', headwidth=6.,headlength=3.,headaxislength=3,scale=40)

                
                plt.draw() # may not be needed (depends on your set up)
                if (Animate == 0):    
                    plt.pause(tpause) # pause to see animation as code v. fast
                else:
                    plt.show()
            t  = t + h # loop time
            ic = ic + 1 # simple integer counter that migth be useful 

"function for animation with parameters defined as:"
"go(*time, id, step size, update size, initial r, intitial v, animation)"
"""
time: number of periods for animation
id: 0 returns rk4 method, 1 uses leapfrog
stepsize: this is h
update size or 'cycle': how many frames per update
initial r: position starting conditions
intitial v: velocity starting conditions
animate: 1 returns the final trajectory, 0 returns animation 
"""
go(1,1, 0.001, 60, rinit, vinit, 0)

#%%
"Initial conditions to now include a perturbation of \delta"

delta = 1e-9
w = 1 + delta
a = ((1/(w**2))*(m2 + m3 - (m1*(1 + 2*l))/(l**2*(1+l)**2)))**(1/3) #find the distance 'a'
x2 = (1/(w**2*a**2))*(m1/(l**2) - m3)  #position of mass 2
x1 = x2 - l*a    #of mass 1
x3 = -(m1*x1 + m2*x2)/m3   #of mass3
v1y = w*x1    #initial velocities
v2y = w*x2
v3y = w*x3

# Fixed initial condition (1)
def init_cond1():
    r, v = np.zeros((3,2)), np.zeros((3,2))
    # initial r and v - set 2
    r[0,0] = x1; r[1,0] = x2; r[2,0] = x3
    v[0,1] = w*x1; v[1,1] = w*x2; v[2,1] = w*x3; 
    return r , v


rinit, vinit = init_cond1()
#%%
"""
Run animation for w = 1 + delta
go function has choice for user inputs - outlined at end of function
"""
def go(time, id, h, cycle, r0, v0, Animate):
    ic=0
    niceFigure(False) #can delete false if you have TeX installed

    c1 = plt.Circle((0,0), 0.07, color = 'black', fill = False) #center of mass is at the origin
    c2 = plt.Circle((r0[0,0],0), 0.04, color = 'purple', fill = True) #starting position for mass 1
    c3 = plt.Circle((r0[1,0],0), 0.04, color = 'red', fill = True) #for mass 2
    c4 = plt.Circle((r0[2,0],0), 0.04, color = 'aqua', fill = True) #for mass 3
    
    fig, (ax1,ax2) = plt.subplots(2, figsize = (10,10), gridspec_kw={'height_ratios': [1,2]})
#    plt.ion()
    ax2.set_xlabel('$x$ (arb. units)')     # add labels
    ax2.set_ylabel('$y$ (arb. units)')
    ax1.set_xlabel('Time $(T_0)$')
    ax1.set_ylabel('E (arb. units)')
    
    #add the starting positions as filled circles
    ax2.add_patch(c1)    
    ax2.add_patch(c2)    
    ax2.add_patch(c3)    
    ax2.add_patch(c4)    

    #add temporary quivers
    Q1=plt.quiver(0,0,-1,width=0.008,color='b',headwidth=8.,headlength=3.,headaxislength=3,scale=40)    
    Q2=plt.quiver(0,0,-1,width=0.008,color='b',headwidth=8.,headlength=3.,headaxislength=3,scale=40)    
    Q3=plt.quiver(0,0,-1,width=0.008,color='b',headwidth=8.,headlength=3.,headaxislength=3,scale=40)    
    
    w_0 = w
    T0 = 2.*np.pi/w_0
    t_fin = time*T0
    t=0.
    tpause = 0.00001 # delay within animation 
    plt.pause(2) # pause for 2 seconds before start
    
    if (id==0):
        
        y_array = ([r0,v0])
        y_init = np.reshape(y_array,(12))
        yl = y_init  #rk4 initial conditions:

        E3body = [] #empty list to store energy values        
        tVals = []
        while t<t_fin:        # loop for 'time' number of periods
        
        # so your leapfrog or RK4 call could be here for example ...
        
            yl = rk4solver(threeb_RK4_derivs, t, yl, h)
            rr, vr = np.reshape(yl[:6],(3,2)), np.reshape(yl[6:], (3,2))
        
            EVals = threeb_Energy(rr, vr)
            E3body.append(EVals)
            tVals.append(t/T0)
        
            # you probably want to downsample if dt is too small (every cycle)
            if (ic % cycle == 0): # very simple animate (update data with pause)
                fig.suptitle("frame time {}".format(ic)) # show current time on graph

                Q1.remove()
                Q2.remove()
                Q3.remove()
                    
                ax2.plot(rr[0,0], rr[0,1], '.', color = 'purple')
                ax2.plot(rr[1,0], rr[1,1], '.', color = 'red')
                ax2.plot(rr[2,0], rr[2,1], '.', color = 'aqua')          
    
                ax1.plot(tVals,E3body, color = 'magenta')
            
                Q1 = ax2.quiver(rr[0,0],rr[0,1],vr[0,0],vr[0,1], width = 0.008, color = 'purple', headwidth=6.,headlength=3.,headaxislength=3,scale=40)
                Q2 = ax2.quiver(rr[1,0],rr[1,1],vr[1,0],vr[1,1], width = 0.008, color = 'red', headwidth=6.,headlength=3.,headaxislength=3,scale=40)
                Q3 = ax2.quiver(rr[2,0],rr[2,1],vr[2,0],vr[2,1], width = 0.008, color = 'aqua', headwidth=6.,headlength=3.,headaxislength=3,scale=40)
            
                plt.draw() # may not be needed (depends on your set up)
                if (Animate == 0):    
                    plt.pause(tpause) # pause to see animation as code v. fast
                else:
                    plt.show()
           
            t  = t + h # loop time
            ic = ic + 1 # simple integer counter that migth be useful 

    else:
        
        rl, vl = r0, v0      # initial values

        E3body = [] #empty list to store energy values        
        tVals = []
        while t<t_fin:        # loop for 'time' number of periods
            
            
            # so your leapfrog or RK4 call could be here for example ...
            rl, vl = leapfrog(threeb_LF_derivs, rl, vl, t, h) # solve using leapfrog
        
            EVals = threeb_Energy(rl, vl)
            E3body.append(EVals)
            tVals.append(t/(T0))
        
            # you probably want to downsample if dt is too small (every cycle)
            if (ic % cycle == 0): # very simple animate (update data with pause)
                fig.suptitle("frame time {}".format(ic)) # show current time on graph
                    
                Q1.remove()
                Q2.remove()
                Q3.remove()

                    
                ax2.plot(rl[0,0], rl[0,1], '.', color = 'purple')
                ax2.plot(rl[1,0], rl[1,1], '.', color = 'red')
                ax2.plot(rl[2,0], rl[2,1], '.', color = 'aqua')
            
                ax1.plot(tVals,E3body, color = 'magenta')
            
                Q1 = ax2.quiver(rl[0,0],rl[0,1],vl[0,0],vl[0,1], width = 0.008, color = 'purple', headwidth=6.,headlength=3.,headaxislength=3,scale=40)
                Q2 = ax2.quiver(rl[1,0],rl[1,1],vl[1,0],vl[1,1], width = 0.008, color = 'red', headwidth=6.,headlength=3.,headaxislength=3,scale=40)
                Q3 = ax2.quiver(rl[2,0],rl[2,1],vl[2,0],vl[2,1], width = 0.008, color = 'aqua', headwidth=6.,headlength=3.,headaxislength=3,scale=40)

                
                plt.draw() # may not be needed (depends on your set up)
                if (Animate == 0):    
                    plt.pause(tpause) # pause to see animation as code v. fast
                else:
                    plt.show()
            t  = t + h # loop time
            ic = ic + 1 # simple integer counter that migth be useful 

"function for animation with parameters defined as:"
"go(*time, id, step size, update size, initial r, intitial v, animation)"
"""
time: number of periods for animation
id: 0 returns rk4 method, 1 uses leapfrog
stepsize: this is h
update size or 'cycle': how many frames per update
initial r: position starting conditions
intitial v: velocity starting conditions
animate: 1 returns the final trajectory, 0 returns animation 
"""
go(4,0, 0.001, 60, rinit, vinit, 0)

#%%

"Initial conditions to now include a perturbation of \delta"

delta = 1e-9
w = 1 - delta
a = ((1/(w**2))*(m2 + m3 - (m1*(1 + 2*l))/(l**2*(1+l)**2)))**(1/3) #find the distance 'a'
x2 = (1/(w**2*a**2))*(m1/(l**2) - m3)  #position of mass 2
x1 = x2 - l*a    #of mass 1
x3 = -(m1*x1 + m2*x2)/m3   #of mass3
v1y = w*x1    #initial velocities
v2y = w*x2
v3y = w*x3

# Fixed initial condition (1)
def init_cond1():
    r, v = np.zeros((3,2)), np.zeros((3,2))
    # initial r and v - set 2
    r[0,0] = x1; r[1,0] = x2; r[2,0] = x3
    v[0,1] = w*x1; v[1,1] = w*x2; v[2,1] = w*x3; 
    return r , v


rinit, vinit = init_cond1()
#%%
"""
Run animation for w = 1 - delta
go function has choice for user inputs - outlined at end of function
"""
def go(time, id, h, cycle, r0, v0, Animate):
    ic=0
    niceFigure(False) #can delete false if you have TeX installed

    c1 = plt.Circle((0,0), 0.07, color = 'black', fill = False) #center of mass is at the origin
    c2 = plt.Circle((r0[0,0],0), 0.04, color = 'purple', fill = True) #starting position for mass 1
    c3 = plt.Circle((r0[1,0],0), 0.04, color = 'red', fill = True) #for mass 2
    c4 = plt.Circle((r0[2,0],0), 0.04, color = 'aqua', fill = True) #for mass 3
    
    fig, (ax1,ax2) = plt.subplots(2, figsize = (10,10), gridspec_kw={'height_ratios': [1,2]})
#    plt.ion()
    ax2.set_xlabel('$x$ (arb. units)')     # add labels
    ax2.set_ylabel('$y$ (arb. units)')
    ax1.set_xlabel('Time $(T_0)$')
    ax1.set_ylabel('E (arb. units)')
    
    #add the starting positions as filled circles
    ax2.add_patch(c1)    
    ax2.add_patch(c2)    
    ax2.add_patch(c3)    
    ax2.add_patch(c4)    

    #add temporary quivers
    Q1=plt.quiver(0,0,-1,width=0.008,color='b',headwidth=8.,headlength=3.,headaxislength=3,scale=40)    
    Q2=plt.quiver(0,0,-1,width=0.008,color='b',headwidth=8.,headlength=3.,headaxislength=3,scale=40)    
    Q3=plt.quiver(0,0,-1,width=0.008,color='b',headwidth=8.,headlength=3.,headaxislength=3,scale=40)    
    
    w_0 = w
    T0 = 2.*np.pi/w_0
    t_fin = time*T0
    t=0.
    tpause = 0.00001 # delay within animation 
    plt.pause(2) # pause for 2 seconds before start
    
    if (id==0):
        
        y_array = ([r0,v0])
        y_init = np.reshape(y_array,(12))
        yl = y_init  #rk4 initial conditions:

        E3body = [] #empty list to store energy values        
        tVals = []
        while t<t_fin:        # loop for 'time' number of periods
        
        # so your leapfrog or RK4 call could be here for example ...
        
            yl = rk4solver(threeb_RK4_derivs, t, yl, h)
            rr, vr = np.reshape(yl[:6],(3,2)), np.reshape(yl[6:], (3,2))
        
            EVals = threeb_Energy(rr, vr)
            E3body.append(EVals)
            tVals.append(t/T0)
        
            # you probably want to downsample if dt is too small (every cycle)
            if (ic % cycle == 0): # very simple animate (update data with pause)
                fig.suptitle("frame time {}".format(ic)) # show current time on graph

                Q1.remove()
                Q2.remove()
                Q3.remove()
                    
                ax2.plot(rr[0,0], rr[0,1], '.', color = 'purple')
                ax2.plot(rr[1,0], rr[1,1], '.', color = 'red')
                ax2.plot(rr[2,0], rr[2,1], '.', color = 'aqua')          
    
                ax1.plot(tVals,E3body, color = 'magenta')
            
                Q1 = ax2.quiver(rr[0,0],rr[0,1],vr[0,0],vr[0,1], width = 0.008, color = 'purple', headwidth=6.,headlength=3.,headaxislength=3,scale=40)
                Q2 = ax2.quiver(rr[1,0],rr[1,1],vr[1,0],vr[1,1], width = 0.008, color = 'red', headwidth=6.,headlength=3.,headaxislength=3,scale=40)
                Q3 = ax2.quiver(rr[2,0],rr[2,1],vr[2,0],vr[2,1], width = 0.008, color = 'aqua', headwidth=6.,headlength=3.,headaxislength=3,scale=40)
            
                plt.draw() # may not be needed (depends on your set up)
                if (Animate == 0):    
                    plt.pause(tpause) # pause to see animation as code v. fast
                else:
                    plt.show()
           
            t  = t + h # loop time
            ic = ic + 1 # simple integer counter that migth be useful 

    else:
        
        rl, vl = r0, v0      # initial values

        E3body = [] #empty list to store energy values        
        tVals = []
        while t<t_fin:        # loop for 'time' number of periods
            
            
            # so your leapfrog or RK4 call could be here for example ...
            rl, vl = leapfrog(threeb_LF_derivs, rl, vl, t, h) # solve using leapfrog
        
            EVals = threeb_Energy(rl, vl)
            E3body.append(EVals)
            tVals.append(t/(T0))
        
            # you probably want to downsample if dt is too small (every cycle)
            if (ic % cycle == 0): # very simple animate (update data with pause)
                fig.suptitle("frame time {}".format(ic)) # show current time on graph
                    
                Q1.remove()
                Q2.remove()
                Q3.remove()

                    
                ax2.plot(rl[0,0], rl[0,1], '.', color = 'purple')
                ax2.plot(rl[1,0], rl[1,1], '.', color = 'red')
                ax2.plot(rl[2,0], rl[2,1], '.', color = 'aqua')
            
                ax1.plot(tVals,E3body, color = 'magenta')
            
                Q1 = ax2.quiver(rl[0,0],rl[0,1],vl[0,0],vl[0,1], width = 0.008, color = 'purple', headwidth=6.,headlength=3.,headaxislength=3,scale=40)
                Q2 = ax2.quiver(rl[1,0],rl[1,1],vl[1,0],vl[1,1], width = 0.008, color = 'red', headwidth=6.,headlength=3.,headaxislength=3,scale=40)
                Q3 = ax2.quiver(rl[2,0],rl[2,1],vl[2,0],vl[2,1], width = 0.008, color = 'aqua', headwidth=6.,headlength=3.,headaxislength=3,scale=40)

                
                plt.draw() # may not be needed (depends on your set up)
                if (Animate == 0):    
                    plt.pause(tpause) # pause to see animation as code v. fast
                else:
                    plt.show()
            t  = t + h # loop time
            ic = ic + 1 # simple integer counter that migth be useful 

"function for animation with parameters defined as:"
"go(*time, id, step size, update size, initial r, intitial v, animation)"
"""
time: number of periods for animation
id: 0 returns rk4 method, 1 uses leapfrog
stepsize: this is h
update size or 'cycle': how many frames per update
initial r: position starting conditions
intitial v: velocity starting conditions
animate: 1 returns the final trajectory, 0 returns animation 
"""
go(4,1, 0.001, 60, rinit, vinit, 0)

#%%
"""
In the following three cells, I tried to solve the system with the odeint solver but 
couldn't quite get there. Any suggestions for how to impliment this for the future 
would be greatly appreciated!
"""

"(b)"

"Three Body EOM's for odeint solver"
def threeb_derivs1(y, t): # my derivatives function
    r1x, r1y, r2x, r2y, r3x, r3y, v1x, v1y, v2x, v2y, v3x, v3y = y 
    # ri, vi = y[:6], y[6:]
    # r, v = np.reshape(ri,(3,2)), np.reshape(vi, (3,2))
    # dr, dv = np.zeros((3,2)), np.zeros((3,2))
        
    r12x, r23x, r31x = r1x - r2x, r2x - r3x, r3x - r1x
    r12y, r23y, r31y = r1y - r2y, r2y - r3y, r3y - r1y

    # s12x, s23x, s31x = np.sqrt(r12x.dot(r12x)), np.sqrt(r23x.dot(r23x)), np.sqrt(r31x.dot(r31x))
    # s12y, s23y, s31y = np.sqrt(r12y.dot(r12y)), np.sqrt(r23y.dot(r23y)), np.sqrt(r31y.dot(r31y))

    
    dr1x = v1x  #dr1x/dt = dv1x; #ie [i,j] corresponds to i = 1,2,3 and j = x,y
    dv1x = -m2*(r12x/(r12x**3)) - m3*((-r31x)/(r31x**3))
    dr2x = v2x 
    dv2x = -m3*(r23x/(r23x**3)) - m1*((-r12x)/(r12x**3))
    dr3x = v3x
    dv3x = -m1*(r31x/(r31x**3)) - m2*((-r23x)/(r23x**3))
    
    dr1y = v1y  #dr1x/dt = dv1x; #ie [i,j] corresponds to i = 1,2,3 and j = x,y
    dv1y = -m2*(r12y/(r12y**3)) - m3*((-r31y)/(r31y**3))
    dr2y = v2y 
    dv2y = -m3*(r23x/(r23y**3)) - m1*((-r12y)/(r12y**3))
    dr3y = v3y
    dv3y = -m1*(r31x/(r31y**3)) - m2*((-r23y)/(r23y**3))
    
    dy = [dr1x, dr1y, dr2x, dr2y, dr3x, dr3y, dv1x, dv1y, dv2x, dv2y, dv3x, dv3y]    
    return dy

#%%

"""
"REFERENCE CODE"
"Solve the three body problem using scipy's odeint solver"
def go(time, h, cycle, r0, v0, Animate):
    ic=0
    niceFigure(False) #can delete false if you have TeX installed

    c1 = plt.Circle((0,0), 0.07, color = 'black', fill = False) #center of mass is at the origin
    c2 = plt.Circle((r0[0,0],0), 0.04, color = 'purple', fill = True) #starting position for mass 1
    c3 = plt.Circle((r0[1,0],0), 0.04, color = 'red', fill = True) #for mass 2
    c4 = plt.Circle((r0[2,0],0), 0.04, color = 'aqua', fill = True) #for mass 3
    
    fig, (ax1,ax2) = plt.subplots(2, figsize = (10,10), gridspec_kw={'height_ratios': [1,2]})
#    plt.ion()
    ax2.set_xlabel('$x$ (arb. units)')     # add labels
    ax2.set_ylabel('$y$ (arb. units)')
    ax1.set_xlabel('Time $(T_0)$')
    ax1.set_ylabel('E (arb. units)')
    
    #add the starting positions as filled circles
    ax2.add_patch(c1)    
    ax2.add_patch(c2)    
    ax2.add_patch(c3)    
    ax2.add_patch(c4)    

    #add temporary quivers
    Q1=plt.quiver(0,0,-1,width=0.008,color='b',headwidth=8.,headlength=3.,headaxislength=3,scale=40)    
    Q2=plt.quiver(0,0,-1,width=0.008,color='b',headwidth=8.,headlength=3.,headaxislength=3,scale=40)    
    Q3=plt.quiver(0,0,-1,width=0.008,color='b',headwidth=8.,headlength=3.,headaxislength=3,scale=40)    
    
    w_0 = w
    T0 = 2.*np.pi/w_0
    t_fin = time*T0
    t=0.
    tpause = 0.00001 # delay within animation 
    plt.pause(2) # pause for 2 seconds before start
    t1 = np.arange(0.01, t_fin,h)
    
        
    y_array = ([r0,v0])
    y_init = np.reshape(y_array,(12))
    print(y_init)
    y0 = y_init  #rk4 initial conditions:
    yl = odeint(threeb_derivs1, y0, t1)

    # E3body = [] #empty list to store energy values        
    # tVals = []
    # while t<t_fin:        # loop for 'time' number of periods
    # for i in range()
        # t1 = np.arange(0.,t,h)
    # so your leapfrog or RK4 call could be here for example ...
        
        # yl = odeint(threeb_derivs1, y0, t1)
        # rr, vr = np.reshape(yl[:6],(3,2)), np.reshape(yl[6:], (3,2))
        # print(yl)
        # EVals = threeb_Energy(rr, vr)
        # E3body.append(EVals)
        # tVals.append(t/T0)
        
        # you probably want to downsample if dt is too small (every cycle)
        # if (ic % cycle == 0): # very simple animate (update data with pause)
            # fig.suptitle("frame time {}".format(ic)) # show current time on graph

            # Q1.remove()
            # Q2.remove()
            # Q3.remove()
                    
    ax2.plot(yl[:,0], yl[:,1], '.', color = 'purple')
    ax2.plot(yl[:,2], yl[:,3], '.', color = 'red')
    ax2.plot(yl[:,4], yl[:,5], '.', color = 'aqua')          
    
            # ax1.plot(tVals,E3body, color = 'magenta')
            
            # Q1 = ax2.quiver(rr[0,0],rr[0,1],vr[0,0],vr[0,1], width = 0.008, color = 'purple', headwidth=6.,headlength=3.,headaxislength=3,scale=40)
            # Q2 = ax2.quiver(rr[1,0],rr[1,1],vr[1,0],vr[1,1], width = 0.008, color = 'red', headwidth=6.,headlength=3.,headaxislength=3,scale=40)
            # Q3 = ax2.quiver(rr[2,0],rr[2,1],vr[2,0],vr[2,1], width = 0.008, color = 'aqua', headwidth=6.,headlength=3.,headaxislength=3,scale=40)
            
            # plt.draw() # may not be needed (depends on your set up)
            # if (Animate == 0):    
                # plt.pause(tpause) # pause to see animation as code v. fast
            # else:
                # plt.show()
           
        # t  = t + h # loop time
        # ic = ic + 1 # simple integer counter that migth be useful 

"function for animation with parameters defined as:"
"go(*time, id, step size, update size, initial r, intitial v, animation)"


time: number of periods for animation
stepsize: this is h
update size or 'cycle': how many frames per update
initial r: position starting conditions
intitial v: velocity starting conditions
animate: 1 returns the final trajectory, 0 returns animation 


go(1, 0.001, 60, rinit, vinit, 0)
"""


#%%
"""
"TESTING THE ODEINT SOLVER WITHOUT THE ANIMATION SECTION"
"Solve the three body problem using scipy's odeint solver"

def go(time, h, cycle, r0, v0, Animate):
    ic=0
    niceFigure(False) #can delete false if you have TeX installed

    c1 = plt.Circle((0,0), 0.07, color = 'black', fill = False) #center of mass is at the origin
    c2 = plt.Circle((r0[0,0],0), 0.04, color = 'purple', fill = True) #starting position for mass 1
    c3 = plt.Circle((r0[1,0],0), 0.04, color = 'red', fill = True) #for mass 2
    c4 = plt.Circle((r0[2,0],0), 0.04, color = 'aqua', fill = True) #for mass 3
    
    fig, (ax1,ax2) = plt.subplots(2, figsize = (10,10), gridspec_kw={'height_ratios': [1,2]})
#    plt.ion()
    ax2.set_xlabel('$x$ (arb. units)')     # add labels
    ax2.set_ylabel('$y$ (arb. units)')
    ax1.set_xlabel('Time $(T_0)$')
    ax1.set_ylabel('E (arb. units)')
    
    #add the starting positions as filled circles
    ax2.add_patch(c1)    
    ax2.add_patch(c2)    
    ax2.add_patch(c3)    
    ax2.add_patch(c4)    

    #add temporary quivers
    Q1=plt.quiver(0,0,-1,width=0.008,color='b',headwidth=8.,headlength=3.,headaxislength=3,scale=40)    
    Q2=plt.quiver(0,0,-1,width=0.008,color='b',headwidth=8.,headlength=3.,headaxislength=3,scale=40)    
    Q3=plt.quiver(0,0,-1,width=0.008,color='b',headwidth=8.,headlength=3.,headaxislength=3,scale=40)    
    
    w_0 = w
    T0 = 2.*np.pi/w_0
    t_fin = time*T0
    # t=0.
    tpause = 0.00001 # delay within animation 
    plt.pause(2) # pause for 2 seconds before start
    t = np.arange(0.,t_fin,h)
    
    y_array = ([r0,v0])
    y_init = np.reshape(y_array,(12))
    y0 = y_init  #rk4 initial conditions:
    
    E3body = [] #empty list to store energy values        
    tVals = []
    # while t<t_fin:        # loop for 'time' number of periods
    # for i in range()
        
    # so your leapfrog or RK4 call could be here for example ...
        
    yl = odeint(threeb_derivs1, y0, t)
    # rr, vr = np.reshape(yl[:6],(3,2)), np.reshape(yl[6:], (3,2))
        
    # EVals = threeb_Energy(rr, vr)
    # E3body.append(EVals)
    tVals.append(t/T0)
        
    # you probably want to downsample if dt is too small (every cycle)
    if (ic % cycle == 0): # very simple animate (update data with pause)
        fig.suptitle("frame time {}".format(ic)) # show current time on graph

        Q1.remove()
        Q2.remove()
        Q3.remove()
                    
        ax2.plot(yl[:,0], yl[:,1], '.', color = 'purple')
        ax2.plot(yl[:,2], yl[:,3], '.', color = 'red')
        ax2.plot(yl[:,4], yl[:,5], '.', color = 'aqua')          
        
        # ax1.plot(tVals,E3body, color = 'magenta')
        
        # Q1 = ax2.quiver(rr[0,0],rr[0,1],vr[0,0],vr[0,1], width = 0.008, color = 'purple', headwidth=6.,headlength=3.,headaxislength=3,scale=40)
        # Q2 = ax2.quiver(rr[1,0],rr[1,1],vr[1,0],vr[1,1], width = 0.008, color = 'red', headwidth=6.,headlength=3.,headaxislength=3,scale=40)
        # Q3 = ax2.quiver(rr[2,0],rr[2,1],vr[2,0],vr[2,1], width = 0.008, color = 'aqua', headwidth=6.,headlength=3.,headaxislength=3,scale=40)
            
        plt.draw() # may not be needed (depends on your set up)
        if (Animate == 0):    
            plt.pause(tpause) # pause to see animation as code v. fast
        else:
            plt.show()
        
        t  = t + h # loop time
        ic = ic + 1 # simple integer counter that migth be useful 

"function for animation with parameters defined as:"
"go(*time, id, step size, update size, initial r, intitial v, animation)"

time: number of periods for animation
stepsize: this is h
update size or 'cycle': how many frames per update
initial r: position starting conditions
intitial v: velocity starting conditions
animate: 1 returns the final trajectory, 0 returns animation 

go(1, 0.001, 60, rinit, vinit, 0)
"""

#%%
"(c)"
"Initial conditions to include a perturbation of \delta"

delta = 1e-9
w = 1 + delta  #question asks for this
a = ((1/(w**2))*(m2 + m3 - (m1*(1 + 2*l))/(l**2*(1+l)**2)))**(1/3) #find the distance 'a'
x2 = (1/(w**2*a**2))*(m1/(l**2) - m3)  #position of mass 2
x1 = x2 - l*a    #of mass 1
x3 = -(m1*x1 + m2*x2)/m3   #of mass3
v1y = w*x1    #initial velocities
v2y = w*x2
v3y = w*x3

# Fixed initial condition (1)
def init_cond1():
    r, v = np.zeros((3,2)), np.zeros((3,2))
    # initial r and v - set 2
    r[0,0] = x1; r[1,0] = x2; r[2,0] = x3
    v[0,1] = w*x1; v[1,1] = w*x2; v[2,1] = w*x3; 
    return r , v


rinit, vinit = init_cond1()

#%%

"animation function that flips the velocities half way through the simulation"
def go(time, id, h, cycle, r0, v0, Animate):
    ic=0
    niceFigure(False) #can delete false if you have TeX installed

    c1 = plt.Circle((0,0), 0.07, color = 'black', fill = False) #center of mass is at the origin
    c2 = plt.Circle((r0[0,0],0), 0.04, color = 'purple', fill = True) #starting position for mass 1
    c3 = plt.Circle((r0[1,0],0), 0.04, color = 'red', fill = True) #for mass 2
    c4 = plt.Circle((r0[2,0],0), 0.04, color = 'aqua', fill = True) #for mass 3
    
    fig, (ax1,ax2) = plt.subplots(2, figsize = (10,10), gridspec_kw={'height_ratios': [1,2]})
#    plt.ion()
    ax2.set_xlabel('$x$ (arb. units)')     # add labels
    ax2.set_ylabel('$y$ (arb. units)')
    ax1.set_xlabel('Time $(T_0)$')
    ax1.set_ylabel('E (arb. units)')
    
    #add the starting positions as filled circles
    ax2.add_patch(c1)    
    ax2.add_patch(c2)    
    ax2.add_patch(c3)    
    ax2.add_patch(c4)    

    #add temporary quivers
    Q1=plt.quiver(0,0,-1,width=0.008,color='b',headwidth=8.,headlength=3.,headaxislength=3,scale=40)    
    Q2=plt.quiver(0,0,-1,width=0.008,color='b',headwidth=8.,headlength=3.,headaxislength=3,scale=40)    
    Q3=plt.quiver(0,0,-1,width=0.008,color='b',headwidth=8.,headlength=3.,headaxislength=3,scale=40)    
    
    w_0 = w
    T0 = 2.*np.pi/w_0
    t_fin = time*T0
    t=0.
    tpause = 0.00001 # delay within animation 
    plt.pause(2) # pause for 2 seconds before start
    
    if (id==0):
        
        y_array = ([r0,v0])
        y_init = np.reshape(y_array,(12))
        yl = y_init  #rk4 initial conditions:

        E3body = [] #empty list to store energy values        
        tVals = []
        while t<t_fin/2:        # loop for 'time' number of periods
            
        
            yl = rk4solver(threeb_RK4_derivs, t, yl, h)
            rr, vr = np.reshape(yl[:6],(3,2)), np.reshape(yl[6:], (3,2))
        
            EVals = threeb_Energy(rr, vr)
            E3body.append(EVals)
            tVals.append(t/T0)
        
            # you probably want to downsample if dt is too small (every cycle)
            if (ic % cycle == 0): # very simple animate (update data with pause)
                fig.suptitle("frame time {}".format(ic)) # show current time on graph

                Q1.remove()
                Q2.remove()
                Q3.remove()
                    
                ax2.plot(rr[0,0], rr[0,1], '.', color = 'purple')
                ax2.plot(rr[1,0], rr[1,1], '.', color = 'red')
                ax2.plot(rr[2,0], rr[2,1], '.', color = 'aqua')          
    
                ax1.plot(tVals,E3body, color = 'magenta')
                    
                Q1 = ax2.quiver(rr[0,0],rr[0,1],vr[0,0],vr[0,1], width = 0.008, color = 'purple', headwidth=6.,headlength=3.,headaxislength=3,scale=40)
                Q2 = ax2.quiver(rr[1,0],rr[1,1],vr[1,0],vr[1,1], width = 0.008, color = 'red', headwidth=6.,headlength=3.,headaxislength=3,scale=40)
                Q3 = ax2.quiver(rr[2,0],rr[2,1],vr[2,0],vr[2,1], width = 0.008, color = 'aqua', headwidth=6.,headlength=3.,headaxislength=3,scale=40)
            
                plt.draw() # may not be needed (depends on your set up)
                if (Animate == 0):    
                    plt.pause(tpause) # pause to see animation as code v. fast
                else:
                    plt.show()
            t = t + h
            ic = ic + 1
            
        while t>t_fin/2:
            yl = rk4solver(threeb_RK4_derivs, t, yl, h)
            rr, vr = np.reshape(yl[:6],(3,2)), np.reshape(yl[6:], (3,2))
            vrn = -vr
            rrn = -rr
            EVals = threeb_Energy(rrn, vrn)
            E3body.append(EVals)
            tVals.append(t/T0)
            
                # you probably want to downsample if dt is too small (every cycle)
            if (ic % cycle == 0): # very simple animate (update data with pause)
                fig.suptitle("frame time {}".format(ic)) # show current time on graph

                Q1.remove()
                Q2.remove()
                Q3.remove()
                        
                ax2.plot(rrn[0,0], rrn[0,1], '.', color = 'purple')
                ax2.plot(rrn[1,0], rrn[1,1], '.', color = 'red')
                ax2.plot(rrn[2,0], rrn[2,1], '.', color = 'aqua')          
        
                ax1.plot(tVals,E3body, color = 'magenta')
                        
                Q1 = ax2.quiver(rrn[0,0],rrn[0,1],vrn[0,0],vrn[0,1], width = 0.008, color = 'purple', headwidth=6.,headlength=3.,headaxislength=3,scale=40)
                Q2 = ax2.quiver(rrn[1,0],rrn[1,1],vrn[1,0],vrn[1,1], width = 0.008, color = 'red', headwidth=6.,headlength=3.,headaxislength=3,scale=40)
                Q3 = ax2.quiver(rrn[2,0],rrn[2,1],vrn[2,0],vrn[2,1], width = 0.008, color = 'aqua', headwidth=6.,headlength=3.,headaxislength=3,scale=40)
                
                plt.draw() # may not be needed (depends on your set up)
                if (Animate == 0):    
                    plt.pause(tpause) # pause to see animation as code v. fast
                else:
                    plt.show()
                # if t==t_fin:
                    # break
            t  = t + h # loop time
            ic = ic + 1 # simple integer counter that migth be useful 
            if t > t_fin:
                break
    else:
        
        rl, vl = r0, v0      # initial values

        E3body = [] #empty list to store energy values        
        tVals = []
        
        while t<t_fin/2:        # loop for 'time' number of periods
            
            
            # so your leapfrog or RK4 call could be here for example ...
            rl, vl = leapfrog(threeb_LF_derivs, rl, vl, t, h) # solve using leapfrog
        
            EVals = threeb_Energy(rl, vl)
            E3body.append(EVals)
            tVals.append(t/(T0))
        
            # you probably want to downsample if dt is too small (every cycle)
            if (ic % cycle == 0): # very simple animate (update data with pause)
                
                fig.suptitle("frame time {}".format(ic)) # show current time on graph
                    
                Q1.remove()
                Q2.remove()
                Q3.remove()

                ax2.plot(rl[0,0], rl[0,1], '.', color = 'purple')
                ax2.plot(rl[1,0], rl[1,1], '.', color = 'red')
                ax2.plot(rl[2,0], rl[2,1], '.', color = 'aqua')
            
                ax1.plot(tVals,E3body, color = 'magenta')
            
                Q1 = ax2.quiver(rl[0,0],rl[0,1],vl[0,0],vl[0,1], width = 0.008, color = 'purple', headwidth=6.,headlength=3.,headaxislength=3,scale=40)
                Q2 = ax2.quiver(rl[1,0],rl[1,1],vl[1,0],vl[1,1], width = 0.008, color = 'red', headwidth=6.,headlength=3.,headaxislength=3,scale=40)
                Q3 = ax2.quiver(rl[2,0],rl[2,1],vl[2,0],vl[2,1], width = 0.008, color = 'aqua', headwidth=6.,headlength=3.,headaxislength=3,scale=40)

                plt.draw() # may not be needed (depends on your set up)
                if (Animate == 0):    
                    plt.pause(tpause) # pause to see animation as code v. fast
                else:
                    plt.show()
            t  = t + h # loop time
            ic = ic + 1 # simple integer counter that migth be useful 
                
        while t>t_fin/2:
            # so your leapfrog or RK4 call could be here for example ...
            rl, vl = leapfrog(threeb_LF_derivs, rl, vl, t, h) # solve using leapfrog
                
            EVals = threeb_Energy(rl, -vl)
            E3body.append(EVals)
            tVals.append(t/(T0))
                
            # you probably want to downsample if dt is too small (every cycle)
            if (ic % cycle == 0): # very simple animate (update data with pause)
                        
                fig.suptitle("frame time {}".format(ic)) # show current time on graph
                            
                Q1.remove()
                Q2.remove()
                Q3.remove()

                ax2.plot(-rl[0,0], -rl[0,1], '.', color = 'purple')
                ax2.plot(-rl[1,0], -rl[1,1], '.', color = 'red')
                ax2.plot(-rl[2,0], -rl[2,1], '.', color = 'aqua')
            
                ax1.plot(tVals,E3body, color = 'magenta')
                    
                Q1 = ax2.quiver(-rl[0,0],-rl[0,1],-vl[0,0],-vl[0,1], width = 0.008, color = 'purple', headwidth=6.,headlength=3.,headaxislength=3,scale=40)
                Q2 = ax2.quiver(-rl[1,0],-rl[1,1],-vl[1,0],-vl[1,1], width = 0.008, color = 'red', headwidth=6.,headlength=3.,headaxislength=3,scale=40)
                Q3 = ax2.quiver(-rl[2,0],-rl[2,1],-vl[2,0],-vl[2,1], width = 0.008, color = 'aqua', headwidth=6.,headlength=3.,headaxislength=3,scale=40)
                        
                plt.draw() # may not be needed (depends on your set up)
                if (Animate == 0):    
                    plt.pause(tpause) # pause to see animation as code v. fast
                else:
                    plt.show()
            t  = t + h # loop time
            ic = ic + 1 # simple integer counter that migth be useful 
            if t > t_fin:
                break

"function for animation with parameters defined as:"
"go(*time, id, step size, update size, initial r, intitial v, animation)"
"""
time: number of periods for animation
id: 0 returns rk4 method, 1 uses leapfrog
stepsize: this is h
update size or 'cycle': how many frames per update
initial r: position starting conditions
intitial v: velocity starting conditions
animate: 1 returns the final trajectory, 0 returns animation 
"""
go(6.5,0, 0.001, 60, rinit, vinit, 0)


#%%
"(d)"
"Starting conditions will change with new masses so we calculate lambda again"

"Find root of interest of Euler's quintic equation with new set of masses"
#new mass values:
m1 = 1
m2 = 1
m3 = 1

#subroutine found on: https://personal.math.ubc.ca/~pwalls/math-python/roots-optimization/newton/
#Newtons method for root finding of a polynomial with the form f(x) = 0:
def newtonSolve(f,Df,x0,epsilon,max_iter):  #takes function (f) and derivative (Df)
    xn = x0    #begin the iteration at an initial guess
    for n in range(0,max_iter):   #iterate a number of times equal to max_iter
        fxn = f(xn)    
        if abs(fxn) < epsilon: #if the function evaluated at xn is less than the tolerance, we have found the solution
            print('Found solution after',n,'iterations.')
            return xn
        Dfxn = Df(xn) #if the derivative is never equal to zero near our initial guess, 
        if Dfxn == 0:  #print there are no solutions 
            print('Zero derivative. No solution found.')
            return None
        xn = xn - fxn/Dfxn
    print('Exceeded maximum iterations. No solution found.') #if the max iteration is reached without a solution
    return None

#quintic polynomial (f) and its derivative function (Df):
f = lambda x: x**5*(m2 + m3) + x**4*(2*m2 + 3*m3) + x**3*(m2 + 3*m3) + x**2*(-3*m1 - m2) + x*(-3*m1 - 2*m2) + (-m1 - m2)
Df = lambda x: 5*x**4*(m2 + m3) + 4*x**3*(2*m2 + 3*m3) + 3*x**2*(m2 + 3*m3) + 2*x*(-3*m1 - m2) + (-3*m1 - 2*m2)


print("Solution to f(x) = 0 is:",newtonSolve(f,Df,1,1e-8,10)) #intitial guess of l = 1

#create the lambda variable
l = newtonSolve(f,Df,1,1e-8,10)   

#%%
"New relative positions and velocities, including the perturbation from (c)"

delta = 1e-9   #small perturbation 
w = 1 + delta
a = ((1/(w**2))*(m2 + m3 - (m1*(1 + 2*l))/(l**2*(1+l)**2)))**(1/3) #find the distance 'a'
x2 = (1/(w**2*a**2))*(m1/(l**2) - m3)  #position of mass 2
x1 = x2 - l*a    #of mass 1
x3 = -(m1*x1 + m2*x2)/m3   #of mass3
v1y = w*x1    #intitial velocities
v2y = w*x2
v3y = w*x3


# Fixed initial condition (2)
def init_cond2 ():
    r, v = np.zeros((3,2)), np.zeros((3,2))
    r[0,0] = 0.97000436 ; r[0,1] = -0.24308753 # x1 , y1
    v[2,0] = -0.93240737 ; v[2,1] = -0.86473146 # v3x , v3y
    v[0,0] = -v[2,0]/2.; v[0,1] = -v[2,1]/2. # v1x , v1y
    r[1,0] = - r[0,0] ; r[1,1]=-r[0,1] # x2 , y2
    v[1,0] = v[0,0]; v[1,1] = v[0,1] # v2x , v2y
    return r , v

rinit2, vinit2 = init_cond2()   #retrieve arrays of initial r and v
#%%

"Animation function - We will choose to use RK4 to run question 4 (d)"
def go(time, id, h, cycle, r0, v0, Animate):
    ic=0
    niceFigure(False) #can delete false if you have TeX installed

    c1 = plt.Circle((0,0), 0.07, color = 'black', fill = False) #center of mass is at the origin
    c2 = plt.Circle((r0[0,0],0), 0.04, color = 'purple', fill = True) #starting position for mass 1
    c3 = plt.Circle((r0[1,0],0), 0.04, color = 'red', fill = True) #for mass 2
    c4 = plt.Circle((r0[2,0],0), 0.04, color = 'aqua', fill = True) #for mass 3
    
    fig, (ax1,ax2) = plt.subplots(2, figsize = (10,10), gridspec_kw={'height_ratios': [1,2]})
#    plt.ion()
    ax2.set_xlabel('$x$ (arb. units)')     # add labels
    ax2.set_ylabel('$y$ (arb. units)')
    ax1.set_xlabel('Time $(T_0)$')
    ax1.set_ylabel('E (arb. units)')
    
    #add the starting positions as filled circles
    ax2.add_patch(c1)    
    ax2.add_patch(c2)    
    ax2.add_patch(c3)    
    ax2.add_patch(c4)    

    #add temporary quivers
    Q1=plt.quiver(0,0,-1,width=0.008,color='b',headwidth=8.,headlength=3.,headaxislength=3,scale=40)    
    Q2=plt.quiver(0,0,-1,width=0.008,color='b',headwidth=8.,headlength=3.,headaxislength=3,scale=40)    
    Q3=plt.quiver(0,0,-1,width=0.008,color='b',headwidth=8.,headlength=3.,headaxislength=3,scale=40)    
    
    w_0 = w
    T0 = 2.*np.pi/w_0
    t_fin = time*T0
    t=0.
    tpause = 0.00001 # delay within animation 
    plt.pause(2) # pause for 2 seconds before start
    
    if (id==0):
        
        y_array = ([r0,v0])
        y_init = np.reshape(y_array,(12))
        yl = y_init  #rk4 initial conditions:

        E3body = [] #empty list to store energy values        
        tVals = []
        while t<t_fin:        # loop for 'time' number of periods
        
        # so your leapfrog or RK4 call could be here for example ...
        
            yl = rk4solver(threeb_RK4_derivs, t, yl, h)
            rr, vr = np.reshape(yl[:6],(3,2)), np.reshape(yl[6:], (3,2))
        
            EVals = threeb_Energy(rr, vr)
            E3body.append(EVals)
            tVals.append(t/T0)
        
            # you probably want to downsample if dt is too small (every cycle)
            if (ic % cycle == 0): # very simple animate (update data with pause)
                fig.suptitle("frame time {}".format(ic)) # show current time on graph

                Q1.remove()
                Q2.remove()
                Q3.remove()
                    
                ax2.plot(rr[0,0], rr[0,1], '.', color = 'purple')
                ax2.plot(rr[1,0], rr[1,1], '.', color = 'red')
                ax2.plot(rr[2,0], rr[2,1], '.', color = 'aqua')          
    
                ax1.plot(tVals,E3body, color = 'magenta')
            
                Q1 = ax2.quiver(rr[0,0],rr[0,1],vr[0,0],vr[0,1], width = 0.008, color = 'purple', headwidth=6.,headlength=3.,headaxislength=3,scale=40)
                Q2 = ax2.quiver(rr[1,0],rr[1,1],vr[1,0],vr[1,1], width = 0.008, color = 'red', headwidth=6.,headlength=3.,headaxislength=3,scale=40)
                Q3 = ax2.quiver(rr[2,0],rr[2,1],vr[2,0],vr[2,1], width = 0.008, color = 'aqua', headwidth=6.,headlength=3.,headaxislength=3,scale=40)
            
                plt.draw() # may not be needed (depends on your set up)
                if (Animate == 0):    
                    plt.pause(tpause) # pause to see animation as code v. fast
                else:
                    plt.show()
           
            t  = t + h # loop time
            ic = ic + 1 # simple integer counter that migth be useful 

    else:
        
        rl, vl = r0, v0      # initial values

        E3body = [] #empty list to store energy values        
        tVals = []
        while t<t_fin:        # loop for 'time' number of periods
            
            
            # so your leapfrog or RK4 call could be here for example ...
            rl, vl = leapfrog(threeb_LF_derivs, rl, vl, t, h) # solve using leapfrog
        
            EVals = threeb_Energy(rl, vl)
            E3body.append(EVals)
            tVals.append(t/(T0))
        
            # you probably want to downsample if dt is too small (every cycle)
            if (ic % cycle == 0): # very simple animate (update data with pause)
                fig.suptitle("frame time {}".format(ic)) # show current time on graph
                    
                Q1.remove()
                Q2.remove()
                Q3.remove()

                    
                ax2.plot(rl[0,0], rl[0,1], '.', color = 'purple')
                ax2.plot(rl[1,0], rl[1,1], '.', color = 'red')
                ax2.plot(rl[2,0], rl[2,1], '.', color = 'aqua')
            
                ax1.plot(tVals,E3body, color = 'magenta')
            
                Q1 = ax2.quiver(rl[0,0],rl[0,1],vl[0,0],vl[0,1], width = 0.008, color = 'purple', headwidth=6.,headlength=3.,headaxislength=3,scale=40)
                Q2 = ax2.quiver(rl[1,0],rl[1,1],vl[1,0],vl[1,1], width = 0.008, color = 'red', headwidth=6.,headlength=3.,headaxislength=3,scale=40)
                Q3 = ax2.quiver(rl[2,0],rl[2,1],vl[2,0],vl[2,1], width = 0.008, color = 'aqua', headwidth=6.,headlength=3.,headaxislength=3,scale=40)

                
                plt.draw() # may not be needed (depends on your set up)
                if (Animate == 0):    
                    plt.pause(tpause) # pause to see animation as code v. fast
                else:
                    plt.show()
            t  = t + h # loop time
            ic = ic + 1 # simple integer counter that migth be useful 

"function for animation with parameters defined as:"
"go(*time, id, step size, update size, initial r, intitial v, animation)"
"""
time: number of periods for animation
id: 0 returns rk4 method, 1 uses leapfrog
stepsize: this is h
update size or 'cycle': how many frames per update
initial r: position starting conditions
intitial v: velocity starting conditions
animate: 1 returns the final trajectory, 0 returns animation 
"""
go(4,0, 0.001, 60, rinit2, vinit2, 0)

#%%
"To test stability, we perturb mass 1 by a factor of 1e-6 and run simulation again"

"Find root of interest of Euler's quintic equation with new set of masses"
#new mass values:
m1 = 1 + delta
m2 = 1
m3 = 1

#subroutine found on: https://personal.math.ubc.ca/~pwalls/math-python/roots-optimization/newton/
#Newtons method for root finding of a polynomial with the form f(x) = 0:
def newtonSolve(f,Df,x0,epsilon,max_iter):  #takes function (f) and derivative (Df)
    xn = x0    #begin the iteration at an initial guess
    for n in range(0,max_iter):   #iterate a number of times equal to max_iter
        fxn = f(xn)    
        if abs(fxn) < epsilon: #if the function evaluated at xn is less than the tolerance, we have found the solution
            print('Found solution after',n,'iterations.')
            return xn
        Dfxn = Df(xn) #if the derivative is never equal to zero near our initial guess, 
        if Dfxn == 0:  #print there are no solutions 
            print('Zero derivative. No solution found.')
            return None
        xn = xn - fxn/Dfxn
    print('Exceeded maximum iterations. No solution found.') #if the max iteration is reached without a solution
    return None

#quintic polynomial (f) and its derivative function (Df):
f = lambda x: x**5*(m2 + m3) + x**4*(2*m2 + 3*m3) + x**3*(m2 + 3*m3) + x**2*(-3*m1 - m2) + x*(-3*m1 - 2*m2) + (-m1 - m2)
Df = lambda x: 5*x**4*(m2 + m3) + 4*x**3*(2*m2 + 3*m3) + 3*x**2*(m2 + 3*m3) + 2*x*(-3*m1 - m2) + (-3*m1 - 2*m2)


print("Solution to f(x) = 0 is:",newtonSolve(f,Df,1,1e-8,10)) #intitial guess of l = 1


l = newton(f, 1) #this seems to be slightly more accurate and sensitive to the perturbated mass

#create the lambda variable
# l = newtonSolve(f, Df, 1, 1e-8, 10)

#%%
"Relative positions and velocities will change slightly with a new m1:"

delta = 1e-9   #small perturbation 
w = 1 + delta
a = ((1/(w**2))*(m2 + m3 - (m1*(1 + 2*l))/(l**2*(1+l)**2)))**(1/3) #find the distance 'a'
x2 = (1/(w**2*a**2))*(m1/(l**2) - m3)  #position of mass 2
x1 = x2 - l*a    #of mass 1
x3 = -(m1*x1 + m2*x2)/m3   #of mass3
v1y = w*x1    #intitial velocities
v2y = w*x2
v3y = w*x3


# Fixed initial condition (2)
def init_cond2 ():
    r, v = np.zeros((3,2)), np.zeros((3,2))
    r[0,0] = 0.97000436 ; r[0,1] = -0.24308753 # x1 , y1
    v[2,0] = -0.93240737 ; v[2,1] = -0.86473146 # v3x , v3y
    v[0,0] = -v[2,0]/2.; v[0,1] = -v[2,1]/2. # v1x , v1y
    r[1,0] = - r[0,0] ; r[1,1]=-r[0,1] # x2 , y2
    v[1,0] = v[0,0]; v[1,1] = v[0,1] # v2x , v2y
    return r , v

rinit2, vinit2 = init_cond2()   #retrieve arrays of initial r and v
#%%
"Finally, running the simulation again but for 8 periods (or time = 8)"

"Animation function - We will choose to use RK4 to run question 4 (d)"
def go(time, id, h, cycle, r0, v0, Animate):
    ic=0
    niceFigure(False) #can delete false if you have TeX installed

    c1 = plt.Circle((0,0), 0.07, color = 'black', fill = False) #center of mass is at the origin
    c2 = plt.Circle((r0[0,0],0), 0.04, color = 'purple', fill = True) #starting position for mass 1
    c3 = plt.Circle((r0[1,0],0), 0.04, color = 'red', fill = True) #for mass 2
    c4 = plt.Circle((r0[2,0],0), 0.04, color = 'aqua', fill = True) #for mass 3
    
    fig, (ax1,ax2) = plt.subplots(2, figsize = (10,10), gridspec_kw={'height_ratios': [1,2]})
#    plt.ion()
    ax2.set_xlabel('$x$ (arb. units)')     # add labels
    ax2.set_ylabel('$y$ (arb. units)')
    ax1.set_xlabel('Time $(T_0)$')
    ax1.set_ylabel('E (arb. units)')
    
    #add the starting positions as filled circles
    ax2.add_patch(c1)    
    ax2.add_patch(c2)    
    ax2.add_patch(c3)    
    ax2.add_patch(c4)    

    #add temporary quivers
    Q1=plt.quiver(0,0,-1,width=0.008,color='b',headwidth=8.,headlength=3.,headaxislength=3,scale=40)    
    Q2=plt.quiver(0,0,-1,width=0.008,color='b',headwidth=8.,headlength=3.,headaxislength=3,scale=40)    
    Q3=plt.quiver(0,0,-1,width=0.008,color='b',headwidth=8.,headlength=3.,headaxislength=3,scale=40)    
    
    w_0 = w
    T0 = 2.*np.pi/w_0
    t_fin = time*T0
    t=0.
    tpause = 0.00001 # delay within animation 
    plt.pause(2) # pause for 2 seconds before start
    
    if (id==0):
        
        y_array = ([r0,v0])
        y_init = np.reshape(y_array,(12))
        yl = y_init  #rk4 initial conditions:

        E3body = [] #empty list to store energy values        
        tVals = []
        while t<t_fin:        # loop for 'time' number of periods
        
        # so your leapfrog or RK4 call could be here for example ...
        
            yl = rk4solver(threeb_RK4_derivs, t, yl, h)
            rr, vr = np.reshape(yl[:6],(3,2)), np.reshape(yl[6:], (3,2))
        
            EVals = threeb_Energy(rr, vr)
            E3body.append(EVals)
            tVals.append(t/T0)
        
            # you probably want to downsample if dt is too small (every cycle)
            if (ic % cycle == 0): # very simple animate (update data with pause)
                fig.suptitle("frame time {}".format(ic)) # show current time on graph

                Q1.remove()
                Q2.remove()
                Q3.remove()
                    
                ax2.plot(rr[0,0], rr[0,1], '.', color = 'purple')
                ax2.plot(rr[1,0], rr[1,1], '.', color = 'red')
                ax2.plot(rr[2,0], rr[2,1], '.', color = 'aqua')          
    
                ax1.plot(tVals,E3body, color = 'magenta')
            
                Q1 = ax2.quiver(rr[0,0],rr[0,1],vr[0,0],vr[0,1], width = 0.008, color = 'purple', headwidth=6.,headlength=3.,headaxislength=3,scale=40)
                Q2 = ax2.quiver(rr[1,0],rr[1,1],vr[1,0],vr[1,1], width = 0.008, color = 'red', headwidth=6.,headlength=3.,headaxislength=3,scale=40)
                Q3 = ax2.quiver(rr[2,0],rr[2,1],vr[2,0],vr[2,1], width = 0.008, color = 'aqua', headwidth=6.,headlength=3.,headaxislength=3,scale=40)
            
                plt.draw() # may not be needed (depends on your set up)
                if (Animate == 0):    
                    plt.pause(tpause) # pause to see animation as code v. fast
                else:
                    plt.show()
           
            t  = t + h # loop time
            ic = ic + 1 # simple integer counter that migth be useful 

    else:
        
        rl, vl = r0, v0      # initial values

        E3body = [] #empty list to store energy values        
        tVals = []
        while t<t_fin:        # loop for 'time' number of periods
            
            
            # so your leapfrog or RK4 call could be here for example ...
            rl, vl = leapfrog(threeb_LF_derivs, rl, vl, t, h) # solve using leapfrog
        
            EVals = threeb_Energy(rl, vl)
            E3body.append(EVals)
            tVals.append(t/(T0))
        
            # you probably want to downsample if dt is too small (every cycle)
            if (ic % cycle == 0): # very simple animate (update data with pause)
                fig.suptitle("frame time {}".format(ic)) # show current time on graph
                    
                Q1.remove()
                Q2.remove()
                Q3.remove()

                    
                ax2.plot(rl[0,0], rl[0,1], '.', color = 'purple')
                ax2.plot(rl[1,0], rl[1,1], '.', color = 'red')
                ax2.plot(rl[2,0], rl[2,1], '.', color = 'aqua')
            
                ax1.plot(tVals,E3body, color = 'magenta')
            
                Q1 = ax2.quiver(rl[0,0],rl[0,1],vl[0,0],vl[0,1], width = 0.008, color = 'purple', headwidth=6.,headlength=3.,headaxislength=3,scale=40)
                Q2 = ax2.quiver(rl[1,0],rl[1,1],vl[1,0],vl[1,1], width = 0.008, color = 'red', headwidth=6.,headlength=3.,headaxislength=3,scale=40)
                Q3 = ax2.quiver(rl[2,0],rl[2,1],vl[2,0],vl[2,1], width = 0.008, color = 'aqua', headwidth=6.,headlength=3.,headaxislength=3,scale=40)

                
                plt.draw() # may not be needed (depends on your set up)
                if (Animate == 0):    
                    plt.pause(tpause) # pause to see animation as code v. fast
                else:
                    plt.show()
            t  = t + h # loop time
            ic = ic + 1 # simple integer counter that migth be useful 

"function for animation with parameters defined as:"
"go(*time, id, step size, update size, initial r, intitial v, animation)"
"""
time: number of periods for animation
id: 0 returns rk4 method, 1 uses leapfrog
stepsize: this is h
update size or 'cycle': how many frames per update
initial r: position starting conditions
intitial v: velocity starting conditions
animate: 1 returns the final trajectory, 0 returns animation 
"""
go(8,0, 0.001, 60, rinit2, vinit2, 0)

