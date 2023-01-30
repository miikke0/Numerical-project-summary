# -*- coding: utf-8 -*-
"""
Calculation of the matrices for Lab 2: Optical Pumping
@author: Michael Jafs
20108572
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as cst
import timeit
import scipy.sparse as sparse


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


# def clear_variables(tf):
#     """
#     Simple command which clears memory of specified relevant initial condition variables. Argument 
#     can take True or False which either does or doesn't clear memory of specified variables.
#     """
#     if tf == True:
#         try:
#             global F1, F2, F3, F4, F5, F6, F7, F8, F9, F10
#             del   F1, F2, F3, F4, F5, F6, F7, F8, F9, F10
#         except NameError:
#             pass
#     elif tf == False:
#         pass
#%%


"Empty matrices to store values in"
M1 = np.zeros((8,8)) #matrix inlcuding elements for H_B
M2 = np.zeros((8,8)) #matrix inlcuding elements for H_hf (S+,I-)
M3 = np.zeros((8,8)) #matrix inlcuding elements for H_B (S-,I+)
M4 = np.zeros((8,8)) #matrix inlcuding elements for H_B (S_z,I_z)

"Constants"
# uB = e*hbar/(2*m_e)
uB = 0.0013996256      #bohr magneton in units of MHz/G
# B0 = 1
# print(uB)
gs = 2.0023193043622   #electron spin g-factor
gI87 = -0.0009951414     #Nuclear g-factor for Rb 87
gI85 = -0.00029364000    #Nuclear g-factor for Rb 85
a_hf87 = 3.417341305452145  #a_hf value (divided by h) from line data for 87 Rb
a_hf85 = 1.011910813       #for 85 rubidium
#%%

"Matrix element functions"
def m1(ms,mI,B0):
    return uB*B0*(gs*ms + gI87*mI)

def m2(S, mS, I, mI):
    return np.sqrt((S  - mS)*(S + mS + 1)*(I + mI)*(I - mI + 1))

def m3(S, mS, I, mI):
    return np.sqrt((S  + mS)*(S - mS + 1)*(I - mI)*(I + mI + 1))

def m4(mS, mI):
    return mS*mI


# "Matrix element functions"
# def m1(ms,mI,B0):
#     return uB*B0*(gs*ms + gI87*mI)

# def m2(S, mS, I, mI):
#     return hbar**2*np.sqrt((S  - mS)*(S + mS + 1)*(I + mI)*(I - mI + 1))

# def m3(S, mS, I, mI):
#     return hbar**2*np.sqrt((S  + mS)*(S - mS + 1)*(I - mI)*(I + mI + 1))

# def m4(mS, mI):
#     return hbar**2*mS*mI
#%%

"Fill the matrix elements"

"for the first matrix"
# M1[0,0] = m1(-1/2,-3/2)
# M1[1,1] = m1(-1/2,-1/2)
# M1[2,2] = m1(-1/2,1/2)
# M1[3,3] = m1(-1/2,3/2)
# M1[4,4] = m1(1/2,-3/2)
# M1[5,5] = m1(1/2,-1/2)
# M1[6,6] = m1(1/2,1/2)
# M1[7,7] = m1(1/2,3/2)

"the second"
M2[1,4] = m2(1/2,-1/2,3/2,-1/2)
M2[2,5] = m2(1/2,-1/2,3/2,1/2)
M2[3,6] = m2(1/2,-1/2,3/2,3/2)

"the third"
M3[4,1] = m3(1/2,1/2,3/2,-3/2)
M3[5,2] = m3(1/2,1/2,3/2,-1/2)
M3[6,3] = m3(1/2,1/2,3/2,1/2)

"the fourth"
M4[0,0] = m4(-1/2,-3/2)
M4[1,1] = m4(-1/2,-1/2)
M4[2,2] = m4(-1/2,1/2)
M4[3,3] = m4(-1/2,3/2)
M4[4,4] = m4(1/2,-3/2)
M4[5,5] = m4(1/2,-1/2)
M4[6,6] = m4(1/2,1/2)
M4[7,7] = m4(1/2,3/2)

"full hamiltonian"
# Mtot = M1 + M2 + M3 + M4
Min = a_hf87*(1/2*(M2 + M3) + M4)
#%%

"Diagonalize the full hamiltonian"
# Eig = np.linalg.eig(Mtot)    #find the eigenenergies

Bvals = np.linspace(0,2500,100)
# Bvals = np.ones(len(Bval))

E = []
for B in Bvals:
    
    M1[0,0] = m1(-1/2,-3/2, B)
    M1[1,1] = m1(-1/2,-1/2, B)
    M1[2,2] = m1(-1/2,1/2, B)
    M1[3,3] = m1(-1/2,3/2, B)
    M1[4,4] = m1(1/2,-3/2, B)
    M1[5,5] = m1(1/2,-1/2, B)
    M1[6,6] = m1(1/2,1/2, B)
    M1[7,7] = m1(1/2,3/2, B)
    
    M = M1 + Min
    Marray = np.array(M)
    EigE, EigV = np.linalg.eig(Marray)
    E.append(EigE)
    Earray = np.array(E)
    

for i in range(8):
    plt.plot(Bvals,Earray[:,i], '.', markersize = '3', label = "{}".format(i))
    niceFigure(True)
    plt.xlabel("Magnetic Field Srength (Gauss)")
    plt.ylabel("Energy/h (MHz)")
    plt.legend()

plt.grid()
plt.xticks([0,500,1000,1500,2000])
plt.yticks([-5,-2.5,0,2.5,5])
plt.savefig('splitting 87.jpg', format='jpg', dpi=1200,bbox_inches = 'tight')

# print(earray[:,0])    

#%%

"Transition Frequency plots"



F1 = np.zeros(len(Bvals))   #create some empty arrays to fill below
F2 = np.zeros(len(Bvals))
F3 = np.zeros(len(Bvals))
F4 = np.zeros(len(Bvals))
F5 = np.zeros(len(Bvals))
F6 = np.zeros(len(Bvals))


"Subtract eigen-energies to find the transition frequencies"
#upper half:
F1 = Earray[:,7] - Earray[:,0]  #transition frequencies
F2 = Earray[:,0] - Earray[:,3]  #transition frequencies
F3a = Earray[:49,3] - Earray[:49,4]  #transition frequencies
F3b = Earray[49:,3] - Earray[49:,5]  #transition frequencies
F3[:49] = F3a
F3[49:] = F3b
F4a = Earray[:49,4] - Earray[:49,6]  #transition frequencies
F4b = Earray[49:,5] - Earray[49:,6]  #transition frequencies
F4[:49] = F4a
F4[49:] = F4b
F5a = Earray[:49,5] - Earray[:49,2]  #transition frequencies
F5b = Earray[49:,4] - Earray[49:,2]  #transition frequencies
F5[:49] = F5a
F5[49:] = F5b
F6 = Earray[:,2] - Earray[:,1]  #transition frequencies


#plot the transition frequencies
plt.plot(Bvals,F1,color = 'red',linewidth = 1)
plt.plot(Bvals,F2,color = 'red',linewidth = 1)
plt.plot(Bvals,F3,color = 'red',linewidth = 1)
plt.plot(Bvals,F4,color = 'red',linewidth = 1)
plt.plot(Bvals,F5,color = 'red',linewidth = 1)
plt.plot(Bvals,F6,color = 'red',linewidth = 1)

plt.grid()
plt.xticks([0,400,800,1200,1600,2000,2400])
plt.yticks([0,1,2,3,4])

plt.xlabel("B (Gauss)")
plt.ylabel("RF Transition Frequency (Ghz)")

plt.savefig('rf frequencies(87).jpg', format = 'jpg', dpi = 1200, bbox_inches = 'tight')
plt.show()


#%%


"""
Now for rubidium 85... Only the matrix elements should change so this is the 
part we alter here. 

"""

"Empty matrices to store values in"
M1 = np.zeros((12,12)) #matrix inlcuding elements for H_B
M2 = np.zeros((12,12)) #matrix inlcuding elements for H_hf (S+,I-)
M3 = np.zeros((12,12)) #matrix inlcuding elements for H_B (S-,I+)
M4 = np.zeros((12,12)) #matrix inlcuding elements for H_B (S_z,I_z)

"Fill the matrix elements"

"for the first matrix"
# M1[0,0] = m1(-1/2,-3/2)
# M1[1,1] = m1(-1/2,-1/2)
# M1[2,2] = m1(-1/2,1/2)
# M1[3,3] = m1(-1/2,3/2)
# M1[4,4] = m1(1/2,-3/2)
# M1[5,5] = m1(1/2,-1/2)
# M1[6,6] = m1(1/2,1/2)
# M1[7,7] = m1(1/2,3/2)

"the second"
M2[1,6] = m2(1/2,-1/2,5/2,-3/2)
M2[2,7] = m2(1/2,-1/2,5/2,-1/2)
M2[3,8] = m2(1/2,-1/2,5/2,1/2)
M2[4,9] = m2(1/2,-1/2,5/2,3/2)
M2[5,10] = m2(1/2,-1/2,5/2,5/2)

"the third"
M3[6,1] = m3(1/2,1/2, 5/2,-5/2)
M3[7,2] = m3(1/2,1/2,5/2,-3/2)
M3[8,3] = m3(1/2,1/2,5/2,-1/2)
M3[9,4] = m3(1/2,1/2,5/2,1/2)
M3[10,5] = m3(1/2,1/2,5/2,3/2)

"the fourth"
M4[0,0] = m4(-1/2,-5/2)
M4[1,1] = m4(-1/2,-3/2)
M4[2,2] = m4(-1/2,-1/2)
M4[3,3] = m4(-1/2,1/2)
M4[4,4] = m4(-1/2,3/2)
M4[5,5] = m4(-1/2,5/2)
M4[6,6] = m4(1/2,-5/2)
M4[7,7] = m4(1/2,-3/2)
M4[8,8] = m4(1/2,-1/2)
M4[9,9] = m4(1/2,1/2)
M4[10,10] = m4(1/2,3/2)
M4[11,11] = m4(1/2,5/2)

"full hamiltonian"
# Mtot = M1 + M2 + M3 + M4
Min = a_hf85*(1/2*(M2 + M3) + M4)
#%%

"Diagonalize the full hamiltonian"
# Eig = np.linalg.eig(Mtot)    #find the eigenenergies

Bvals = np.linspace(0,2500,100)
# Bvals = np.ones(len(Bval))

En = []
for B in Bvals:
    
    M1[0,0] = m1(-1/2,-5/2, B)
    M1[1,1] = m1(-1/2,-3/2, B)
    M1[2,2] = m1(-1/2,-1/2, B)
    M1[3,3] = m1(-1/2, 1/2, B)
    M1[4,4] = m1(-1/2, 3/2, B)
    M1[5,5] = m1(-1/2, 5/2, B)
    M1[6,6] = m1(1/2,-5/2, B)
    M1[7,7] = m1(1/2,-3/2, B)
    M1[8,8] = m1(1/2,-1/2, B)
    M1[9,9] = m1(1/2, 1/2, B)
    M1[10,10] = m1(1/2,3/2, B)
    M1[11,11] = m1(1/2,5/2, B)

    Mn = M1 + Min
    Mnarray = np.array(Mn)
    EigEn, EigVn = np.linalg.eig(Mnarray)
    EigEn = EigEn.real
    En.append(EigEn)
    Earrayn = np.array(En)
    

for i in range(12):
    plt.plot(Bvals,Earrayn[:,i], '.', markersize = '3',color = 'blue', label = "Eigen-energy {}".format(i))
    niceFigure(True)
    plt.xlabel("Magnetic Field Srength (Gauss)")
    plt.ylabel("Energy/h (GHz)")
    # plt.legend(fontsize = 13)

plt.xticks([0,500,1000,1500,2000])
plt.yticks([-5,-2.5,0,2.5,5])
plt.grid()
plt.savefig('splitting 85.jpg', format='jpg', dpi=1200,bbox_inches = 'tight')
print(Earrayn[:,2])
# print(earray[:,0])    


#%%

array = [1,2,3,4,5,6,7]
a1=array[1:]
print(a1)



#%%
"Transition Frequency plots"



F1 = np.zeros(len(Bvals))   #create some empty arrays to fill below
F2 = np.zeros(len(Bvals))
F3 = np.zeros(len(Bvals))
F4 = np.zeros(len(Bvals))
F5 = np.zeros(len(Bvals))
F6 = np.zeros(len(Bvals))
F7 = np.zeros(len(Bvals))
F8 = np.zeros(len(Bvals))
F9 = np.zeros(len(Bvals))
F10 = np.zeros(len(Bvals))

"Subtract eigen-energies to find the transition frequencies"
#upper half:
F1 = Earrayn[:,11] - Earrayn[:,0]  #transition frequencies
F2 = Earrayn[:,0] - Earrayn[:,5]
F3 = Earrayn[:,5] - Earrayn[:,8]
F4a = Earrayn[:15,8] - Earrayn[:15,2]  #calc frequencies that split weird from our algorithm seperately
F4b = Earrayn[15:,8] - Earrayn[15:,3]
F4[:15] = F4a   #stitch together array with seperate pieces above
F4[15:] = F4b
F5a = Earrayn[:15,2] - Earrayn[:15,7]
F5b = Earrayn[15:29,3] - Earrayn[15:29,7]
F5c = Earrayn[29:,3] - Earrayn[29:,6]
F5[:15] = F5a
F5[15:29] = F5b
F5[29:] = F5c
F6a = Earrayn[:29,7] - Earrayn[:29,10]
F6b = Earrayn[29:,6] - Earrayn[29:,10]
F6[:29] = F6a
F6[29:] = F6b

#lower half:
F7a = Earrayn[:15,6] - Earrayn[:15,3]
F7b = Earrayn[15:29,6] - Earrayn[15:29,2]
F7c = Earrayn[29:,7] - Earrayn[29:,2]
F7[:15] = F7a
F7[15:29] = F7b
F7[29:] = F7c
F8a = Earrayn[:15,3] - Earrayn[:15,9]
F8b = Earrayn[15:,2] - Earrayn[15:,9]
F8[:15] = F8a
F8[15:] = F8b
F9 = Earrayn[:,9] - Earrayn[:,4]
F10 = Earrayn[:,4] - Earrayn[:,1]


#plot the transition frequencies
plt.plot(Bvals,F1,color = 'blue',linewidth = 1)
plt.plot(Bvals,F2,color = 'blue',linewidth = 1)
plt.plot(Bvals,F3,color = 'blue',linewidth = 1)
plt.plot(Bvals,F4,color = 'blue',linewidth = 1)
plt.plot(Bvals,F5,color = 'blue',linewidth = 1)
plt.plot(Bvals,F6,color = 'blue',linewidth = 1)
plt.plot(Bvals,F7,color = 'blue',linewidth = 1)
plt.plot(Bvals,F8,color = 'blue',linewidth = 1)
plt.plot(Bvals,F9,color = 'blue',linewidth = 1)
plt.plot(Bvals,F10,color = 'blue',linewidth = 1)
plt.grid()
plt.xticks([0,400,800,1200,1600,2000,2400])
plt.yticks([0,1,2,3,4])

plt.xlabel("B (Gauss)")
plt.ylabel("RF Transition Frequency (Ghz)")

plt.savefig('rf frequencies(85).jpg', format = 'jpg', dpi = 1200, bbox_inches = 'tight')
plt.show()



#%%




#%%
"explore the transition frequencies in the weak field regime"

Bvals = np.linspace(0,2500,100)

# Bvals1 = np.linspace(0,10,100)
# plt.plot(Bvals1,F1[:100], label = "")
# # plt.plot(Bvals1,F2)
# # plt.plot(Bvals1,F3)
# # plt.plot(Bvals1,F4)
# # plt.plot(Bvals1,F5)
# # plt.plot(Bvals1,F6)
# # plt.plot(Bvals1,F7)
# # plt.plot(Bvals1,F8)
# # plt.plot(Bvals1,F9)
# # plt.plot(Bvals1,F10)
# # plt.xticks([0,400,800,1200,1600,2000,2400])
# # plt.yticks([0,1,2,3,4])

# plt.xlabel("B (Gauss)")
# plt.ylabel("RF Transition Frequency (Ghz)")

# plt.savefig('rf frequencies(85).pdf', format = 'pdf', dpi = 1200, bbox_inches = 'tight')
# plt.show()


"Diagonalize the full hamiltonian"
# Eig = np.linalg.eig(Mtot)    #find the eigenenergies

Bvals1 = np.linspace(0,10,100)
# Bvals = np.ones(len(Bval))

En = []
for B in Bvals1:
    
    M1[0,0] = m1(-1/2,-5/2, B)
    M1[1,1] = m1(-1/2,-3/2, B)
    M1[2,2] = m1(-1/2,-1/2, B)
    M1[3,3] = m1(-1/2, 1/2, B)
    M1[4,4] = m1(-1/2, 3/2, B)
    M1[5,5] = m1(-1/2, 5/2, B)
    M1[6,6] = m1(1/2,-5/2, B)
    M1[7,7] = m1(1/2,-3/2, B)
    M1[8,8] = m1(1/2,-1/2, B)
    M1[9,9] = m1(1/2, 1/2, B)
    M1[10,10] = m1(1/2,3/2, B)
    M1[11,11] = m1(1/2,5/2, B)

    Mn = M1 + Min
    Mnarray = np.array(Mn)
    EigEn, EigVn = np.linalg.eig(Mnarray)
    EigEn = EigEn.real
    En.append(EigEn)
    Earrayn = np.array(En)
    

for i in range(12):
    plt.plot(Bvals,Earrayn[:,i], '.', markersize = '4', label = "Eigen-energy {}".format(i))
    niceFigure(True)
    plt.xlabel("Magnetic Field Srength (Gauss)")
    plt.ylabel("Energy/h (GHz)")
    plt.legend(fontsize = 13)

plt.savefig('splitting 85(weakfield).pdf', format='pdf', dpi=1200,bbox_inches = 'tight')
print(Earrayn[:,2])
# print(earray[:,0])    
plt.show()

#%%
"Transition Frequency plots"

F1 = np.zeros(len(Bvals))   #create some empty arrays to fill below
F2 = np.zeros(len(Bvals))
F3 = np.zeros(len(Bvals))
F4 = np.zeros(len(Bvals))
F5 = np.zeros(len(Bvals))
F6 = np.zeros(len(Bvals))
F7 = np.zeros(len(Bvals))
F8 = np.zeros(len(Bvals))
F9 = np.zeros(len(Bvals))
F10 = np.zeros(len(Bvals))

"Subtract eigen-energies to find the transition frequencies"
#upper half:
F1 = Earrayn[:,11] - Earrayn[:,0]  #transition frequencies
F2 = Earrayn[:,0] - Earrayn[:,5]
F3 = Earrayn[:,5] - Earrayn[:,8]
F4 = Earrayn[:,8] - Earrayn[:,2]  #calc frequencies that split weird from our algorithm seperately
# F4b = Earrayn[15:,8] - Earrayn[15:,3]
# F4[:15] = F4a   #stitch together array with seperate pieces above
# F4[15:] = F4b
F5 = Earrayn[:,2] - Earrayn[:,7]
# F5b = Earrayn[15:29,3] - Earrayn[15:29,7]
# F5c = Earrayn[29:,3] - Earrayn[29:,6]
# F5[:15] = F5a
# F5[15:29] = F5b
# F5[29:] = F5c
F6 = Earrayn[:,7] - Earrayn[:,10]
# F6b = Earrayn[29:,6] - Earrayn[29:,10]
# F6[:29] = F6a
# F6[29:] = F6b

# #lower half:
F7 = Earrayn[:,6] - Earrayn[:,3]
# F7b = Earrayn[15:29,6] - Earrayn[15:29,2]
# F7c = Earrayn[29:,7] - Earrayn[29:,2]
# F7[:15] = F7a
# F7[15:29] = F7b
# F7[29:] = F7c
F8 = Earrayn[:,3] - Earrayn[:,9]
# F8b = Earrayn[15:,2] - Earrayn[15:,9]
# F8[:15] = F8a
# F8[15:] = F8b
F9 = Earrayn[:,9] - Earrayn[:,4]
F10 = Earrayn[:,4] - Earrayn[:,1]


#plot the transition frequencies
plt.plot(Bvals1,F1,color = 'blue')
plt.plot(Bvals1,F2,color = 'blue')
plt.plot(Bvals1,F3,color = 'blue')
plt.plot(Bvals1,F4,color = 'blue')
plt.plot(Bvals1,F5,color = 'blue')
plt.plot(Bvals1,F6,color = 'blue')
plt.plot(Bvals1,F7,color = 'blue')
plt.plot(Bvals1,F8,color = 'blue')
plt.plot(Bvals1,F9,color = 'blue')
plt.plot(Bvals1,F10,color = 'blue')
plt.grid()
# plt.xticks([0,400,800,1200,1600,2000,2400])
# plt.yticks([0,1,2,3,4])

plt.xlabel("B (Gauss)")
plt.ylabel("RF Transition Frequency (Ghz)")

plt.savefig('rf frequencies(85)init.jpg', format = 'jpg', dpi = 1200, bbox_inches = 'tight')
plt.show()

#%%
"Simulation of optical pumping"

def rk4(f,y0,t0,h):
    # four equations defined in fourth order Runge-kutta 
    k1 = h * np.asarray(f(y0,t0)) 
    k2 = h* np.asarray(f(y0+(k1/2),t0+(h/2))) 
    k3 = h* np.asarray(f(y0+(k2/2),t0+(h/2)))
    k4 =h* np.asarray(f(y0+k3,t0+h))
    #y value is now approximated using the RK formulas
    y=y0 + (1.0 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4)
    y0=y
    #x0 is now updated using the step size 
    t0= t0+h
    return y


def ODE_rate(y,t): # derivatives function 

    G =y[:-8]    #ground states
    E =y[-8:]    #excited states
    

    dy=np.zeros(len(y)) 
    
    C=np.zeros((8,8))
    C[0]= np.array([1/12,1/12,0,1/2,1/4,1/12,0,0])
    C[1]= np.array([1/12,0,1/12,0,1/4,1/3,1/4,0])
    C[2]= np.array([0,1/12,1/12,0,0,1/12,1/4,1/2])
    C[3]= np.array([1/2,0,0,1/3,1/6,0,0,0])
    C[4]= np.array([1/4,1/4,0,1/6,1/12,1/4,0,0])
    C[5]= np.array([1/12,1/3,1/12,0,1/4,0,1/4,0])
    C[6]= np.array([0,1/4,1/4,0,0,1/4,1/12,1/6])
    C[7]= np.array([0,0,1/2,0,0,0,1/6,1/3])
    
  


    diag = np.ones(8)                                  # values that will go on main diagonal     
    data = [diag, diag, diag]                          # list of all the data
    diags = [-3,1,5]                                    # which diagonal each vector goes into
    F = sparse.spdiags(data,diags,8,8,format='csc')  # create the matrix
    F= F.todense()   
    
    Eng=3
    w= 2*np.pi*3.77e14
    m=1                   #unknown atm so ser to 1 
    tau= 2.768e-8  #in seconds 
    const = (6*np.pi * cst.c**2)/(cst.hbar*w)
    I=cst.c*m*Eng**2/2
    P = const*I*F@C
    
    R= (1/tau)*F
    G1=G[0]
    G2=G[1]
    G3=G[2]
    G4=G[3]
    G5=G[4]
    G6=G[5]
    G7=G[6]
    G8=G[7]            
   
                 
    E1=E[0]
    E2=E[1]
    E3=E[2]
    E4=E[3]
    E5=E[4]
    E6=E[5]
    E7=E[6]
    E8=E[7]    
   
    P1=P[0]
    P2=P[1]
    P3=P[2]             
    P4=P[3]
    P5=P[4]
    P6=P[5]
    P7=P[6]
    P8=P[7]  
   
    # P_1=P[0,:]
    # P_2=P[1,:]
    # P_3=P[2,:]             
    # P_4=P[3,:]
    # P_5=P[4,:]
    # P_6=P[5,:]
    # P_7=P[6,:]
    # P_8=P[7,:]  
   
    R1=R[0]
    R2=R[1]
    R3= R[2]             
    R4=R[3]
    R5= R[4]
    R6= R[5]
    R7=R[6]
    R8= R[7]
   
    # R_1=R[0,:]
    # R_2=R[1,:]
    # R_3= R[2,:]             
    # R_4=R[3,:]
    # R_5= R[4,:]
    # R_6= R[5,:]
    # R_7=R[6,:]
    # R_8= R[7,:]
    
    
    dy[0] = (P1@ np.reshape((E-G1),(8,1))) + R1@ E
    dy[1] = (P2@ np.reshape((E-G2),(8,1)))+ R2@ E
    dy[2] = (P3@ np.reshape((E-G3),(8,1)))+ R3@ E
    dy[3] = (P4@ np.reshape((E-G4),(8,1)))+ R4@ E
    dy[4] = (P5@ np.reshape((E-G5),(8,1))) +R5@ E
    dy[5] = (P6@ np.reshape((E-G6),(8,1)))+ R6@ E
    dy[6] = (P7@ np.reshape((E-G7),(8,1)))+ R7@ E
    dy[7] = (P8@ np.reshape((E-G8),(8,1)))+ R8@ E
    
    
    dy[8] = (P1@ np.reshape((G-E1),(8,1)))- R1@ G
    dy[9] = (P2@ np.reshape((G-E2),(8,1)))-  R2@ G
    dy[10] = (P3@ np.reshape((G-E3),(8,1)))- R3@ G
    dy[11] = (P4@ np.reshape((G-E4),(8,1)))- R4@ G
    dy[12] = (P5@ np.reshape((G-E5),(8,1)))- R5@ G
    dy[13] = (P6@ np.reshape((G-E6),(8,1)))- R6@ G
    dy[14] = (P7@ np.reshape((G-E7),(8,1)))- R7@ G
    dy[15] = (P8@ np.reshape((G-E8),(8,1)))- R8@ G
    
    return dy


C=np.zeros((8,8))
C[0]= np.array([1/12,1/12,0,1/2,1/4,1/12,0,0])
C[1]= np.array([1/12,0,1/12,0,1/4,1/3,1/4,0])
C[2]= np.array([0,1/12,1/12,0,0,1/12,1/4,1/2])
C[3]= np.array([1/2,0,0,1/3,1/6,0,0,0])
C[4]= np.array([1/4,1/4,0,1/6,1/12,1/4,0,0])
C[5]= np.array([1/12,1/3,1/12,0,1/4,0,1/4,0])
C[6]= np.array([0,1/4,1/4,0,0,1/4,1/12,1/6])
C[7]= np.array([0,0,1/2,0,0,0,1/6,1/3])
print(C)

# diag1= np.array([1/12,1/4,1/2])
# diag2= np.array([1/4,1/3,1/4,0])
# diag3= np.array([1/2,1/4,1/12,0,0])
# diag4= np.array([1/12,1/12,0,1/6,1/4,1/4,1/6])
# diag5= np.array([1/12,0,1/12,1/3,1/12,0,1/12,1/3])

# data = [diag1, diag2, diag3, diag4, diag5, diag4, diag3, diag2, diag1]   # list of all the data
# diags = [-5,-4,-3,-1,0,1,3,4,5]                  # which diagonal each vector goes into
# C = sparse.spdiags(data,diags,8,8,format='csc')  # create the matrix
# C= C.todense()
# print(C)
# print(np.zeros((8,8)))

diag = np.ones(8)  # values that will go on main diagonal
      
data2 = [diag, diag, diag]   # list of all the data
diags2 = [-3,1,5]                  # which diagonal each vector goes into
F = sparse.spdiags(data2,diags2,8,8,format='csc')  # create the matrix
F= F.todense()   
  
print(F)

E=3
m=1             #unknown atm so ser to 1 
tau= 2.768e-8  #in seconds 
const = (6*np.pi * cst.c**2)/(cst.hbar*2*np.pi)
# const = 1
I=cst.c*m*E**2
P = const*I*F@C  
print(P)
#%% initial data 

N_i, N_k =0, 1/8

dt = 1.0e-8                    #seperation between normalized time, made small to account for the quick variation in u
tmax =2.0e-6   
# numpy arrays for time and y
tlist=np.arange(0.0, tmax, dt) # gauarantees the same step size
npts = len(tlist)


y=np.zeros((npts,16))          #array of zeros
#E =np.ones((8))*0.0
#G =np.ones((8))*1/8
#yinit = np.concatenate((G, E)) # initial conditions (TLS in ground state)
#y1=np.reshape(yinit,(16) )                     # just a temp array to pass into solver
y1 = np.array([1/8,1/8,1/8,1/8,1/8,1/8,1/8,1/8,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
y[0,:]= y1
print(y.shape)

# print('ymat',y)
# G =y1[:-8]
# E =y1[-8:]
# print('G', G[0])
# print('E', np.reshape(E,(8,1)))
# #E= np.reshape(E,(8,1))
# G1=G[0]
# #G1= np.reshape(G[0],(5,1))
# P1= P[0]
# #print(P1.shape)
# #print((E-G1))
# #P1=np.reshape(P[:,0],(1,8))
# print(P1@ np.reshape(E,(8,1)))
# #print('G', P1@ (E-G1) )
# print('G', P1@ np.reshape((E-G1),(8,1))) 



# start = timeit.default_timer()
# y= rk4(ODE_rate,y1,tlist,dt)
# stop = timeit.default_timer()
# t1= stop - start
# print ("Time for odeint ODE Solver", t1) 

start = timeit.default_timer()  # start timer for solver
for i in range(1,npts):         # loop over time
    y[i,:]= rk4(ODE_rate,y[i-1,:],tlist[i-1],dt) 

stop = timeit.default_timer()
print ("Time for RK4 ODE Solver", stop - start)

#%%

plt.figure()
plt.plot(tlist, y[:,0], linewidth=2,linestyle = 'solid',label= "G1")
plt.plot(tlist, y[:,1], linewidth=2,linestyle = 'solid',label= "G2")
plt.plot(tlist, y[:,2], linewidth=2,linestyle = 'solid',label= "G3")
plt.plot(tlist, y[:,3], linewidth=2,linestyle = 'solid',label= "G4")
plt.plot(tlist, y[:,4], linewidth=2,linestyle = 'solid',label= "G5")
plt.plot(tlist, y[:,5], linewidth=2,linestyle = 'solid',label= "G6")
plt.plot(tlist, y[:,6],linewidth=2,linestyle = 'solid',label= "G7")
plt.plot(tlist, y[:,7], linewidth=2,linestyle = 'solid',label= "G8")
plt.plot(tlist, y[:,8], linewidth=2,linestyle = 'solid',label= "E1")
plt.plot(tlist, y[:,9], linewidth=2,linestyle = 'solid',label= "E2")
plt.plot(tlist, y[:,10], linewidth=2,linestyle = 'solid',label= "E3")
plt.plot(tlist, y[:,11], linewidth=2,linestyle = 'solid',label= "E4")
plt.plot(tlist, y[:,12], linewidth=2,linestyle = 'solid',label= "E5")
plt.plot(tlist, y[:,13], linewidth=2,linestyle = 'solid',label= "E6")
plt.plot(tlist, y[:,14], linewidth=2,linestyle = 'solid',label= "E7")
plt.plot(tlist, y[:,15],linewidth=2,linestyle = 'solid',label= "E8")
plt.grid()
plt.legend(loc=1, fontsize=5 )
plt.ylabel("Population of states", fontsize=15)
plt.xlabel("time (s)",fontsize=15)
plt.title("Population of Quantum states $Rb^{87}$ ",fontsize=15)
plt.savefig('population.jpg', format = 'jpg', dpi = 1200, bbox_inches = 'tight')


#%%


A = np.array([[1,2],[3,4]])
B = A - 3
print(B)