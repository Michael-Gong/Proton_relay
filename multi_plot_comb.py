#%matplotlib inline
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from numpy import ma
from matplotlib import colors, ticker, cm
from matplotlib.mlab import bivariate_normal
from optparse import OptionParser
import os
from mpl_toolkits.mplot3d import Axes3D
import random
from mpl_toolkits import mplot3d
from matplotlib import rc
#plt.rcParams['mathtext.fontset'] = 'dejavuserif'
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})

#rc('text', usetex=True) #open this for latex
######## Constant defined here ########
pi        =     3.1415926535897932384626
q0        =     1.602176565e-19 # C
m0        =     9.10938291e-31  # kg
v0        =     2.99792458e8    # m/s^2
kb        =     1.3806488e-23   # J/K
mu0       =     4.0e-7*pi       # N/A^2
epsilon0  =     8.8541878176203899e-12 # F/m
h_planck  =     6.62606957e-34  # J s
wavelength=     1.0e-6
frequency =     v0*2*pi/wavelength

exunit    =     m0*v0*frequency/q0
bxunit    =     m0*frequency/q0
denunit    =     frequency**2*epsilon0*m0/q0**2
print('electric field unit: '+str(exunit))
print('magnetic field unit: '+str(bxunit))
print('density unit nc: '+str(denunit))
font = {'family' : 'monospace',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 25,
        }

font2 = {'family' : 'monospace',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 20,
        }

font_size =25
font_size2=20


from_path = './spin_a70_n15_n5_fine_2/'
part_name = 'ion_s'
part_mass = 1836
ion_px=np.loadtxt(from_path+part_name+'_px.txt')
ion_py=np.loadtxt(from_path+part_name+'_py.txt')
ion_pz=np.loadtxt(from_path+part_name+'_pz.txt')
ion_xx=np.loadtxt(from_path+part_name+'_xx.txt')
ion_yy=np.loadtxt(from_path+part_name+'_yy.txt')
ion_zz=np.loadtxt(from_path+part_name+'_zz.txt')
ion_sx=np.loadtxt(from_path+part_name+'_sx.txt')*(part_mass*m0*v0)
ion_sy=np.loadtxt(from_path+part_name+'_sy.txt')*(part_mass*m0*v0)
ion_sz=np.loadtxt(from_path+part_name+'_sz.txt')*(part_mass*m0*v0)
ion_ww=np.loadtxt(from_path+part_name+'_ww.txt')
ion_ex=np.loadtxt(from_path+part_name+'_ex.txt')
ion_ey=np.loadtxt(from_path+part_name+'_ey.txt')
ion_ez=np.loadtxt(from_path+part_name+'_ez.txt')
ion_bx=np.loadtxt(from_path+part_name+'_bx.txt')
ion_by=np.loadtxt(from_path+part_name+'_by.txt')
ion_bz=np.loadtxt(from_path+part_name+'_bz.txt')

ion_tt=np.linspace(0.333333,75,225)  
ion_pp=(ion_px**2+ion_py**2+ion_pz**2)**0.5
ion_ek=((ion_px**2+ion_py**2+ion_pz**2+1)**0.5-1)*918.0
ion_ss=(ion_sx**2+ion_sy**2+ion_sz**2)**0.5


lwdth = 0.02
#plt.subplot(3,3,1)
#for n in range(np.size(ion_ww[:,0])):      
#      plt.plot(ion_tt, ion_ek[n,:], linestyle='-',color='red',linewidth=lwdth)
##n=10
##plt.scatter(ion_tt, ion_ek[n,:], c=ion_ek[n,:], norm=colors.Normalize(vmin=0,vmax=150), s=25, cmap='magma', edgecolors='None', alpha=1,zorder=3)
##n=47
##plt.scatter(ion_tt, ion_ek[n,:], c=ion_ek[n,:], norm=colors.Normalize(vmin=0,vmax=150), s=25, cmap='magma', edgecolors='None', alpha=1,zorder=3)
#plt.xlabel('$t\ [\mathrm{fs}]$',fontdict=font)
#plt.ylabel(r'$\varepsilon_i\ [\mathrm{MeV}]$',fontdict=font)
#plt.xticks(fontsize=font_size); 
#plt.yticks(fontsize=font_size);
#plt.xlim(20,70)
#plt.ylim(-10,210)

plt.subplot(4,3,1)
for n in range(np.size(ion_ww[:,0])):      
      plt.plot(ion_tt, ion_sx[n,:], linestyle='-',color='red',linewidth=lwdth)
      plt.plot(ion_tt, ion_ss[n,:], linestyle='-',color='black',linewidth=lwdth)
plt.ylabel('$s_x$',fontdict=font)
plt.xlabel('$t\ [\mathrm{fs}]$',fontdict=font)
plt.xticks(fontsize=font_size); 
plt.yticks(fontsize=font_size);
#plt.xlim(20,70)

plt.subplot(4,3,2)
for n in range(np.size(ion_ww[:,0])):      
      plt.plot(ion_tt, ion_sy[n,:], linestyle='-',color='red',linewidth=lwdth)
      plt.plot(ion_tt, ion_ss[n,:], linestyle='-',color='black',linewidth=lwdth)
plt.ylabel('$s_y$',fontdict=font)
plt.xlabel('$t\ [\mathrm{fs}]$',fontdict=font)
plt.xticks(fontsize=font_size); 
plt.yticks(fontsize=font_size);
#plt.xlim(20,70)

plt.subplot(4,3,3)
for n in range(np.size(ion_ww[:,0])):      
      plt.plot(ion_tt, ion_sz[n,:], linestyle='-',color='red',linewidth=lwdth)
      plt.plot(ion_tt, ion_ss[n,:], linestyle='-',color='black',linewidth=lwdth)
plt.ylabel('$s_y$',fontdict=font)
plt.xlabel('$t\ [\mathrm{fs}]$',fontdict=font)
plt.xticks(fontsize=font_size); 
plt.yticks(fontsize=font_size);
#plt.xlim(20,70)

plt.subplot(4,3,4)
dsx_dt = ion_sy*ion_bz-ion_sz*ion_by
for n in range(np.size(ion_ww[:,0])):      
      plt.plot(ion_tt, dsx_dt[n,:], linestyle='-',color='red',linewidth=lwdth)
plt.ylabel(r'$ds_x/dt\approx s_yB_z-s_zBy$',fontdict=font)
plt.xlabel('$t\ [\mathrm{fs}]$',fontdict=font)
plt.xticks(fontsize=font_size); 
plt.yticks(fontsize=font_size);
#plt.xlim(20,70)

plt.subplot(4,3,5)
dsy_dt = ion_sz*ion_bx-ion_sx*ion_bz
for n in range(np.size(ion_ww[:,0])):      
      plt.plot(ion_tt, dsy_dt[n,:], linestyle='-',color='red',linewidth=lwdth)
plt.ylabel(r'$ds_y/dt\approx s_zB_x-s_xBz$',fontdict=font)
plt.xlabel('$t\ [\mathrm{fs}]$',fontdict=font)
plt.xticks(fontsize=font_size); 
plt.yticks(fontsize=font_size);
#plt.xlim(20,70)

plt.subplot(4,3,6)
dsz_dt = ion_sx*ion_by-ion_sy*ion_bx
for n in range(np.size(ion_ww[:,0])):      
      plt.plot(ion_tt, dsy_dt[n,:], linestyle='-',color='red',linewidth=lwdth)
plt.ylabel(r'$ds_z/dt\approx s_xB_y-s_yBx$',fontdict=font)
plt.xlabel('$t\ [\mathrm{fs}]$',fontdict=font)
plt.xticks(fontsize=font_size); 
plt.yticks(fontsize=font_size);
#plt.xlim(20,70)

plt.subplot(4,3,7)
for n in range(np.size(ion_ww[:,0])):      
      plt.plot(ion_tt, ion_ex[n,:], linestyle='-',color='red',linewidth=lwdth)
plt.xlabel('$t\ [\mathrm{fs}]$',fontdict=font)
plt.ylabel('$E_x\ [m_ec\omega_0/|e|]$',fontdict=font)
plt.xticks(fontsize=font_size); 
plt.yticks(fontsize=font_size);
#plt.xlim(20,70)

plt.subplot(4,3,8)
for n in range(np.size(ion_ww[:,0])):      
      plt.plot(ion_tt, ion_ey[n,:], linestyle='-',color='red',linewidth=lwdth)
plt.xlabel('$t\ [\mathrm{fs}]$',fontdict=font)
plt.ylabel('$E_y\ [m_ec\omega_0/|e|]$',fontdict=font)
plt.xticks(fontsize=font_size); 
plt.yticks(fontsize=font_size);
#plt.xlim(20,70)

plt.subplot(4,3,9)
for n in range(np.size(ion_ww[:,0])):      
      plt.plot(ion_tt, ion_ez[n,:], linestyle='-',color='red',linewidth=lwdth)
plt.xlabel('$t\ [\mathrm{fs}]$',fontdict=font)
plt.ylabel('$E_z\ [m_ec\omega_0/|e|]$',fontdict=font)
plt.xticks(fontsize=font_size); 
plt.yticks(fontsize=font_size);
#plt.xlim(20,70)

plt.subplot(4,3,10)
for n in range(np.size(ion_ww[:,0])):      
      plt.plot(ion_tt, ion_bx[n,:], linestyle='-',color='red',linewidth=lwdth)
plt.xlabel('$t\ [\mathrm{fs}]$',fontdict=font)
plt.ylabel('$B_x\ [m_e\omega_0/|e|]$',fontdict=font)
plt.xticks(fontsize=font_size); 
plt.yticks(fontsize=font_size);
#plt.xlim(20,70)

plt.subplot(4,3,11)
for n in range(np.size(ion_ww[:,0])):      
      plt.plot(ion_tt, ion_by[n,:], linestyle='-',color='red',linewidth=lwdth)
plt.xlabel('$t\ [\mathrm{fs}]$',fontdict=font)
plt.ylabel('$B_y\ [m_e\omega_0/|e|]$',fontdict=font)
plt.xticks(fontsize=font_size); 
plt.yticks(fontsize=font_size);
#plt.xlim(20,70)

plt.subplot(4,3,12)
for n in range(np.size(ion_ww[:,0])):      
      plt.plot(ion_tt, ion_bz[n,:], linestyle='-',color='red',linewidth=lwdth)
plt.xlabel('$t\ [\mathrm{fs}]$',fontdict=font)
plt.ylabel('$B_z\ [m_e\omega_0/|e|]$',fontdict=font)
plt.xticks(fontsize=font_size); 
plt.yticks(fontsize=font_size);
#plt.xlim(20,70)


plt.subplots_adjust(left=0.1, bottom=0.12, right=0.98, top=0.98, wspace=0.24, hspace=0.2)
fig = plt.gcf()
fig.set_size_inches(18, 18)
#plt.show()
fig.savefig('./multi_plot_spin.png',format='png',dpi=160, transparent=None)
plt.close("all")
