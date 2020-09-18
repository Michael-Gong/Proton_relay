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


plt.subplot(2,2,1)
from_path = './spin_a70_n15_n5_fine/'
E_2d = np.load(from_path+'E_2d.npy') 
t_1d = np.load(from_path+'t_1d.npy') 
x_1d = np.load(from_path+'x_1d.npy') 
x_1d, t_1d = np.meshgrid(x_1d,t_1d)  
eee = 30.0
E_2d[E_2d>eee] = eee
E_2d[E_2d<-eee]=-eee
levels = np.linspace(-eee, eee, 40)
plt.contourf(x_1d, t_1d, E_2d.T, levels=levels, norm=colors.Normalize(vmin=-eee, vmax=eee), cmap='bwr')
#### manifesting colorbar, changing label and axis properties ####
#cbar=plt.colorbar(pad=0.01,ticks=[-eee, -eee/2, 0, eee/2, eee])
#cbar.set_label('$E_x\ [m_ec\omega/|e|]$',fontdict=font)
#cbar.ax.tick_params(labelsize=font_size2)
#plt.plot(grid_time, gg[n,:], color='r', linewidth=5, linestyle='--',label='$\gamma$')
from_path = './spin_a70_n15_n5_fine/'
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
ion_tt=np.linspace(0.333333,75,225)  
ion_pp=(ion_px**2+ion_py**2+ion_pz**2)**0.5
ion_ek=((ion_px**2+ion_py**2+ion_pz**2+1)**0.5-1)*918.0
ion_ss=(ion_sx**2+ion_sy**2+ion_sz**2)**0.5
n=10
plt.scatter(ion_xx[n,:],ion_tt, c=ion_ek[n,:], norm=colors.Normalize(vmin=0,vmax=150), s=25, cmap='magma', edgecolors='None', alpha=1,zorder=3)
n=47
plt.scatter(ion_xx[n,:],ion_tt, c=ion_ek[n,:], norm=colors.Normalize(vmin=0,vmax=150), s=25, cmap='magma', edgecolors='None', alpha=1,zorder=3)
#cbar=plt.colorbar( ticks=np.linspace(0, 200, 3) ,pad=0.01)
#cbar.set_label('$E_k$ [MeV]',fontdict=font)
#cbar.ax.tick_params(labelsize=font_size2)
plt.xlabel('$x\ [\mu m]$',fontdict=font)
plt.ylabel('$t\ [\mathrm{fs}]$',fontdict=font)
plt.xticks(fontsize=font_size); 
plt.yticks(fontsize=font_size);
plt.xlim(-0.7,5.7)
plt.ylim(12,70)

plt.subplot(2,2,3)
mp  = 1836. # proton mass
V_s = 0.145  # in unit [c]
K   = 100.0 # electric field ratio
E_0 = 30
px_1d  = np.linspace(-0.2,1.2,201)
xi_1d  = np.linspace(-15,15,201)
xi_0=-0.45*2*np.pi
xi_1= 0.0
xi_2= 0.45*2*np.pi
xi, px = np.meshgrid(xi_1d, px_1d)
phi = np.zeros_like(xi)
condition = xi<=xi_0
phi[condition] = E_0*(xi_2-xi_0)*0.5
condition = (xi_0<xi)&(xi<=xi_1)
phi[condition] = E_0*(xi_2-xi_0)*0.5-E_0/(xi_1-xi_0)*0.5*(xi[condition]-xi_0)**2
condition = (xi_1<xi)&(xi<=xi_2)
phi[condition] = E_0*(xi_2-xi[condition])**2*0.5/(xi_2-xi_1)
condition = (xi>xi_2)
phi[condition] = 0
Hami= ((px**2+1)**0.5-1)*mp-V_s*px*mp+phi
Hami=Hami/mp
plt.contour(xi/(2*np.pi), px, Hami, levels=[0,0.015,0.03],colors='k', linewidths=4, linestyles='--',zorder=4)
mp  = 1836. # proton mass
V_s = 0.507  # in unit [c]
K   = 100.0 # electric field ratio
E_0 = 15
px_1d  = np.linspace(-0.2,1.2,201)
xi_1d  = np.linspace(-15,15,201)
xi_0=-0.9*2*np.pi
xi_1= 0.0
xi_2= 0.9*2*np.pi
xi, px = np.meshgrid(xi_1d, px_1d)
phi = np.zeros_like(xi)
condition = xi<=xi_0
phi[condition] = E_0*(xi_2-xi_0)*0.5
condition = (xi_0<xi)&(xi<=xi_1)
phi[condition] = E_0*(xi_2-xi_0)*0.5-E_0/(xi_1-xi_0)*0.5*(xi[condition]-xi_0)**2
condition = (xi_1<xi)&(xi<=xi_2)
phi[condition] = E_0*(xi_2-xi[condition])**2*0.5/(xi_2-xi_1)
condition = (xi>xi_2)
phi[condition] = 0

Hami= ((px**2+1)**0.5-1)*mp-V_s*px*mp+phi
Hami=Hami/mp
plt.contour(xi/(2*np.pi), px, Hami, levels=[-0.125,-0.092,-0.05,0.0],colors='blue',linewidths=4,linestyles='--')

#plt.plot((np.zeros_like(px_1d)+xi_0)/(2*np.pi),px_1d,'--',color='k')
#plt.plot((np.zeros_like(px_1d)+xi_2)/(2*np.pi),px_1d,'--',color='k') 
v_s1=0.145; v_s2=0.507
for dir_list in ['n5','n15','n15_n5']:
    from_path = './spin_a70_'+dir_list+'_fine/'
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
    ion_tt=np.linspace(0.333333,75,225)  
    ion_pp=(ion_px**2+ion_py**2+ion_pz**2)**0.5
    ion_ek=((ion_px**2+ion_py**2+ion_pz**2+1)**0.5-1)*918.0
    ion_ss=(ion_sx**2+ion_sy**2+ion_sz**2)**0.5
    for n in range(np.size(ion_ww[:,0])):
      if dir_list == 'n15_n5':
          xi = np.zeros_like(ion_tt)
          condition = (ion_tt<=50.0)
          xi[condition] = ion_xx[n,condition]-v_s1*(ion_tt[condition]-20.0)/3.333333333333
          condition = (ion_tt>50.0)
          xi[condition] = ion_xx[n,condition]-v_s2*(ion_tt[condition]-50.0)/3.333333333333-v_s1*30.0/3.333333333333
          color_choice='red'
          lwidth=0.1
      if dir_list == 'n15':
          xi = np.zeros_like(ion_tt)
          xi = ion_xx[n,:]-v_s1*(ion_tt-20.0)/3.333333333333
          color_choice='limegreen'
          lwidth=0.3
      if dir_list == 'n5':
          xi = np.zeros_like(ion_tt)
          xi = ion_xx[n,:]-v_s2*(ion_tt-20.0)/3.333333333333+0.6
          color_choice='gold'
          lwidth=0.12
      #plt.scatter(xi, ion_px[n,:], c=ion_tt, norm=colors.Normalize(vmin=0,vmax=70), s=3, cmap='viridis', edgecolors='None', alpha=1,zorder=10)
      #plt.scatter(xi, ion_px[n,:], c=ion_tt, norm=colors.Normalize(vmin=0,vmax=70), s=3, cmap='viridis', edgecolors='None', alpha=1,zorder=10)
      print(dir_list,n)
      plt.plot(xi, ion_px[n,:], color=color_choice, linewidth=lwidth, linestyle='-',zorder=10)
      if (dir_list=='n15_n5') and (n ==10):
        plt.scatter(xi, ion_px[n,:], c=ion_ek[n,:], norm=colors.Normalize(vmin=0,vmax=150), s=25, cmap='magma', edgecolors='None', alpha=1,zorder=20)
      if (dir_list=='n15_n5') and (n ==47):
        plt.scatter(xi, ion_px[n,:], c=ion_ek[n,:], norm=colors.Normalize(vmin=0,vmax=150), s=25, cmap='magma', edgecolors='None', alpha=1,zorder=20)
    #cbar=plt.colorbar(pad=0.01,ticks=[0,10,20,30,40,50,60,70])
    #cbar.set_label('$t\ [\mathrm{fs}]$',fontdict=font)
    #cbar.ax.tick_params(labelsize=font_size2)
#### manifesting colorbar, changing label and axis properties ####
plt.xlabel(r'$\xi\ [\mu m]$',fontdict=font)
plt.ylabel(r'$p_x\ [m_ic]$',fontdict=font)
plt.xlim(-1.2,1.2)
plt.ylim(-0.02,1.03)
plt.xticks(fontsize=20); 
plt.yticks(fontsize=20);

plt.subplot(2,2,2)
from_path = './spin_a70_n15_n5_fine/'
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
ion_tt=np.linspace(0.333333,75,225)  
ion_pp=(ion_px**2+ion_py**2+ion_pz**2)**0.5
ion_ek=((ion_px**2+ion_py**2+ion_pz**2+1)**0.5-1)*918.0
ion_ss=(ion_sx**2+ion_sy**2+ion_sz**2)**0.5
for n in range(np.size(ion_ww[:,0])):      
      plt.plot(ion_tt, ion_ek[n,:], linestyle='-',color='red',linewidth=0.1)
n=10
plt.scatter(ion_tt, ion_ek[n,:], c=ion_ek[n,:], norm=colors.Normalize(vmin=0,vmax=150), s=25, cmap='magma', edgecolors='None', alpha=1,zorder=3)
n=47
plt.scatter(ion_tt, ion_ek[n,:], c=ion_ek[n,:], norm=colors.Normalize(vmin=0,vmax=150), s=25, cmap='magma', edgecolors='None', alpha=1,zorder=3)
plt.xlabel('$t\ [\mathrm{fs}]$',fontdict=font)
plt.ylabel(r'$\varepsilon_i\ [\mathrm{MeV}]$',fontdict=font)
plt.xticks(fontsize=font_size); 
plt.yticks(fontsize=font_size);
plt.xlim(20,70)
plt.ylim(-10,210)

plt.subplot(2,2,4)
from_path = './spin_a70_n15_n5_fine/'
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
ion_tt=np.linspace(0.333333,75,225)  
ion_pp=(ion_px**2+ion_py**2+ion_pz**2)**0.5
ion_ek=((ion_px**2+ion_py**2+ion_pz**2+1)**0.5-1)*918.0
ion_ss=(ion_sx**2+ion_sy**2+ion_sz**2)**0.5
for n in range(np.size(ion_ww[:,0])):      
      plt.plot(ion_tt, ion_sx[n,:], linestyle='-',color='red',linewidth=0.1)
      plt.plot(ion_tt, ion_ss[n,:], linestyle='--',color='black',linewidth=3)
#n=10
#plt.scatter(ion_tt, ion_sx[n,:], c=ion_ek[n,:], norm=colors.Normalize(vmin=0,vmax=150), s=25, cmap='magma', edgecolors='None', alpha=1,zorder=3)
#n=47
#plt.scatter(ion_tt, ion_sx[n,:], c=ion_ek[n,:], norm=colors.Normalize(vmin=0,vmax=150), s=25, cmap='magma', edgecolors='None', alpha=1,zorder=3)
plt.ylabel('$s_x$',fontdict=font)
plt.xlabel('$t\ [\mathrm{fs}]$',fontdict=font)
plt.xticks(fontsize=font_size); 
plt.yticks(fontsize=font_size);
plt.xlim(20,70)


plt.subplots_adjust(left=0.1, bottom=0.12, right=0.98, top=0.98, wspace=0.24, hspace=0.2)
fig = plt.gcf()
fig.set_size_inches(15, 12)
#plt.show()
fig.savefig('./wrap_hami_relay.png',format='png',dpi=160, transparent=None)
plt.close("all")
