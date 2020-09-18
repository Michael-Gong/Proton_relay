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

def find_field_E(x,y,grid_x,grid_y,E):
    x_i = np.max(np.where(grid_x<x))
    x_r = (x-grid_x[x_i])/(grid_x[-1]-grid_x[-2])
    x_j = x_i+1
    y_i = np.max(np.where(grid_y<y))
    y_r = (y-grid_y[y_i])/(grid_y[-1]-grid_y[-2])
    y_j = y_i+1
    E_want = (E[x_i,y_i]*(1-x_r)+E[x_j,y_i]*x_r)*(1-y_r) + (E[x_i,y_j]*(1-x_r)+E[x_j,y_j]*x_r)*y_r
    return E_want

c_red   = matplotlib.colors.colorConverter.to_rgba('red')
c_blue  = matplotlib.colors.colorConverter.to_rgba('blue')
c_yellow= matplotlib.colors.colorConverter.to_rgba('yellow')
c_cyan  = matplotlib.colors.colorConverter.to_rgba('cyan')
c_black = matplotlib.colors.colorConverter.to_rgba('black')
c_green = matplotlib.colors.colorConverter.to_rgba('lime')
c_white_trans = matplotlib.colors.colorConverter.to_rgba('white',alpha = 0.0)
cmap_rb = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',[c_red,c_white_trans,c_blue],128)
cmap_br = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',[c_blue,c_white_trans,c_red],128)
cmap_ryb= matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',[c_red,c_yellow,c_blue],128)
cmap_yc = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',[c_yellow,c_white_trans,c_cyan],128)


font_size =25
font_size2=20


fig=plt.subplot(1,2,1)
from_path = './spin_a70_n15_n5_fine/'
E_2d = np.load(from_path+'E_2d.npy') 
t_1d = np.load(from_path+'t_1d.npy') 
x_1d = np.load(from_path+'x_1d.npy') 
x_1d, t_1d = np.meshgrid(x_1d,t_1d)  
eee = 30.0
E_2d[E_2d>eee] = eee
E_2d[E_2d<-eee]=-eee
levels = np.linspace(-eee, eee, 40)
plt.contourf(x_1d, t_1d, E_2d.T, levels=levels, norm=colors.Normalize(vmin=-eee, vmax=eee), cmap=cmap_br)
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
n_space = 20
dpx_dt = (ion_px[:,n_space:]-ion_px[:,:-n_space])/(ion_tt[n_space:]-ion_tt[:-n_space])
data_t    = ion_tt[n_space//2:-n_space//2]
#for n in range(np.size(ion_ww[:,0])):      
#      plt.plot(ion_xx[n,:], ion_tt, linestyle='-',color='red',linewidth=0.1)
#n=10
#plt.scatter(ion_xx[n,:],ion_tt, c=ion_ek[n,:], norm=colors.Normalize(vmin=0,vmax=150), s=50, cmap='magma', edgecolors='k', alpha=1,zorder=3)
n=47
plt.scatter(ion_xx[n,::3],ion_tt[::3], c=ion_ek[n,::3], norm=colors.Normalize(vmin=0,vmax=150), s=200, cmap='magma', edgecolors='k', alpha=1,zorder=3,linewidth=3)
#cbar=plt.colorbar( ticks=np.linspace(0, 200, 3) ,pad=0.01)
#cbar.set_label('$E_k$ [MeV]',fontdict=font)
#cbar.ax.tick_params(labelsize=font_size2)
plt.xlabel('$x\ [\mu m]$',fontdict=font)
plt.ylabel('$t\ [\mathrm{fs}]$',fontdict=font)
plt.xticks(fontsize=font_size); 
plt.yticks(fontsize=font_size);
plt.xlim(-0.4,5.2)
plt.ylim(20,70)

par1 = fig.twiny()
#par1.spines["top"].set_position(("axes", 1.))
par1.plot((dpx_dt*(1836.0*m0*v0/1e-15)/q0/exunit)[n,:],ion_tt[n_space//2:-n_space//2], color='darkmagenta',linewidth=3,label='$dE_k/dt$')
#par1.set_xlim(18.5,71.5)
par1.set_xlim(160,0)
par1.set_xticks([0, 10, 20, 30])  
#par1.set_xlabel('$E_x\ [m_ec\omega_0/|e|]$',fontdict=font2,color='black')
par1.tick_params(axis='x',labelsize=font_size2,colors='darkmagenta')


fig=plt.subplot(2,2,2)
#fig.yaxis.tick_right()
#fig.yaxis.set_label_position("right")
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
#n=10
#plt.scatter(ion_tt, ion_ek[n,:], c=ion_ek[n,:], norm=colors.Normalize(vmin=0,vmax=150), s=25, cmap='magma', edgecolors='k', alpha=1,zorder=3)
n=47
plt.scatter(ion_tt, ion_ek[n,:], c=ion_ek[n,:], norm=colors.Normalize(vmin=0,vmax=150), s=25, cmap='magma', edgecolors='k', alpha=1,zorder=3)
#plt.xlabel('$t\ [\mathrm{fs}]$',fontdict=font)
plt.ylabel(r'$\varepsilon_p\ [\mathrm{MeV}]$',fontdict=font)
plt.xticks(fontsize=0.001); 
plt.yticks([0,50,100,150,200],fontsize=font_size);
plt.xlim(18.5,70)
plt.ylim(-10,210)

fig=plt.subplot(2,2,4)
#fig.yaxis.tick_right()
#fig.yaxis.set_label_position("right")
n_space = 20
n_space2= 20
dek_dx = (ion_ek[:,n_space:]-ion_ek[:,:-n_space])/(ion_xx[:,n_space:]-ion_xx[:,:-n_space])
dek_dt = (ion_ek[:,n_space2:]-ion_ek[:,:-n_space2])/(ion_tt[n_space2:]-ion_tt[:-n_space2])
dpx_dt = (ion_px[:,n_space:]-ion_px[:,:-n_space])/(ion_tt[n_space:]-ion_tt[:-n_space])
data_t    = ion_tt[n_space//2:-n_space//2:3]
data_dpdt = np.zeros_like(data_t)
#print(np.shape(dpx_dt[1,:]))
#print(ion_tt[5:-5])
for n in range(np.size(ion_ww[:,0])):
     data_dpdt = data_dpdt + dpx_dt[n,::3] 
data_dpdt = data_dpdt/np.size(ion_ww[:,0])    
width=0.8
pl=plt.bar(data_t, 100.*data_dpdt, width, color='crimson',edgecolor='black',linewidth=2,alpha=0.7,label=r'$dp_x/dt$')
#pl=plt.bar(rr_x, rr_y, width, color='deepskyblue',edgecolor='black',linewidth=1,alpha=0.5,label='Classical')
#for n in range(np.size(ion_ww[:,0])):      
#      plt.plot(ion_tt[5:-5], dpx_dt[n,:], linestyle='-',color='red',linewidth=0.1)
#n=10
#plt.scatter(ion_tt[n_space//2:-n_space//2], dpx_dt[n,:], c=ion_ek[n,n_space//2:-n_space//2], norm=colors.Normalize(vmin=0,vmax=150), s=25, cmap='magma', edgecolors='None', alpha=1,zorder=3)
#n=47
#plt.scatter(ion_tt[n_space//2:-n_space//2], dpx_dt[n,:], c=ion_ek[n,n_space//2:-n_space//2], norm=colors.Normalize(vmin=0,vmax=150), s=25, cmap='magma', edgecolors='None', alpha=1,zorder=3)
plt.xlabel('$t\ [\mathrm{fs}]$',fontdict=font)
plt.ylabel(r'$dp_x/dt\ [0.01m_pc/\mathrm{fs}]$',fontdict=font)
plt.xticks(fontsize=font_size); 
plt.yticks(fontsize=font_size);
plt.xlim(18.5,70)
#plt.ylim(0,3.0)

#par1 = fig.twinx()
#data_t    = ion_tt[n_space2//2:-n_space2//2:3]
#data_dedt = np.zeros_like(data_t)
#for n in range(np.size(ion_ww[:,0])):
#     data_dedt = data_dedt + dek_dt[n,::3] 
#data_dedt = data_dedt/np.size(ion_ww[:,0])    
#par1.bar(data_t, data_dedt, width, color='dodgerblue',edgecolor='black',linewidth=2,alpha=0.5,label='$dE_k/dt$')
##par1.set_xlim(18.5,71.5)
##par1.set_ylim(0,18)
#par1.set_ylabel('$dE_k/dt\ [\mathrm{MeV}/\mathrm{fs}]$',fontdict=font2,color='black')
#par1.tick_params(axis='y',labelsize=font_size2,colors='blue')


plt.subplots_adjust(left=0.08, bottom=0.11, right=0.98, top=0.98, wspace=0.2, hspace=0.03)
fig = plt.gcf()
fig.set_size_inches(18,7.5)
#plt.show()
fig.savefig('./wrap_ex_proton_traj.png',format='png',dpi=160, transparent=None)
plt.close("all")
