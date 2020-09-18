import sdf
import matplotlib
matplotlib.use('agg')
#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import os
from numpy import ma
from matplotlib import colors, ticker, cm
from matplotlib.mlab import bivariate_normal
import matplotlib.colors as mcolors
import scipy.ndimage as ndimage
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.gridspec as gridspec

import multiprocessing as mp


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
        'size'   : 20,  
        }  

font_size = 20
font_size2=14

if __name__ == '__main__':
  
  x_1d = np.zeros(240) 
  t_1d = np.zeros(225) 
  E_2d = np.zeros([240,225]) 
  
  from_path = './spin_a70_n15_fine/'
  to_path   = './spin_a70_n15_fine_fig/' 

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

  name = 'Ex'  
  ######### Script code drawing figure ################
  E_2d = np.load(from_path+'E_2d.npy') 
  t_1d = np.load(from_path+'t_1d.npy') 
  x_1d = np.load(from_path+'x_1d.npy') 
  x_1d, t_1d = np.meshgrid(x_1d,t_1d)  

#  for n in [10]: #range(np.size(ion_ww[:,0])):
  for n in range(np.size(ion_ww[:,0])):
    
      plt.subplot(2,3,1)
      eee = 20.0
      E_2d[E_2d>eee] = eee
      E_2d[E_2d<-eee]=-eee
      levels = np.linspace(-eee, eee, 40)
      plt.contourf(t_1d.T, x_1d.T, E_2d, levels=levels, norm=mcolors.Normalize(vmin=-eee, vmax=eee), cmap='bwr')
      #### manifesting colorbar, changing label and axis properties ####
      cbar=plt.colorbar(pad=0.01,ticks=[-eee, -eee/2, 0, eee/2, eee])
      cbar.set_label('$E_x\ [m_ec\omega/|e|]$',fontdict=font)
      cbar.ax.tick_params(labelsize=font_size2)
    #  cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(),fontsize=15)        
    #plt.plot(grid_time, gg[n,:], color='r', linewidth=5, linestyle='--',label='$\gamma$')
      plt.scatter(ion_tt, ion_xx[n,:], c=ion_ek[n,:], norm=colors.Normalize(vmin=0,vmax=200), s=30, cmap='rainbow', edgecolors='None', alpha=1,zorder=3)
      cbar=plt.colorbar( ticks=np.linspace(0, 200, 3) ,pad=0.01)
      cbar.set_label('$E_k$ [MeV]',fontdict=font)
      cbar.ax.tick_params(labelsize=font_size2)
      plt.ylabel('$x\ [\mu m]$',fontdict=font)
      plt.xlabel('$t\ [\mathrm{fs}]$',fontdict=font)
      plt.xticks(fontsize=font_size); 
      plt.yticks(fontsize=font_size);
      plt.ylim(-0.7,5.7)
      plt.xlim(10,70)
  
      plt.subplot(2,3,2)
      plt.scatter(ion_tt, ion_px[n,:], c=ion_ek[n,:], norm=colors.Normalize(vmin=0,vmax=200), s=30, cmap='rainbow', edgecolors='None', alpha=1,zorder=3)
      plt.plot(ion_tt, ion_pp[n,:], linestyle=':',color='black',linewidth=3)
      #cbar=plt.colorbar( ticks=np.linspace(0, 200, 3) ,pad=0.01)
      #cbar.set_label('$E_k$ [MeV]',fontdict=font)
      #cbar.ax.tick_params(labelsize=font_size2)
      plt.ylabel('$p_x\ [m_ic]$',fontdict=font)
      plt.xlabel('$t\ [\mathrm{fs}]$',fontdict=font)
      plt.xticks(fontsize=font_size); 
      plt.yticks(fontsize=font_size);
      plt.xlim(10,70)
  
      plt.subplot(2,3,3)
      plt.plot(ion_tt, ion_py[n,:], linestyle='-',color='red',linewidth=3,label='$p_y$')
      plt.plot(ion_tt, ion_pz[n,:], linestyle='-',color='blue',linewidth=3,label='$p_z$')
      plt.ylabel('$p_{y,z}\ [m_ic]$',fontdict=font)
      plt.xlabel('$t\ [\mathrm{fs}]$',fontdict=font)
      plt.xticks(fontsize=font_size); 
      plt.yticks(fontsize=font_size);
      plt.xlim(10,70)
      plt.legend(loc='best',fontsize=font_size2,framealpha=0.5)
 
      v_s1=0.145; v_s2=0.507
      xi = np.zeros_like(ion_tt)
#      condition = (ion_tt<=50.0)
#      xi[condition] = ion_xx[n,condition]-v_s1*(ion_tt[condition]-20.0)/3.333333333333
#      condition = (ion_tt>50.0)
      xi = ion_xx[n,:]-v_s1*(ion_tt-20.0)/3.333333333333
      plt.subplot(2,3,4)
      plt.scatter(xi, ion_px[n,:], c=ion_tt, norm=colors.Normalize(vmin=0,vmax=70), s=30, cmap='jet', edgecolors='None', alpha=1,zorder=3)
      cbar=plt.colorbar(pad=0.01,ticks=[0,10,20,30,40,50,60,70])
      cbar.set_label('$t\ [\mathrm{fs}]$',fontdict=font)
      cbar.ax.tick_params(labelsize=font_size2)
      plt.ylabel('$p_x\ [m_ic]$',fontdict=font)
      plt.xlabel(r'$\xi\ [\mu m]$',fontdict=font)
      plt.xticks(fontsize=font_size); 
      plt.yticks(fontsize=font_size);
#      plt.xlim(-10,70)
#      plt.legend(loc='best',fontsize=font_size2,framealpha=0.5)
  
      plt.subplot(2,3,5)
      plt.scatter(ion_tt, ion_sx[n,:], c=ion_ek[n,:], norm=colors.Normalize(vmin=0,vmax=200), s=30, cmap='rainbow', edgecolors='None', alpha=1,zorder=3)
      plt.subplot(2,3,5)
      plt.scatter(ion_tt, ion_sx[n,:], c=ion_ek[n,:], norm=colors.Normalize(vmin=0,vmax=200), s=30, cmap='rainbow', edgecolors='None', alpha=1,zorder=3)
      plt.plot(ion_tt, ion_ss[n,:], linestyle='--',color='black',linewidth=3)
      #cbar=plt.colorbar( ticks=np.linspace(0, 200, 3) ,pad=0.01)
      #cbar.set_label('$E_k$ [MeV]',fontdict=font)
      #cbar.ax.tick_params(labelsize=font_size2)
      plt.ylabel('$s_x$',fontdict=font)
      plt.xlabel('$t\ [\mathrm{fs}]$',fontdict=font)
      plt.xticks(fontsize=font_size); 
      plt.yticks(fontsize=font_size);
      plt.xlim(10,70)
  
      plt.subplot(2,3,6)
      plt.plot(ion_tt, ion_sy[n,:], linestyle='-',color='red',linewidth=3,label='$s_y$')
      plt.plot(ion_tt, ion_sz[n,:], linestyle='-',color='blue',linewidth=3,label='$s_z$')
      plt.ylabel('$s_{y,z}$',fontdict=font)
      plt.xlabel('$t\ [\mathrm{fs}]$',fontdict=font)
      plt.xticks(fontsize=font_size); 
      plt.yticks(fontsize=font_size);
      plt.xlim(10,70)
      plt.legend(loc='best',fontsize=font_size2,framealpha=0.5)
  
      plt.subplots_adjust(left=0.06, bottom=0.08, right=0.98, top=0.98, wspace=0.24, hspace=0.2)
     
    #  plt.title(name+' at '+str(round(time/1.0e-15,6))+' fs',fontdict=font)
      fig = plt.gcf()
      fig.set_size_inches(20, 10)
      fig.savefig(to_path+'ion_s_trace_evolve'+str(n).zfill(4)+'.png',format='png',dpi=160)
      plt.close("all")
      print(to_path+'ion_s_trace_evolve'+str(n).zfill(4)+'.png')
