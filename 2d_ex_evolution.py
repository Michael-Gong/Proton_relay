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
  start   =  0 # start time
  stop    =  224  # end time
  step    =  1  # the interval or step

  
  x_1d = np.zeros(240) 
  t_1d = np.zeros(225) 
  E_2d = np.zeros([240,225]) 
  
  from_path = './spin_a70_n15_n5_fine_2/'
  to_path   = './spin_a70_n15_n5_fine_2_fig/' 

  name = 'Ex'  
  have_saved = 0
  ######### Script code drawing figure ################
  if have_saved == 0:
    for n in range(start,stop+step,step):
  #### header data ####
      data = sdf.read(from_path+'fields'+str(n).zfill(4)+".sdf",dict=True)
      time = data['Header']['time']/1e-15
      t_1d[n] = time
      x  = data['Grid/Grid_mid'].data[0]/1.0e-6
      y  = data['Grid/Grid_mid'].data[1]/1.0e-6
      z  = data['Grid/Grid_mid'].data[2]/1.0e-6
      if 'Electric Field/'+str.capitalize(name) not in data:
          continue
      ex = data['Electric Field/'+str.capitalize(name)].data/exunit
      x_1d    = x

      dim_y = np.size(y)
      dim_z = np.size(z)

      y,z = np.meshgrid(y,z,indexing='ij')
      rr = (y**2+z**2)**0.5
#      ex = (ex[:,dim_y//2-1,dim_z//2-1]+ex[:,dim_y//2,dim_z//2-1]+ex[:,dim_y//2-1,dim_z//2]+ex[:,dim_y//2,dim_z//2])/4.0
      print(np.shape(ex[:,rr<1]),np.size(rr[rr<1]))
      ex = np.sum(ex[:,rr<1],axis=1)/np.size(rr[rr<1])
     
      E_2d[:,n] = ex
      print('finished',n)
    np.save(from_path+'E_2d',E_2d)
    np.save(from_path+'t_1d',t_1d)
    np.save(from_path+'x_1d',x_1d)
  else:
    E_2d = np.load(from_path+'E_2d.npy') 
    t_1d = np.load(from_path+'t_1d.npy') 
    x_1d = np.load(from_path+'x_1d.npy') 
  x_1d, t_1d = np.meshgrid(x_1d,t_1d)  

  eee = 30.0
  E_2d[E_2d>eee] = eee
  E_2d[E_2d<-eee]=-eee
  levels = np.linspace(-eee, eee, 40)
  plt.contourf(x_1d, t_1d, E_2d.T, levels=levels, norm=mcolors.Normalize(vmin=-eee, vmax=eee), cmap='bwr')
  #### manifesting colorbar, changing label and axis properties ####
  cbar=plt.colorbar(pad=0.01,ticks=[-eee, -eee/2, 0, eee/2, eee])
  cbar.set_label('$E_x\ [m_ec\omega/|e|]$',fontdict=font)
  cbar.ax.tick_params(labelsize=font_size2)
#  cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(),fontsize=15)        
  plt.xlabel('$x\ [\mu m]$',fontdict=font)
  plt.ylabel('$t\ [\mathrm{fs}]$',fontdict=font)
  plt.xticks(fontsize=font_size); 
  plt.yticks(fontsize=font_size);
  plt.xlim(-0.7,5.7)
  plt.ylim(10,70)
#  plt.title(name+' at '+str(round(time/1.0e-15,6))+' fs',fontdict=font)
  fig = plt.gcf()
  fig.set_size_inches(10, 8)
  fig.savefig(to_path+name+'_evolve.png',format='png',dpi=160)
  plt.close("all")
