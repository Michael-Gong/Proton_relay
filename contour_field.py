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
font_size2= 15

c_red = matplotlib.colors.colorConverter.to_rgba('red')
c_blue= matplotlib.colors.colorConverter.to_rgba('blue')
c_crimson = matplotlib.colors.colorConverter.to_rgba('crimson')
c_mediblue= matplotlib.colors.colorConverter.to_rgba('mediumblue')
c_white_trans = matplotlib.colors.colorConverter.to_rgba('white',alpha = 0.0)
cmap_rb = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',[c_red,c_white_trans,c_blue],128)
cmap_br = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',[c_blue,c_white_trans,c_red],128)
cmap_mc = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',[c_mediblue,c_white_trans,c_crimson],128)



if __name__ == '__main__':
  from_path = './spin_a70_n15_n5/'
  to_path   = './spin_a70_n15_n5_fig/' 
  start   =  3 # start time
  stop    =  19  # end time
  step    =  1  # the interval or step
    
  for n in range(start,stop+step,step):
    data = sdf.read(from_path+'fields'+str(n).zfill(4)+".sdf",dict=True)
    header=data['Header']
    time=header['time']
    x  = data['Grid/Grid_mid'].data[0]/1.0e-6
    y  = data['Grid/Grid_mid'].data[1]/1.0e-6
    X, Y = np.meshgrid(x, y)
    ex     = data['Electric Field/'+str.capitalize('ex')].data/exunit
    ex_ave = data['Electric Field/'+str.capitalize('ex_averaged')].data/exunit
    by_ave = data['Magnetic Field/'+str.capitalize('by_averaged')].data/bxunit
    bz_ave = data['Magnetic Field/'+str.capitalize('bz_averaged')].data/bxunit
    ey_ave = data['Electric Field/'+str.capitalize('ey_averaged')].data/exunit
    ez_ave = data['Electric Field/'+str.capitalize('ez_averaged')].data/exunit
    n3d=len(ex_ave[0,0,:])
    ex = (ex[:,:,n3d//2-1]+ex[:,:,n3d//2])/2.0 
    ex_ave = (ex_ave[:,:,n3d//2-1]+ex_ave[:,:,n3d//2])/2.0 
    by_ave = (by_ave[:,n3d//2-1,:]+by_ave[:,n3d//2,:])/2.0 
    bz_ave = (bz_ave[:,:,n3d//2-1]+bz_ave[:,:,n3d//2])/2.0 
    ey_ave = (ey_ave[:,:,n3d//2-1]+ey_ave[:,:,n3d//2])/2.0 
    ez_ave = (ez_ave[:,n3d//2-1,:]+ez_ave[:,n3d//2,:])/2.0 
    if np.min(ex.T) == np.max(ex.T):
        continue
    eee=25.0
    levels = np.linspace(-eee, eee, 40)
  
    plt.subplot(3,2,1)
    plt.contourf(X, Y, ex.T, levels=levels, norm=mcolors.Normalize(vmin=-eee, vmax=eee), cmap=cmap_br)
    #### manifesting colorbar, changing label and axis properties ####
    cbar=plt.colorbar(ticks=[-eee, -eee/2, 0, eee/2, eee])
    cbar.set_label('$E_x$',fontdict=font)
    cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(),fontsize=font_size2)
    plt.xlabel('$x\ [\mu m]$',fontdict=font)
    plt.ylabel('$y\ [\mu m]$',fontdict=font)
    plt.xticks(fontsize=font_size); 
    plt.yticks(fontsize=font_size);
    plt.title('Ex at '+str(round(time/1.0e-15,6))+' fs',fontdict=font)
  
    plt.subplot(3,2,2)
    plt.contourf(X, Y, ex_ave.T, levels=levels, norm=mcolors.Normalize(vmin=-eee, vmax=eee), cmap=cmap_br)
    #### manifesting colorbar, changing label and axis properties ####
    cbar=plt.colorbar(ticks=[-eee, -eee/2, 0, eee/2, eee])
    cbar.set_label('$\overline{E}_x$',fontdict=font)
    cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(),fontsize=font_size2)
    plt.xlabel('$x\ [\mu m]$',fontdict=font)
    plt.ylabel('$y\ [\mu m]$',fontdict=font)
    plt.xticks(fontsize=font_size); 
    plt.yticks(fontsize=font_size);
    plt.title('Ex_ave at '+str(round(time/1.0e-15,6))+' fs',fontdict=font)
  
    plt.subplot(3,2,3)
    plt.contourf(X, Y, by_ave.T, levels=levels, norm=mcolors.Normalize(vmin=-eee, vmax=eee), cmap=cmap_br)
    #### manifesting colorbar, changing label and axis properties ####
    cbar=plt.colorbar(ticks=[-eee, -eee/2, 0, eee/2, eee])
    cbar.set_label('$\overline{B}_y$',fontdict=font)
    cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(),fontsize=font_size2)
    plt.xlabel('$x\ [\mu m]$',fontdict=font)
    plt.ylabel('$z\ [\mu m]$',fontdict=font)
    plt.xticks(fontsize=font_size); 
    plt.yticks(fontsize=font_size);
    plt.title('By_ave at '+str(round(time/1.0e-15,6))+' fs',fontdict=font)
  
    plt.subplot(3,2,4)
    plt.contourf(X, Y, bz_ave.T, levels=levels, norm=mcolors.Normalize(vmin=-eee, vmax=eee), cmap=cmap_br)
    #### manifesting colorbar, changing label and axis properties ####
    cbar=plt.colorbar(ticks=[-eee, -eee/2, 0, eee/2, eee])
    cbar.set_label('$\overline{B}_z$',fontdict=font)
    cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(),fontsize=font_size2)
    plt.xlabel('$x\ [\mu m]$',fontdict=font)
    plt.ylabel('$y\ [\mu m]$',fontdict=font)
    plt.xticks(fontsize=font_size); 
    plt.yticks(fontsize=font_size);
    plt.title('Bz_ave at '+str(round(time/1.0e-15,6))+' fs',fontdict=font)
  
    plt.subplot(3,2,5)
    plt.contourf(X, Y, ey_ave.T, levels=levels, norm=mcolors.Normalize(vmin=-eee, vmax=eee), cmap=cmap_br)
    #### manifesting colorbar, changing label and axis properties ####
    cbar=plt.colorbar(ticks=[-eee, -eee/2, 0, eee/2, eee])
    cbar.set_label('$\overline{E}_y$',fontdict=font)
    cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(),fontsize=font_size2)
    plt.xlabel('$x\ [\mu m]$',fontdict=font)
    plt.ylabel('$y\ [\mu m]$',fontdict=font)
    plt.xticks(fontsize=font_size); 
    plt.yticks(fontsize=font_size);
    plt.title('Ey_ave at '+str(round(time/1.0e-15,6))+' fs',fontdict=font)
  
    plt.subplot(3,2,6)
    plt.contourf(X, Y, ez_ave.T, levels=levels, norm=mcolors.Normalize(vmin=-eee, vmax=eee), cmap=cmap_br)
    #### manifesting colorbar, changing label and axis properties ####
    cbar=plt.colorbar(ticks=[-eee, -eee/2, 0, eee/2, eee])
    cbar.set_label('$\overline{E}_z$',fontdict=font)
    cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(),fontsize=font_size2)
    plt.xlabel('$x\ [\mu m]$',fontdict=font)
    plt.ylabel('$y\ [\mu m]$',fontdict=font)
    plt.xticks(fontsize=font_size); 
    plt.yticks(fontsize=font_size);
    plt.title('Ez_ave at '+str(round(time/1.0e-15,6))+' fs',fontdict=font)
  
  
  
    fig = plt.gcf()
    fig.set_size_inches(16, 24)
    fig.savefig(to_path+'contour_field_'+str(n).zfill(4)+'.png',format='png',dpi=100)
    plt.close("all")
    print(to_path+'contour_field_'+str(n).zfill(4)+'.png')
