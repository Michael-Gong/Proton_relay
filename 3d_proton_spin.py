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
from mpl_toolkits.mplot3d import Axes3D

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
        'size'   : 26,  
        }  

font_size = 20


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

def processplot(n): 
  from_path = './spin_a70_n15_n5_revised/'  
  to_path   = './spin_a70_n15_n5_revised_fig/'
  
  data = sdf.read(from_path+"track"+str(n).zfill(4)+".sdf",dict=True)
  header=data['Header']
  time1=header['time']
  px = data['Particles/Px/subset_high_p/ion_s'].data/(1836*m0*v0)
  py = data['Particles/Py/subset_high_p/ion_s'].data/(1836*m0*v0)
  pz = data['Particles/Pz/subset_high_p/ion_s'].data/(1836*m0*v0)
  sx = data['Particles/Sx/subset_high_p/ion_s'].data
  sy = data['Particles/Sy/subset_high_p/ion_s'].data
  sz = data['Particles/Sz/subset_high_p/ion_s'].data
  gg = (px**2+py**2+pz**2+1)**0.5
  Ek = (gg-1)*1836*0.51
  theta = np.arctan2((py**2+pz**2)**0.5,px)*180.0/np.pi
  grid_x = data['Grid/Particles/subset_high_p/ion_s'].data[0]/1.0e-6      
  grid_y = data['Grid/Particles/subset_high_p/ion_s'].data[1]/1.0e-6      
  grid_z = data['Grid/Particles/subset_high_p/ion_s'].data[2]/1.0e-6      
#  temp_id = data['Particles/ID/subset_high_p/ion_s'].data
  grid_r =  (grid_y**2+grid_z**2)**0.5

  condition = (Ek>30.0)&(theta<10.0)
  grid_x = grid_x[condition]
  grid_y = grid_y[condition]
  grid_z = grid_z[condition]
  px     = px[condition]
  Ek     = Ek[condition] 
  sx     = sx[condition]
  sy     = sy[condition]
  sz     = sz[condition]

  sx_min=0.8

  if np.size(px) == 0:
    return 0;
#  sx[sx < sx_min] = sx_min
  sx[0] = 1.0
  sx[-1] = sx_min
  color_index = sx
  fig = plt.figure()
  ax = plt.axes(projection='3d')

  makersize = 0.4
#    plt.subplot()
  #normalize = matplotlib.colors.Normalize(vmin=0, vmax=20, clip=True)
  condition = (grid_y>-10000)|(grid_z<1000000)
  pt3d=ax.scatter(grid_x[condition], grid_y[condition], grid_z[condition], c=color_index[condition], s=makersize*10, marker='.', cmap='viridis', edgecolors='face', alpha=0.8,lw=0,norm=colors.Normalize(vmin=0.7,vmax=1.0))
#  pt3d_2=ax.scatter(grid_x_2, grid_y_2, grid_z_2, c='springgreen', s=makersize*1, edgecolors='face', alpha=1.0, marker='.')

#  cbar=plt.colorbar(pt3d, ticks=[0.8,0.9,1.0] ,pad=0.05)
  cbar=plt.colorbar(pt3d,pad=0.05)
#  cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=20)
  cbar.ax.tick_params(labelsize=20)
  cbar.set_label(r'$s_x$',fontdict=font)
#  cbar.set_clim(0,ek_max)
#plt.plot(np.linspace(-500,900,1001), np.zeros([1001]),':k',linewidth=2.5)
#plt.plot(np.zeros([1001]), np.linspace(-500,900,1001),':k',linewidth=2.5)
#plt.plot(np.linspace(-500,900,1001), np.linspace(-500,900,1001),'-g',linewidth=3)
#plt.plot(np.linspace(-500,900,1001), 200-np.linspace(-500,900,1001),'-',color='grey',linewidth=3)
 #   plt.legend(loc='upper right')
  ax.set_xlim([-1,12])
  ax.set_ylim([-8.,8])
  ax.set_zlim([-8.,8])
  ax.set_xlabel('\n\nX [$\mu m$]',fontdict=font)
  ax.set_ylabel('\n\nY [$\mu m$]',fontdict=font)
  ax.set_zlabel('\n\nZ [$\mu m$]',fontdict=font)
  for t in ax.xaxis.get_major_ticks(): t.label.set_fontsize(font_size)
  for t in ax.yaxis.get_major_ticks(): t.label.set_fontsize(font_size)
  for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(font_size)

  ax.grid(linestyle='--', linewidth='0.5', color='grey')
  ax.view_init(elev=45, azim=-45)

  ax.xaxis.pane.set_edgecolor('black')
  ax.yaxis.pane.set_edgecolor('black')
  ax.zaxis.pane.set_edgecolor('black')
  # Set the background color of the pane YZ
  ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
  ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
  ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

  ax.scatter(grid_x,grid_z,c=color_index,s=makersize*10, alpha=0.8,zdir='y',zs=8,cmap='viridis',marker='.',edgecolors='face',lw=0,norm=colors.Normalize(vmin=0.7,vmax=1.0))
#  ax.scatter(grid_x_2,grid_z_2,c='springgreen',s=makersize*1, alpha=0.5,zdir='y',zs=19,marker='.')
  ax.scatter(grid_x,grid_y,c=color_index,s=makersize*10, alpha=0.8,zdir='z',zs=-8,cmap='viridis',marker='.',edgecolors='face',lw=0,norm=colors.Normalize(vmin=0.7,vmax=1.0))
#  ax.scatter(grid_x_2,grid_y_2,c='springgreen',s=makersize*1, alpha=0.5,zdir='z',zs=-19,marker='.')
  ax.scatter(grid_y,grid_z,c=color_index,s=makersize*10, alpha=0.8,zdir='x',zs=-1,cmap='viridis',marker='.',edgecolors='face',lw=0,norm=colors.Normalize(vmin=0.7,vmax=1.0))
#  ax.scatter(grid_y_2,grid_z_2,c='springgreen',s=makersize*1, alpha=0.5,zdir='x',zs=0,marker='.')


#  plt.text(-100,650,' t = '++' fs',fontdict=font)
  plt.subplots_adjust(left=0.16, bottom=None, right=0.97, top=None,
                wspace=None, hspace=None)
#  plt.title('At '+str(round(time1/1.0e-15,2))+' fs',fontdict=font)
#plt.show()
#lt.figure(figsize=(100,100))


  fig = plt.gcf()
  fig.set_size_inches(12, 10.5)
  fig.savefig(to_path+'new_ion_3d_spin'+str(n).zfill(4)+'.png',format='png',dpi=160)
  plt.close("all")
  print('finised '+str(n).zfill(4))
#  return 0

if __name__ == '__main__':
  start   =  1  # start time
  stop    =  12  # end time
  step    =  1  # the interval or step
    
  inputs = range(start,stop+step,step)
  pool = mp.Pool(processes=1)
  results = pool.map(processplot,inputs)
  print(results)
