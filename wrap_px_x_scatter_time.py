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
        'size'   : 25,  
        }  

font2 = {'family' : 'monospace',  
        'color'  : 'black',  
        'weight' : 'normal',  
        'size'   : 18,  
        }  

font_size=25

def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

if __name__ == '__main__':
  from_path = './spin_a70_n15_n5/'
  to_path   = './'
  
  mass = 1836.

  fig, ax1 = plt.subplots()
#  serial_color = ['purple','dodgerblue','limegreen','orange','red']
  serial_n = [3,5,7,9]
#  serial_range = np.array([ [34,41], [28,34], [21,26], [16.5,19.5], [13,16.5] ])
#  ax2 = ax1.twinx()

  serial_color = ['darkblue']*10+['dodgerblue']*10+['purple']*10+['crimson']*10
  cmap=matplotlib.colors.ListedColormap(serial_color)
  norm = matplotlib.colors.Normalize(vmin=25, vmax=91)

  for i in range(4):
      print(from_path+'track'+str(serial_n[i]).zfill(4)+".sdf")
      data = sdf.read(from_path+'track'+str(serial_n[i]).zfill(4)+".sdf",dict=True)
      header=data['Header']
      time1=header['time']/1.0e-15
      px      = data['Particles/Px/subset_high_p/ion_s'].data/(mass*m0*v0)
      grid_x  = data['Grid/Particles/subset_high_p/ion_s'].data[0]/1.0e-6      
      grid_y  = data['Grid/Particles/subset_high_p/ion_s'].data[1]/1.0e-6      
      grid_z  = data['Grid/Particles/subset_high_p/ion_s'].data[2]/1.0e-6     
      rr = (grid_y**2+grid_z**2)**0.5
      px = px[rr<2.0] 
      grid_x = grid_x[rr<2.0] 
#      temp_id = data['Particles/ID/proton'].data
#      px = px[np.in1d(temp_id,part13_id)]
#      grid_x = grid_x[np.in1d(temp_id,part13_id)]
      time_c = np.zeros_like(grid_x) + time1
      img = ax1.scatter(grid_x, px, c=time_c, norm=colors.Normalize(vmin=25, vmax=91), cmap=colors.ListedColormap(serial_color), s=0.02, edgecolors='None', alpha=0.7)
#      x  = np.loadtxt(from_path+'ex_lineout_x.txt')
#      ex = np.loadtxt(from_path+'ex_lineout_r15_'+str(serial_n[i]).zfill(4)+'.txt')
#      ex = ex[ (x>= serial_range[i][0]) & (x<=serial_range[i][1])] 
#      x = x[ (x>= serial_range[i][0]) & (x<=serial_range[i][1])] 
#      ax2.plot(x,ex, '--', linewidth=3, color=serial_color[45-i*10], label="Ex")

  cax = fig.add_axes([0.35,0.95,0.53,0.02])
  cbar = fig.colorbar(img,cax=cax,label='t [fs]', ticks=[33,50,67,83], orientation='horizontal')
  cbar.set_label('time [fs]',fontdict=font2)
  cbar.ax.tick_params(labelsize=font2['size'])
  #cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(),fontsize=18)

  ax1.set_xlim(-2,11)
  ax1.set_ylim(-0.2,1.4)
  ax1.set_xlabel('$x$ [$\mu m$]',fontdict=font)
  ax1.set_ylabel('$p_x$ [$m_pc$]',fontdict=font)
  ax1.set_xticks([-2,0,2,4,6,8,10])  
  ax1.set_yticks([0,0.4,0.8,1.2])  
  ax1.tick_params(axis='x',labelsize=font_size)
  ax1.tick_params(axis='y',labelsize=font_size)
  ax1.grid(linestyle=':', linewidth=0.4, color='grey')

#  ax2.set_ylim(0.,25)
#  ax2.set_ylabel('$E_x$ [$m_ec\omega/|e|$]',fontdict=font,color='r')
#  ax2.tick_params(axis='y',labelsize=25,colors='r')
#  plt.text(-100,650,' t = '++' fs',fontdict=font)
  plt.subplots_adjust(left=0.23, bottom=0.14, right=0.99, top=0.98, wspace=None, hspace=None)
#  plt.title('At '+str(round(time1/1.0e-15,2))+' fs',fontdict=font)
#plt.show()
#lt.figure(figsize=(100,100))
  #par1.set_ylabel(r'$E_x\ [m_ec\omega/|e|]$',fontdict=font)
  #par1.yaxis.label.set_color('red')
  #par1.tick_params(axis='y', colors='red', labelsize=20)
  #par1.set_ylim(-5,12)
  fig = plt.gcf()
  fig.set_size_inches(8, 6.5)
  fig.savefig('./wrap_px_x_scatter_time.png',format='png',dpi=160)
  plt.close("all")
  print('./wrap_px_x_scatter_time.png')
