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

c_red  = matplotlib.colors.colorConverter.to_rgba('red')
c_blue = matplotlib.colors.colorConverter.to_rgba('blue')
c_black = matplotlib.colors.colorConverter.to_rgba('black')
c_green= matplotlib.colors.colorConverter.to_rgba('lime')
c_white_trans = matplotlib.colors.colorConverter.to_rgba('white',alpha = 0.0)
cmap_mycolor1 = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',[c_white_trans,c_red,c_green],128)
cmap_my_rw = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',[c_red,c_white_trans],128)
cmap_my_bw = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',[c_blue,c_white_trans],128)
cmap_my_bwr = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',[c_blue,c_white_trans,c_white_trans,c_white_trans,c_red],128)
cmap_my_kw = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',[c_black,c_white_trans],128)
cmap_my_wr = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',[c_white_trans,c_red],128)
cmap_my_wb = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',[c_white_trans,c_blue],128)
cmap_my_wk = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',[c_white_trans,c_black],128)

def rebin3d(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1],shape[2],a.shape[2]//shape[2]
    return a.reshape(sh).mean(-1).mean(3).mean(1)


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

def processplot(n): 
  from_path = './spin_a70_n15_n5/'  
  to_path   = './spin_a70_n15_n5_fig/'
  x_start=0; x_stop=300; y_start=0; y_stop=200; z_start=0; z_stop=200;
  x_size = x_stop-x_start; y_size = y_stop-y_start; z_size = z_stop-z_start
  name = 'Ey_laser'
  
  data = sdf.read(from_path+"track"+str(n).zfill(4)+".sdf",dict=True)
  header=data['Header']
  time1=header['time']
  px = data['Particles/Px/subset_high_p/ion_s'].data/(1836*m0*v0)
  py = data['Particles/Py/subset_high_p/ion_s'].data/(1836*m0*v0)
  pz = data['Particles/Pz/subset_high_p/ion_s'].data/(1836*m0*v0)
  gg = (px**2+py**2+pz**2+1)**0.5
  Ek = (gg-1)*1836*0.51
  theta = np.arctan2((py**2+pz**2)**0.5,px)*180.0/np.pi
  grid_x = data['Grid/Particles/subset_high_p/ion_s'].data[0]/1.0e-6      
  grid_y = data['Grid/Particles/subset_high_p/ion_s'].data[1]/1.0e-6      
  grid_z = data['Grid/Particles/subset_high_p/ion_s'].data[2]/1.0e-6      
  grid_r =  (grid_y**2+grid_z**2)**0.5

  condition = Ek>30.0
  grid_x = grid_x[condition]
  grid_y = grid_y[condition]
  grid_z = grid_z[condition]
  px     = px[condition]
  Ek     = Ek[condition] 

  ek_max=300

  if np.size(px) == 0:
    return 0;
  Ek[Ek > ek_max] = ek_max
  Ek[0] = 0.0
  Ek[-1] = ek_max
  color_index = Ek
  fig = plt.figure()
  ax = plt.axes(projection='3d')
  makersize = 0.4
  pt3d=ax.scatter(grid_x, grid_y, grid_z, c=color_index, s=makersize*10, marker='.', cmap='magma', edgecolors='face', alpha=0.8,lw=0,norm=colors.Normalize(vmin=0,vmax=300))
  cbar=plt.colorbar(pt3d, ticks=[0,100,200,300] ,pad=0.05)
#  cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=20)
  cbar.ax.tick_params(labelsize=20)
  cbar.set_label(r'$E_k$'+' [MeV]',fontdict=font)


  data = sdf.read(from_path+'fields'+str(n).zfill(4)+'.sdf',dict=True)
  header=data['Header']
  time =header['time']
  x    = data['Grid/Grid_mid'].data[0]/1.e-6
  y    = data['Grid/Grid_mid'].data[1]/1.e-6
  z    = data['Grid/Grid_mid'].data[2]/1.e-6
  var1  = data['Electric Field/Ey'].data/exunit
  var   = var1 #(var1**2+var2**2)**0.5
  X, Y, Z = np.meshgrid(x, y, z, sparse=False, indexing='ij')
  var  = var[x_start:x_stop,y_start:y_stop,z_start:z_stop]
  X    =  X[x_start:x_stop,y_start:y_stop,z_start:z_stop]
  Y    =  Y[x_start:x_stop,y_start:y_stop,z_start:z_stop]
  Z    =  Z[x_start:x_stop,y_start:y_stop,z_start:z_stop]
  var = rebin3d(var, (x_size//2, y_size//2, z_size//2))
  X = rebin3d(X, (x_size//2, y_size//2, z_size//2))
  Y = rebin3d(Y, (x_size//2, y_size//2, z_size//2))
  Z = rebin3d(Z, (x_size//2, y_size//2, z_size//2))
  var  = var.reshape(np.size(var))
  X    = X.reshape(np.size(X))
  Y    = Y.reshape(np.size(Y))
  Z    = Z.reshape(np.size(Z))

  plotkws = {'marker':'.','edgecolors':'none'}
  eee = 50.0
  var[var>eee] = eee
  var[var<-eee]=-eee
  im = ax.scatter(X, Y, Z, c=var, norm=colors.Normalize(vmin=-eee,vmax=eee), cmap=cmap_my_bwr, **plotkws)
  cbar=plt.colorbar(im, pad=0.01)
  cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=20)
  cbar.set_label('$E_y $'+r'$[m_ec\omega/|e|]$',fontdict=font)
 
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


  X,Z = np.meshgrid(x,z,indexing='ij')
  eexx = data['Electric Field/Ex_averaged'].data/exunit
  ex = (eexx[:,(y_start+y_stop)//2-1,:]+eexx[:,(y_start+y_stop)//2,:])/2
  ex = ex[x_start:x_stop,z_start:z_stop]
  X  = X[x_start:x_stop,z_start:z_stop]
  Z  = Z[x_start:x_stop,z_start:z_stop]
  if np.min(ex.T) == np.max(ex.T):
         #continue
         return
  eee = 30
  levels = np.linspace(-eee, eee, 40)
  ex[ex>eee] = eee
  ex[ex<-eee] = -eee
  ax.contourf(X.T, ex.T, Z.T, levels=levels, norm=mcolors.Normalize(vmin=-eee, vmax=eee), cmap=cm.bwr, zdir='y', offset=(y_stop-y_start)/2/10)

  X,Y = np.meshgrid(x,y,indexing='ij')
  eexx = data['Electric Field/Ex_averaged'].data/exunit
  ex = (eexx[:,:,(z_start+z_stop)//2-1]+eexx[:,:,(z_start+z_stop)//2])/2
  ex = ex[x_start:x_stop,y_start:y_stop]
  X  = X[x_start:x_stop,y_start:y_stop]
  Y  = Y[x_start:x_stop,y_start:y_stop]
  if np.min(ex.T) == np.max(ex.T):
         #continue
         return
  eee = 30
  levels = np.linspace(-eee, eee, 40)
  ex[ex>eee] = eee
  ex[ex<-eee] = -eee
  im2=ax.contourf(X.T, Y.T, ex.T, levels=levels, norm=mcolors.Normalize(vmin=-eee, vmax=eee), cmap=cm.bwr, zdir='z', offset=-(z_stop-z_start)/2/10)
#    ax.set_xlim([x_start/20-5,x_stop/20-5])
#    ax.set_ylim([-(y_stop-y_start)/2/12,(y_stop-y_start)/2/12])
#    ax.set_zlim([-(z_stop-z_start)/2/12,(z_stop-z_start)/2/12])
  cbar = plt.colorbar(im2,  ticks=np.linspace(-eee, eee, 5))
  cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=20)
  cbar.set_label(r'$E_x\ [m_ec\omega/|e|]$',fontdict=font)



#  ax.scatter(grid_x,grid_z,c=color_index,s=makersize*10, alpha=0.8,zdir='y',zs=8,cmap='magma',marker='.',edgecolors='face',lw=0)
#  ax.scatter(grid_x_2,grid_z_2,c='springgreen',s=makersize*1, alpha=0.5,zdir='y',zs=19,marker='.')
#  ax.scatter(grid_x,grid_y,c=color_index,s=makersize*10, alpha=0.8,zdir='z',zs=-8,cmap='magma',marker='.',edgecolors='face',lw=0)
#  ax.scatter(grid_x_2,grid_y_2,c='springgreen',s=makersize*1, alpha=0.5,zdir='z',zs=-19,marker='.')
  ax.scatter(grid_y,grid_z,c=color_index,s=makersize*10, alpha=0.8,zdir='x',zs=-1,cmap='magma',marker='.',edgecolors='face',lw=0)
#  ax.scatter(grid_y_2,grid_z_2,c='springgreen',s=makersize*1, alpha=0.5,zdir='x',zs=0,marker='.')


#  plt.text(-100,650,' t = '++' fs',fontdict=font)
  plt.subplots_adjust(left=0.16, bottom=None, right=0.97, top=None,
                wspace=None, hspace=None)
  plt.title('At '+str(round(time1/1.0e-15,2))+' fs',fontdict=font)
#plt.show()
#lt.figure(figsize=(100,100))


  fig = plt.gcf()
  fig.set_size_inches(20, 10.5)
  fig.savefig(to_path+'3d_comb_ek_ex_ey_'+str(n).zfill(4)+'.png',format='png',dpi=160)
  plt.close("all")
  print(to_path+'3d_comb_ek_ex_ey_'+str(n).zfill(4)+'.png')
#  return 0

if __name__ == '__main__':
  start   =  1  # start time
  stop    =  12  # end time
  step    =  1  # the interval or step
    
  inputs = range(start,stop+step,step)
  pool = mp.Pool(processes=1)
  results = pool.map(processplot,inputs)
  print(results)
