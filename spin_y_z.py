import sdf
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from numpy import ma
from matplotlib import colors, ticker, cm
from matplotlib.mlab import bivariate_normal
  
if __name__ == "__main__":
  print ('This is main of module "test2d.py"')
  ######## Constant defined here ########
  pi        =     3.1415926535897932384626
  q0        =     1.602176565e-19 # C
  m0        =     9.10938291e-31  # kg
  v0        =     2.99792458e8    # m/s^2
  kb        =     1.3806488e-23   # J/K
  mu0       =     4.0e-7*np.pi       # N/A^2
  epsilon0  =     8.8541878176203899e-12 # F/m
  h_planck  =     6.62606957e-34  # J s
  wavelength=     0.8e-6
  frequency =     v0*2*pi/wavelength
  
  exunit    =     m0*v0*frequency/q0
  bxunit    =     m0*frequency/q0
  denunit    =     frequency**2*epsilon0*m0/q0**2
  jalf      =     4*np.pi*epsilon0*m0*v0**3/q0/wavelength**2
  print('electric field unit: '+str(exunit))
  print('magnetic field unit: '+str(bxunit))
  print('density unit nc: '+str(denunit))
  
  font = {'family' : 'monospace',  
          'color'  : 'black',  
          'weight' : 'normal',  
          'size'   : 20,  
          }  
 
  font_size=20 
  font_size2=14
  marker_size = 0.1 
##below is for norm colorbar
  class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y)) 
##end for norm colorbar####

  def pxpy_to_energy(gamma, weight):
      binsize = 200
      en_grid = np.linspace(0,1000,200)
      en_bin  = np.linspace(0,1000,201)
      en_value = np.zeros_like(en_grid) 
      for i in range(binsize):
        en_value[i] = sum(weight[ (en_bin[i]<=gamma) & (gamma<en_bin[i+1]) ])/(en_bin[-1]-en_bin[-2])
      return (en_grid, en_value)

  upper = matplotlib.cm.rainbow(np.arange(256))
  lower = np.ones((int(256/4),4))
  for i in range(3):
      lower[:,i] = np.linspace(1, upper[0,i], lower.shape[0])
  cmap = np.vstack(( lower, upper ))
  mycolor_rainbow = matplotlib.colors.ListedColormap(cmap, name='myColorMap', N=cmap.shape[0])


  from_path='./spin_a70_n15_n5_revised/'
  to_path  ='./spin_a70_n15_n5_revised_fig/'

  ######### Parameter you should set ###########
  start   =  0  # start time
  stop    =  20  # end time
  step    =  1  # the interval or step
  
  ######### Script code drawing figure ################
  for n in range(start,stop+step,step):
    #### header data ####
    data = sdf.read(from_path+'track'+str(n).zfill(4)+".sdf",dict=True)
    header=data['Header']
    time=header['time']
    print('reading',from_path+'track'+str(n).zfill(4)+".sdf")
    ion_x = data['Grid/Particles/subset_high_p/ion_s'].data[0]/1e-6
    ion_y = data['Grid/Particles/subset_high_p/ion_s'].data[1]/1e-6
    ion_z = data['Grid/Particles/subset_high_p/ion_s'].data[2]/1e-6
    ion_px = data['Particles/Px/subset_high_p/ion_s'].data/(1836*m0*v0)
    ion_py = data['Particles/Py/subset_high_p/ion_s'].data/(1836*m0*v0)
    ion_pz = data['Particles/Pz/subset_high_p/ion_s'].data/(1836*m0*v0)
    ion_sx = data['Particles/Sx/subset_high_p/ion_s'].data
    ion_sy = data['Particles/Sy/subset_high_p/ion_s'].data
    ion_sz = data['Particles/Sz/subset_high_p/ion_s'].data
    ion_ww = data['Particles/Weight/subset_high_p/ion_s'].data
    condition = ((ion_y**2+ion_z**2)**0.5<10.0)&(ion_px>0.01)
    if np.size(ion_x[condition]) == 0:
        continue
    ion_x = ion_x[condition] 
    ion_y = ion_y[condition] 
    ion_z = ion_z[condition] 
    ion_px = ion_px[condition] 
    ion_py = ion_py[condition] 
    ion_pz = ion_pz[condition]
    ion_sx = ion_sx[condition] 
    ion_sy = ion_sy[condition] 
    ion_sz = ion_sz[condition]
    ion_ww = ion_ww[condition]  
    theta  = np.arctan2((ion_py**2+ion_pz**2)**0.5,ion_px)/np.pi*180 
    gg = (ion_px**2+ion_py**2+ion_pz**2+1)**0.5
    ek = (gg-1)*1836*0.51
  
    grid_en, value_en = pxpy_to_energy(ek, ion_ww) 
 
    plt.subplot(2,2,1)
    condition1=(ek>100)
    if np.size(condition1) >=1:
        sy= ion_sy[condition1]
        sz= ion_sz[condition1]
#        Q=plt.quiver(ion_y[condition1], ion_z[condition1], sy/(sy**2+sz**2)**0.5, sz/(sy**2+sz**2)**0.5, pivot='mid', units='inches')
        plt.scatter(ion_y[condition1], ion_z[condition1], c=ion_sy[condition1], cmap='bwr', norm=colors.Normalize(vmin=-0.3,vmax=0.3), s=marker_size*100, marker='.',alpha=0.8,label='ion_s',zorder=3,lw=0)
        cbar=plt.colorbar(ticks=[-0.3,0,0.3],pad=0.01)
#  cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=20)
        cbar.ax.tick_params(labelsize=font_size2)
        cbar.set_label(r'$s_y$',fontdict=font)
#        qk = plt.quiverkey(Q, 0.9, 0.9, 0.1, r'$s$', labelpos='E', coordinates='figure')
#        plt.scatter(ion_y[condition1],ion_z[condition1], color='r', s=1)
        plt.xlim(-6,6)
        plt.ylim(-6,6)
        plt.xlabel('$y\ [\mu m]$',fontdict=font)
        plt.ylabel('$z\ [\mu m]$',fontdict=font)
        plt.xticks(fontsize=font_size); 
        plt.yticks(fontsize=font_size);
        plt.title('t='+str(round(time/1.0e-15,0))+' fs',fontdict=font)

    plt.subplot(2,2,2)
    condition1=(ek>100)
    if np.size(condition1) >=1:
        sy= ion_sy[condition1]
        sz= ion_sz[condition1]
#        Q=plt.quiver(ion_y[condition1], ion_z[condition1], sy/(sy**2+sz**2)**0.5, sz/(sy**2+sz**2)**0.5, pivot='mid', units='inches')
        plt.scatter(ion_y[condition1], ion_z[condition1], c=ion_sz[condition1], cmap='bwr', norm=colors.Normalize(vmin=-0.3,vmax=0.3), s=marker_size*100, marker='.',alpha=0.8,label='ion_s',zorder=3,lw=0)
        cbar=plt.colorbar(ticks=[-0.3,0,0.3],pad=0.01)
#  cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=20)
        cbar.ax.tick_params(labelsize=font_size2)
        cbar.set_label(r'$s_z$',fontdict=font)
#        qk = plt.quiverkey(Q, 0.9, 0.9, 0.1, r'$s$', labelpos='E', coordinates='figure')
#        plt.scatter(ion_y[condition1],ion_z[condition1], color='r', s=1)
        plt.xlim(-6,6)
        plt.ylim(-6,6)
        plt.xlabel('$y\ [\mu m]$',fontdict=font)
        plt.ylabel('$z\ [\mu m]$',fontdict=font)
        plt.xticks(fontsize=font_size); 

    plt.subplot(2,2,3)
    data = sdf.read(from_path+'fields'+str(n).zfill(4)+".sdf",dict=True)
    header=data['Header']
    time =header['time']
    x    = data['Grid/Grid_mid'].data[0]/1.e-6
    y    = data['Grid/Grid_mid'].data[1]/1.e-6
    z    = data['Grid/Grid_mid'].data[2]/1.e-6
    var1  = data['Magnetic Field/By_averaged'].data/bxunit
    var2  = data['Magnetic Field/Bz_averaged'].data/bxunit
    Y,Z  = np.meshgrid(y,z)
    eexx = (var1**2+var2**2)**0.5
    ex = np.sum(eexx[120:180,:,:],axis=0)/np.size(eexx[120:180,0,0])
    var1 = np.sum(var1[120:180,:,:],axis=0)/np.size(var1[120:180,0,0])
    var2 = np.sum(var2[120:180,:,:],axis=0)/np.size(var2[120:180,0,0])
    eee = 10#np.max([np.max(ex),abs(np.min(ex))])
    ex[ex>eee] =eee
    levels = np.linspace(0, eee, 40)
    plt.contourf(Y, Z, ex.T, levels=levels, norm=colors.Normalize(vmin=0, vmax=eee), cmap=mycolor_rainbow)
#    ax.set_xlim([x_start/20-5,x_stop/20-5])
#    ax.set_xlim([-(y_stop-y_start)/2/12,(y_stop-y_start)/2/12])
#    ax.set_ylim([-(z_stop-z_start)/2/12,(z_stop-z_start)/2/12])
    cbar = plt.colorbar(pad=0.01, ticks=np.linspace(0, eee, 3))
    cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=20)
    cbar.set_label(r'$B_\phi\ [m_e\omega/|e|]$',fontdict=font)
    Q=plt.quiver(Y[::5,::5], Z[::5,::5], (var1.T)[::5,::5], (var2.T)[::5,::5], pivot='mid', units='x',scale=10)
    plt.xlim(-6,6)
    plt.ylim(-6,6)
    plt.xlabel('$y\ [\mu m]$',fontdict=font)
    plt.ylabel('$z\ [\mu m]$',fontdict=font)
    plt.xticks(fontsize=font_size); 
    plt.yticks(fontsize=font_size); 


    plt.subplots_adjust(left=0.12, bottom=0.1, right=0.98, top=0.97, wspace=0.1, hspace=0.1)


    fig = plt.gcf()
    fig.set_size_inches(18, 14.)
    fig.savefig(to_path+'spin_y_z_'+str(n).zfill(4)+'.png',format='png',dpi=160)
    plt.close("all")
    print(to_path+'spin_y_z_'+str(n).zfill(4)+'.png')
