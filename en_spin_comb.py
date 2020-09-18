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

  from_path='./spin_a70_n15_n5_fine/'
  to_path  ='./spin_a70_n15_n5_fine_fig/'

  ######### Parameter you should set ###########
  start   =  0  # start time
  stop    =  224  # end time
  step    =  5  # the interval or step
  
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
    condition = ((ion_y**2+ion_z**2)**0.5<1.0)&(ion_px>0.01)
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
    plt.plot(grid_en,value_en,color='k',linewidth=3,label='ion_s')
    plt.xlabel('Energy [MeV]',fontdict=font)
    plt.ylabel('dN/dE [MeV${-1}$]',fontdict=font)
    plt.xticks(fontsize=font_size); 
    plt.yticks(fontsize=font_size);
    plt.yscale('log')
    plt.xlim(0,1000)
#    plt.ylim(1e1,1e5)
    plt.legend(loc='best',fontsize=font_size2,framealpha=0.5)
    plt.title('t='+str(round(time/1.0e-15,0))+' fs',fontdict=font)

    plt.subplot(2,2,2)
    plt.scatter(ion_x, ion_px, c=ion_sx, cmap='rainbow', norm=colors.Normalize(vmin=0.9,vmax=1), s=marker_size*2, marker='.',alpha=0.8,label='ion_s',zorder=3,lw=0)
    print(np.max(ion_sx),np.min(ion_sx))
    cbar=plt.colorbar(pad=0.01)
#  cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=20)
    cbar.ax.tick_params(labelsize=font_size2)
    cbar.set_label(r'$s_x$',fontdict=font)
    plt.legend(loc='best',fontsize=font_size2)
    plt.xlim(0,12)
    plt.ylim(0,1.4)
    plt.xlabel('$x\ [\mu m]$',fontdict=font)
    plt.ylabel('$p_x\ [m_ic]$',fontdict=font)
    plt.xticks(fontsize=font_size); 
    plt.yticks(fontsize=font_size);

    plt.subplot(2,2,3)
    condition1=(ek>200)
    if np.size(condition1) >=1:
        sy= ion_sy[condition1]
        sz= ion_sz[condition1]
        Q=plt.quiver(ion_y[condition1], ion_z[condition1], sy/(sy**2+sz**2)**0.5, sz/(sy**2+sz**2)**0.5, pivot='mid', units='inches')
#        qk = plt.quiverkey(Q, 0.9, 0.9, 0.1, r'$s$', labelpos='E', coordinates='figure')
#        plt.scatter(ion_y[condition1],ion_z[condition1], color='r', s=1)
        plt.xlim(-2,2)
        plt.ylim(-2,2)
        plt.xlabel('$x\ [\mu m]$',fontdict=font)
        plt.ylabel('$y\ [\mu m]$',fontdict=font)
        plt.xticks(fontsize=font_size); 
        plt.yticks(fontsize=font_size);

    plt.subplot(2,2,4)
    plt.scatter(theta, ek, c=ion_sx, cmap='rainbow', norm=colors.Normalize(vmin=0.9,vmax=1), s=marker_size*2, marker='.',alpha=0.8,label='ion_s',zorder=3,lw=0)
    cbar=plt.colorbar(ticks=[0.9,0.95,1],pad=0.01)
#  cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=20)
    cbar.ax.tick_params(labelsize=font_size2)
    cbar.set_label(r'$s_x$',fontdict=font)
    plt.legend(loc='best',fontsize=font_size2)
    plt.xlim(-1,45)
    plt.ylim(0,1000)
    plt.xlabel(r'$\theta [^\circ]$',fontdict=font)
    plt.ylabel('Energy [MeV]',fontdict=font)
    plt.xticks(fontsize=font_size); 
    plt.yticks(fontsize=font_size);


    plt.subplots_adjust(left=0.12, bottom=0.1, right=0.98, top=0.97, wspace=0.1, hspace=0.1)


    fig = plt.gcf()
    fig.set_size_inches(18, 14.)
    fig.savefig(to_path+'en_spin_comb_'+str(n).zfill(4)+'.png',format='png',dpi=80)
    plt.close("all")
    print('finised '+str(round(100.0*(n-start+step)/(stop-start+step),4))+'%')
