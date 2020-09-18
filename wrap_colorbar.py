#!/public/home/users/bio001/tools/python-2.7.11/bin/python
import sdf
import matplotlib
matplotlib.use('agg')
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
          'size'   : 25,  
          }  
  font2 = {'family' : 'monospace',  
          'color'  : 'black',  
          'weight' : 'normal',  
          'size'   : 16,  
          } 
  space_1 = 5
  space_2 = 5
  font_size = 25
  marker_size=0.05
##below is for generating mid transparent colorbar
  c_red = matplotlib.colors.colorConverter.to_rgba('red')
  c_blue= matplotlib.colors.colorConverter.to_rgba('blue')
  c_yellow= matplotlib.colors.colorConverter.to_rgba('yellow')
  c_cyan= matplotlib.colors.colorConverter.to_rgba('cyan')
  c_black = matplotlib.colors.colorConverter.to_rgba('black')
  c_green= matplotlib.colors.colorConverter.to_rgba('limegreen')
  c_white_trans = matplotlib.colors.colorConverter.to_rgba('white',alpha = 0.0)
  cmap_rb = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',[c_red,c_white_trans,c_blue],128) 
  cmap_br = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',[c_blue,c_white_trans,c_red],128)
  cmap_ryb= matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',[c_red,c_yellow,c_blue],128)
  cmap_yc = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',[c_yellow,c_white_trans,c_cyan],128) 
  cmap_wg = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',[c_white_trans,c_green],128)

  cmap_my_rw = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',[c_red,c_white_trans],128)
  cmap_my_bw = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',[c_blue,c_white_trans],128)
  cmap_my_kw = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',[c_black,c_white_trans],128)
  cmap_my_wr = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',[c_white_trans,c_red],128)
  cmap_my_wb = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',[c_white_trans,c_blue],128)
  cmap_my_wk = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',[c_white_trans,c_black],128)

  cmap_my_laser = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',[c_blue,c_blue,c_white_trans,c_red,c_red],128)
  cmap_my_laser2 = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',[c_cyan,c_white_trans,c_white_trans,c_white_trans,c_yellow],128)

  c_brown = matplotlib.colors.colorConverter.to_rgba('gold')
  c_green = matplotlib.colors.colorConverter.to_rgba('springgreen')
  cmap_bg = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',[c_brown,c_white_trans,c_white_trans,c_green],128)
  c_black = matplotlib.colors.colorConverter.to_rgba('black')
  c_white = matplotlib.colors.colorConverter.to_rgba('white')
  cmap_bw = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',[c_white_trans,c_white,c_black],128)
   
##end for transparent colorbar##
  upper = matplotlib.cm.jet(np.arange(256))
  lower = np.ones((int(256/4),4))
  for i in range(3):
      lower[:,i] = np.linspace(1, upper[0,i], lower.shape[0])
  cmap = np.vstack(( lower, upper ))
  mycolor_jet = matplotlib.colors.ListedColormap(cmap, name='myColorMap', N=cmap.shape[0])

  upper = matplotlib.cm.pink_r(np.arange(256))
  lower = np.ones((int(256/4),4))
  for i in range(3):
      lower[:,i] = np.linspace(1, upper[0,i], lower.shape[0])
  cmap = np.vstack(( lower, upper ))
  mycolor_pink_r = matplotlib.colors.ListedColormap(cmap, name='myColorMap', N=cmap.shape[0])

  upper = matplotlib.cm.Greens(np.arange(256))
  lower = np.ones((int(256/4),4))
  for i in range(3):
      lower[:,i] = np.linspace(1, upper[0,i], lower.shape[0])
  cmap = np.vstack(( lower, upper ))
  mycolor_Greens = matplotlib.colors.ListedColormap(cmap, name='myColorMap', N=cmap.shape[0])
 
  upper = matplotlib.cm.viridis(np.arange(256))
  lower = np.ones((int(256/4),4))
  for i in range(3):
      lower[:,i] = np.linspace(1, upper[0,i], lower.shape[0])
  cmap = np.vstack(( lower, upper ))
  mycolor_viridis = matplotlib.colors.ListedColormap(cmap, name='myColorMap', N=cmap.shape[0])
 
  upper = matplotlib.cm.rainbow(np.arange(256))
  lower = np.ones((int(256/4),4))
  for i in range(3):
      lower[:,i] = np.linspace(1, upper[0,i], lower.shape[0])
  cmap = np.vstack(( lower, upper ))
  mycolor_rainbow = matplotlib.colors.ListedColormap(cmap, name='myColorMap', N=cmap.shape[0])
 
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

  color_list = ['blue','limegreen','red'] 
  
if __name__ == '__main__':
#    fig = plt.figure()
#    ax1 = fig.add_axes([0.1, 0.2, 0.85, 0.15])
#    cb1 = matplotlib.colorbar.ColorbarBase(ax1, cmap=mycolor_Greens, norm=colors.LogNorm(vmin=1e-2,vmax=1e2), 
#                                    orientation='horizontal',ticks=[1e-2,1e-1,1e0,1e1,1e2])
#    cb1.ax.xaxis.set_label_position('top'); cb1.ax.xaxis.set_ticks_position('top')
#    cb1.ax.tick_params(labelsize=font_size) 
#    cb1.set_label('$n_e$ [$n_c$]',fontdict=font)
#    plt.subplots_adjust(top=0.98, bottom=0.13, left=0.15, right=0.85, hspace=0.10, wspace=0.05)
#    fig = plt.gcf()
#    fig.set_size_inches(7, 2)
#    plt.show()
#    fig.savefig('./colorbar_fig/den_field_greens.png',format='png',dpi=160, transparent=True)
#    plt.close("all")

    fig = plt.figure()
    ax1 = fig.add_axes([0.1, 0.2, 0.5, 0.15])
    cb1 = matplotlib.colorbar.ColorbarBase(ax1, cmap=cmap_br, norm=colors.Normalize(vmin=-30,vmax=30), 
                                    orientation='horizontal',ticks=[-30,0,30])
    cb1.ax.xaxis.set_label_position('top'); cb1.ax.xaxis.set_ticks_position('top')
    cb1.ax.tick_params(labelsize=font_size) 
    cb1.set_label('$E_x$ [$m_ec\omega_0/|e|$]',fontdict=font)
    plt.subplots_adjust(top=0.98, bottom=0.13, left=0.15, right=0.85, hspace=0.10, wspace=0.05)
    fig = plt.gcf()
    fig.set_size_inches(7, 2)
    plt.show()
    fig.savefig('./colorbar_fig/evo_ex.png',format='png',dpi=160, transparent=True)
    plt.close("all")

    fig = plt.figure()
    ax1 = fig.add_axes([0.1, 0.2, 0.5, 0.15])
    cb1 = matplotlib.colorbar.ColorbarBase(ax1, cmap='magma', norm=colors.Normalize(vmin=0,vmax=150), 
                                    orientation='horizontal',ticks=[0,75,150])
    cb1.ax.xaxis.set_label_position('top'); cb1.ax.xaxis.set_ticks_position('top')
    cb1.ax.tick_params(labelsize=font_size) 
    cb1.set_label(r'$\varepsilon_p$ [MeV]',fontdict=font)
    plt.subplots_adjust(top=0.98, bottom=0.13, left=0.15, right=0.85, hspace=0.10, wspace=0.05)
    fig = plt.gcf()
    fig.set_size_inches(7, 2)
    plt.show()
    fig.savefig('./colorbar_fig/evo_trace_ek.png',format='png',dpi=160, transparent=True)
    plt.close("all")

    fig = plt.figure()
    ax1 = fig.add_axes([0.1, 0.2, 0.5, 0.15])
    cb1 = matplotlib.colorbar.ColorbarBase(ax1, cmap=cmap_my_bw, norm=colors.Normalize(vmin=-0.15,vmax=-0.05), 
                                    orientation='horizontal',ticks=[-0.15,-0.10,-0.05])
    cb1.ax.xaxis.set_label_position('top'); cb1.ax.xaxis.set_ticks_position('top')
    cb1.ax.tick_params(labelsize=font_size) 
    cb1.set_label('$\mathcal{H}_2$ [$m_pc^2$]',fontdict=font)
    plt.subplots_adjust(top=0.98, bottom=0.13, left=0.15, right=0.85, hspace=0.10, wspace=0.05)
    fig = plt.gcf()
    fig.set_size_inches(7, 2)
    plt.show()
    fig.savefig('./colorbar_fig/hami_theory_blue.png',format='png',dpi=160, transparent=True)
    plt.close("all")

    fig = plt.figure()
    ax1 = fig.add_axes([0.1, 0.2, 0.5, 0.15])
    cb1 = matplotlib.colorbar.ColorbarBase(ax1, cmap=cmap_my_kw, norm=colors.Normalize(vmin=-0.02,vmax=0.05), 
                                    orientation='horizontal',ticks=[-0.02,0.00,0.02,0.04])
    cb1.ax.xaxis.set_label_position('top'); cb1.ax.xaxis.set_ticks_position('top')
    cb1.ax.tick_params(labelsize=font_size) 
    cb1.set_label('$\mathcal{H}_1$ [$m_pc^2$]',fontdict=font)
    plt.subplots_adjust(top=0.98, bottom=0.13, left=0.15, right=0.85, hspace=0.10, wspace=0.05)
    fig = plt.gcf()
    fig.set_size_inches(7, 2)
    plt.show()
    fig.savefig('./colorbar_fig/hami_theory_black.png',format='png',dpi=160, transparent=True)
    plt.close("all")

    fig = plt.figure()
    ax1 = fig.add_axes([0.1, 0.2, 0.5, 0.15])
    cb1 = matplotlib.colorbar.ColorbarBase(ax1, cmap='rainbow', norm=colors.Normalize(vmin=25,vmax=70), 
                                    orientation='horizontal',ticks=[30,50,70])
    cb1.ax.xaxis.set_label_position('top'); cb1.ax.xaxis.set_ticks_position('top')
    cb1.ax.tick_params(labelsize=font_size) 
    cb1.set_label(r'$t$ [fs]',fontdict=font)
    plt.subplots_adjust(top=0.98, bottom=0.13, left=0.15, right=0.85, hspace=0.10, wspace=0.05)
    fig = plt.gcf()
    fig.set_size_inches(7, 2)
    plt.show()
    fig.savefig('./colorbar_fig/hami_theory_trajectory_rainbow.png',format='png',dpi=160, transparent=True)
    plt.close("all")

    fig = plt.figure()
    ax1 = fig.add_axes([0.1, 0.2, 0.8, 0.15])
    cb1 = matplotlib.colorbar.ColorbarBase(ax1, cmap='magma', norm=colors.Normalize(vmin=0,vmax=300), 
                                    orientation='horizontal',ticks=[0,100,200,300])
    cb1.ax.xaxis.set_label_position('top'); cb1.ax.xaxis.set_ticks_position('top')
    cb1.ax.tick_params(labelsize=font_size) 
    cb1.set_label(r'$\varepsilon_p$ [MeV]',fontdict=font)
    plt.subplots_adjust(top=0.98, bottom=0.13, left=0.15, right=0.85, hspace=0.10, wspace=0.05)
    fig = plt.gcf()
    fig.set_size_inches(7, 2)
    plt.show()
    fig.savefig('./colorbar_fig/3d_ek.png',format='png',dpi=160, transparent=True)
    plt.close("all")

    fig = plt.figure()
    ax1 = fig.add_axes([0.1, 0.2, 0.5, 0.15])
    cb1 = matplotlib.colorbar.ColorbarBase(ax1, cmap=cmap_my_laser, norm=colors.Normalize(vmin=-30,vmax=30), 
                                    orientation='horizontal',ticks=[-30,0,30])
    cb1.ax.xaxis.set_label_position('top'); cb1.ax.xaxis.set_ticks_position('top')
    cb1.ax.tick_params(labelsize=font_size) 
    cb1.set_label(r'$\overline{E}_x\ [m_ec\omega_0/|e|]$',fontdict=font)
    plt.subplots_adjust(top=0.98, bottom=0.13, left=0.15, right=0.85, hspace=0.10, wspace=0.05)
    fig = plt.gcf()
    fig.set_size_inches(7, 2)
    plt.show()
    fig.savefig('./colorbar_fig/3d_ex.png',format='png',dpi=160, transparent=True)
    plt.close("all")

    fig = plt.figure()
    ax1 = fig.add_axes([0.1, 0.2, 0.5, 0.15])
    cb1 = matplotlib.colorbar.ColorbarBase(ax1, cmap=cmap_my_laser2, norm=colors.Normalize(vmin=-50,vmax=50), 
                                    orientation='horizontal',ticks=[-50,0,50])
    cb1.ax.xaxis.set_label_position('top'); cb1.ax.xaxis.set_ticks_position('top')
    cb1.ax.tick_params(labelsize=font_size) 
    cb1.set_label(r'$E_y\ [m_ec\omega_0/|e|]$',fontdict=font)
    plt.subplots_adjust(top=0.98, bottom=0.13, left=0.15, right=0.85, hspace=0.10, wspace=0.05)
    fig = plt.gcf()
    fig.set_size_inches(7, 2)
    plt.show()
    fig.savefig('./colorbar_fig/3d_ey.png',format='png',dpi=160, transparent=True)
    plt.close("all")

    fig = plt.figure()
    ax1 = fig.add_axes([0.1, 0.2, 0.5, 0.15])
    cb1 = matplotlib.colorbar.ColorbarBase(ax1, cmap='viridis', norm=colors.Normalize(vmin=0.7,vmax=1), 
                                    orientation='horizontal',ticks=[0.7,0.8,0.9,1.0])
    cb1.ax.xaxis.set_label_position('top'); cb1.ax.xaxis.set_ticks_position('top')
    cb1.ax.tick_params(labelsize=font_size) 
    cb1.set_label(r'$s_x$',fontdict=font)
    plt.subplots_adjust(top=0.98, bottom=0.13, left=0.15, right=0.85, hspace=0.10, wspace=0.05)
    fig = plt.gcf()
    fig.set_size_inches(7, 2)
    plt.show()
    fig.savefig('./colorbar_fig/3d_sx.png',format='png',dpi=160, transparent=True)
    plt.close("all")

    fig = plt.figure()
    ax1 = fig.add_axes([0.1, 0.2, 0.6, 0.15])
    cb1 = matplotlib.colorbar.ColorbarBase(ax1, cmap=mycolor_viridis, norm=colors.LogNorm(vmin=4e5, vmax=4e9), 
                                    orientation='horizontal',ticks=[1e6,1e7,1e8,1e9])
    cb1.ax.xaxis.set_label_position('top'); cb1.ax.xaxis.set_ticks_position('top')
    cb1.ax.tick_params(labelsize=font_size) 
#    cb1.set_label(r'$\frac{dN}{ds_x}$',fontdict=font)
    plt.subplots_adjust(top=0.98, bottom=0.13, left=0.15, right=0.85, hspace=0.10, wspace=0.05)
    fig = plt.gcf()
    fig.set_size_inches(7, 2)
    plt.show()
    fig.savefig('./colorbar_fig/spin_evo_sx.png',format='png',dpi=160, transparent=True)
    plt.close("all")

    fig = plt.figure()
    ax1 = fig.add_axes([0.1, 0.1, 0.15, 0.5])
    cb1 = matplotlib.colorbar.ColorbarBase(ax1, cmap=cmap_wg, norm=colors.Normalize(vmin=0,vmax=0.3), 
                                    orientation='vertical',ticks=[0,0.1,0.2,0.3])
    cb1.ax.xaxis.set_label_position('top'); cb1.ax.xaxis.set_ticks_position('top')
    cb1.ax.tick_params(labelsize=font_size) 
    cb1.set_label(r'$s_r$',fontdict=font)
    plt.subplots_adjust(top=0.98, bottom=0.13, left=0.15, right=0.85, hspace=0.10, wspace=0.05)
    fig = plt.gcf()
    fig.set_size_inches(2, 5)
    plt.show()
    fig.savefig('./colorbar_fig/spin_contour_sr.png',format='png',dpi=160, transparent=True)
    plt.close("all")

    fig = plt.figure()
    ax1 = fig.add_axes([0.1, 0.1, 0.15, 0.5])
    cb1 = matplotlib.colorbar.ColorbarBase(ax1, cmap='Oranges', norm=colors.Normalize(vmin=0,vmax=0.2), 
                                    orientation='vertical',ticks=[0,0.1,0.2])
    cb1.ax.xaxis.set_label_position('top'); cb1.ax.xaxis.set_ticks_position('top')
    cb1.ax.tick_params(labelsize=font_size) 
    cb1.set_label(r'$B_\phi\ $'+'[MT]',fontdict=font)
    plt.subplots_adjust(top=0.98, bottom=0.13, left=0.15, right=0.85, hspace=0.10, wspace=0.05)
    fig = plt.gcf()
    fig.set_size_inches(2, 5)
    plt.show()
    fig.savefig('./colorbar_fig/azimutial_B_field.png',format='png',dpi=160, transparent=True)
    plt.close("all")

#    fig = plt.figure()
#    ax1 = fig.add_axes([0.1, 0.1, 0.15, 0.8])
#    cb1 = matplotlib.colorbar.ColorbarBase(ax1, cmap='pink_r', norm=colors.Normalize(vmin=0,vmax=0.3), 
#                                    orientation='vertical',ticks=[0,0.1,0.2,0.3])
##    cb1.ax.xaxis.set_label_position('top'); cb1.ax.xaxis.set_ticks_position('top')
#    cb1.ax.tick_params(labelsize=font_size) 
#    cb1.set_label('$\mathcal{H}\ [m_pc^2]$',fontdict=font)
#    plt.subplots_adjust(top=0.98, bottom=0.13, left=0.15, right=0.85, hspace=0.10, wspace=0.05)
#    fig = plt.gcf()
#    fig.set_size_inches(2, 3.5)
#    plt.show()
#    fig.savefig('./colorbar_fig/Hami_pink.png',format='png',dpi=160, transparent=True)
#    plt.close("all")
