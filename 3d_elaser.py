import numpy as np
import sdf 
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from numpy import ma
from matplotlib import colors, ticker, cm
from matplotlib.mlab import bivariate_normal
from functools import reduce
import multiprocessing as mp
import sys, getopt
import os, time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors


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
        'size'   : 25 ,  
        }  
font_size = 20

c_red  = matplotlib.colors.colorConverter.to_rgba('red')
c_blue = matplotlib.colors.colorConverter.to_rgba('blue')
c_black = matplotlib.colors.colorConverter.to_rgba('black')
c_green= matplotlib.colors.colorConverter.to_rgba('lime')
c_orange= matplotlib.colors.colorConverter.to_rgba('orange')
c_limegreen= matplotlib.colors.colorConverter.to_rgba('limegreen')
c_cyan= matplotlib.colors.colorConverter.to_rgba('cyan')
c_yellow= matplotlib.colors.colorConverter.to_rgba('yellow')

c_white_trans = matplotlib.colors.colorConverter.to_rgba('white',alpha = 0.0)
cmap_mycolor1 = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',[c_white_trans,c_red,c_green],128)
cmap_my_rw = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',[c_red,c_white_trans],128)
cmap_my_bw = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',[c_blue,c_white_trans],128)
cmap_my_bwr = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',[c_blue,c_blue,c_white_trans,c_red,c_red],128)
cmap_my_laser = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',[c_limegreen,c_white_trans,c_white_trans,c_white_trans,c_orange],128)
cmap_my_laser2 = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',[c_cyan,c_white_trans,c_white_trans,c_white_trans,c_yellow],128)
cmap_my_kw = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',[c_black,c_white_trans],128)
cmap_my_wr = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',[c_white_trans,c_red],128)
cmap_my_wb = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',[c_white_trans,c_blue],128)
cmap_my_wk = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',[c_white_trans,c_black],128)

def rebin3d(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1],shape[2],a.shape[2]//shape[2]
    return a.reshape(sh).mean(-1).mean(3).mean(1)

def reg_cmap_transparent(iname,alpha):
    oname = iname + '_transparent'
    cmap = plt.get_cmap(iname)
    values = np.linspace(0,1,256)
    colors = cmap(values)
    for i in range(256):
        colors[i][3] = alpha[i]
    colorlist = [(values[i],colors[i]) for i in range(256)]
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list(oname,colorlist)
    plt.cm.register_cmap(cmap=cmap)
    return cmap

def create_alpha(func):
    return [ 1 if func(i)>1 else 0 if func(i)<0 else func(i) for i in range(256)]


def processplot(n): 
  #for n in range(start,stop+step,step):
    from_path='./spin_a70_n15_n5/'
    to_path  ='./spin_a70_n15_n5_fig/'
    x_start=0; x_stop=300; y_start=0; y_stop=200; z_start=0; z_stop=200;
    x_size = x_stop-x_start; y_size = y_stop-y_start; z_size = z_stop-z_start
    name = 'Ey_laser'

    data = sdf.read(from_path+'fields'+str(n).zfill(4)+'.sdf',dict=True)
    header=data['Header']
    time =header['time']
    x    = data['Grid/Grid_mid'].data[0]/1.e-6
    y    = data['Grid/Grid_mid'].data[1]/1.e-6
    z    = data['Grid/Grid_mid'].data[2]/1.e-6
    var1  = data['Electric Field/Ey'].data/exunit
    var2  = 0 # data['Electric Field/Ez_averaged'].data/exunit
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
    norm = True

    index = 6.0
    _abs  = True # True is for ex; False is for density
    log   = False
    elev  = None
    azim  = None

    if _abs:
        norm = 0
        _min = max(np.max(var),np.min(var))**(0.002**(1.0/index)) if log else max(np.max(var),np.min(var))*0.002**(1.0/index)
        plt.set_cmap(reg_cmap_transparent('bwr',create_alpha(lambda x:abs(x/127.5-1)**index)))
    else:
        _min = np.max(var)**(0.002**(1.0/index)) if log else np.max(var)*0.002**(1.0/index)
        plt.set_cmap(reg_cmap_transparent('hsv_r',create_alpha(lambda x:abs(x/255.0)**index)))

        #special code
        _min = max(_min,1.1e27*0.8)
    #    var._cutrange(lambda x : x[1] < 3)

    if log:
        plotkws['norm'] = matplotlib.colors.LogNorm()

    #var.cutrange(_min=_min,_abs=_abs)
    #point_scatter3D(var,norm=norm,plotkws=plotkws)
    #def point_scatter3D(var,elev=None,azim=None,hold=False,iso=False,norm=None,plotkws={}):
    cmap = plt.get_cmap()
    if norm is not None:
        v0 = np.min(var) - norm
        v1 = np.max(var) - norm
        if abs(v0/v1) > 1:
            low = 0
            high = 0.5 * (1 - v1/v0)
        else:
            low = 0.5 * (1 + v0/v1)
            high = 1.0

        cmap = plt.cm.colors.LinearSegmentedColormap.from_list('tr',
                cmap(np.linspace(low,high,256)))

    eee = 50.0
    var[var>eee] = eee
    var[var<-eee]=-eee
#    print('here1')
    fig = plt.figure()
    ax  = plt.axes(projection='3d')
    ax.view_init(elev=elev, azim=azim)
#    print('here2')
    cut_quarter = False #True
    if cut_quarter:
        var_quater=var
        var_quater[(Y<0)&(Z>0)]=0
        im = ax.scatter(X, Y, Z, c=var_quater, norm=colors.Normalize(vmin=-eee,vmax=eee), cmap=cmap_my_laser2, **plotkws)
    else:
        im = ax.scatter(X, Y, Z, c=var, norm=colors.Normalize(vmin=-eee,vmax=eee), cmap=cmap_my_laser2, **plotkws)
#    print('here3')
    ax.set_xlabel('\n\nX'+ '[$\mu m$]',fontdict=font)
    ax.set_ylabel('\n\nY'+ '[$\mu m$]',fontdict=font)
    ax.set_zlabel('\n\nZ'+ '[$\mu m$]',fontdict=font)

#    ax.set_xlim([x_start/20-2,x_stop/20-2])
#    ax.set_ylim([-(y_stop-y_start)/2/10,(y_stop-y_start)/2/10])
#    ax.set_zlim([-(z_stop-z_start)/2/10,(z_stop-z_start)/2/10])

    ax.set_xlim([-1,12])
    ax.set_ylim([-8.,8])
    ax.set_zlim([-8.,8])
    
    #cbar=plt.colorbar(im, ticks=np.linspace(np.min(color_index), np.max(color_index), 5) ,pad=0.01)
#    cbar=plt.colorbar(im, pad=0.01, ticks=np.linspace(-50, 50, 5))
#    cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=20)
#    cbar.set_label(name+r'$[m_ec\omega/|e|]$',fontdict=font)
    #cbar.set_clim(300,600)

    #print('here4')

    for t in ax.xaxis.get_major_ticks(): t.label.set_fontsize(font_size)
    for t in ax.yaxis.get_major_ticks(): t.label.set_fontsize(font_size)
    for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(font_size)



#    ax.scatter(X,Z,c=var, zdir='y',zs=(y_stop-y_start)/2/12, marker='.', edgecolors='none', cmap=cmap)
#    ax.scatter(X,Y,c=var, zdir='z',zs=-(z_stop-z_start)/2/12,  marker='.', edgecolors='none', cmap=cmap)
#    ax.scatter(Y,Z,c=var, zdir='x',zs=x_start/20-5,  marker='.', edgecolors='none', cmap=cmap)

    #plot for y_z plane
#    Y,Z  = np.meshgrid(y,z,indexing='ij')
#    eexx = (var1**2+var2**2)**0.5
#    ex = (eexx[200-1,:,:]+eexx[200,:,:])/2
#    ex = ex[y_start:y_stop,z_start:z_stop]
#    eee = 20#np.max([np.max(ex),abs(np.min(ex))])
#    ex[ex>eee] =eee
#    Y  = Y[y_start:y_stop,z_start:z_stop]
#    Z  = Z[y_start:y_stop,z_start:z_stop]
#    levels = np.linspace(0, eee, 40)
#    im2=ax.contourf(ex.T, Y.T, Z.T, levels=levels, norm=mcolors.Normalize(vmin=0, vmax=eee), cmap=cm.gray_r, zdir='x', offset=x_start/20-2)
#    ax.set_xlim([x_start/20-5,x_stop/20-5])
#    ax.set_xlim([-(y_stop-y_start)/2/12,(y_stop-y_start)/2/12])
#    ax.set_ylim([-(z_stop-z_start)/2/12,(z_stop-z_start)/2/12])
#    cbar = plt.colorbar(im2,  ticks=np.linspace(0, eee, 3))
#    cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=20)
#    cbar.set_label(name+r'$[m_ec\omega/|e|]$',fontdict=font)
    

    #plot for x_z plane
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
    ax.contourf(X.T, ex.T, Z.T, levels=levels, norm=mcolors.Normalize(vmin=-eee, vmax=eee), cmap=cmap_my_bwr, zdir='y', offset=8.0)
#    ax.set_xlim([x_start/20-5,x_stop/20-5])
#    ax.set_ylim([-(y_stop-y_start)/2/12,(y_stop-y_start)/2/12])
#    ax.set_ylim([-(z_stop-z_start)/2/12,(z_stop-z_start)/2/12])

    #plot for x_y plane
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
    im2=ax.contourf(X.T, Y.T, ex.T, levels=levels, norm=mcolors.Normalize(vmin=-eee, vmax=eee), cmap=cmap_my_bwr, zdir='z', offset=-8.0)
#    ax.set_xlim([x_start/20-5,x_stop/20-5])
#    ax.set_ylim([-(y_stop-y_start)/2/12,(y_stop-y_start)/2/12])
#    ax.set_zlim([-(z_stop-z_start)/2/12,(z_stop-z_start)/2/12])
    cbar = plt.colorbar(im2,  ticks=np.linspace(-eee, eee, 5))
    cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=20)
    cbar.set_label(r'$\overline{E}_x\ [m_ec\omega_0/|e|]$',fontdict=font)

    #ax.scatter(X,Z,c=var,**plotkws ,zdir='y',zs=4)
    #ax.scatter(X,Y,c=var,**plotkws, zdir='z',zs=-4)
    #ax.scatter(Y,Z,c=var,**plotkws, zdir='x',zs=15)
    ax.view_init(elev=45, azim=-45)

    plt.show()
#    ax.grid(False)
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    #ax.grid(linestyle='None', linewidth='0.5', color='white')
    plt.subplots_adjust(left=0.16, bottom=None, right=0.97, top=None,
                    wspace=None, hspace=None)
  #  plt.title('At '+str(round(time/1.0e-15,2))+' fs',fontdict=font)


    fig = plt.gcf()
    fig.set_size_inches(12, 10.5)
    fig.savefig(to_path+'3d_'+name+str(n).zfill(4)+'.png',format='png',dpi=80)
    plt.close("all")
    print('finised '+str(n).zfill(4))
    #print('here5')

if __name__ == '__main__':
  start   =  12 # start time
  stop    =  12  # end time
  step    =  1  # the interval or step
    
  inputs = range(start,stop+step,step)
  pool = mp.Pool(processes=1)
  results = pool.map(processplot,inputs)
  print(results)
