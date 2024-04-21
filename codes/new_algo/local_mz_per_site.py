from qutip import sigmax, sigmay, sigmaz
import numpy as np
from scipy import signal
import random
import matplotlib.pyplot as plt
from scipy.special import jn_zeros
from itertools import combinations
from multiprocessing import Pool
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import ImageGrid
from funcs import *

freezing_pts = jn_zeros(0, 5)
plt.rcParams.update({"figure.figsize": (6,6),"text.usetex": True,"font.family": "sans-serif",
    "font.size": 12,"font.sans-serif": ["Helvetica"]})

N = 8
N1= int(N/2)
N2= N-N1
omega = 20.0
w = omega
T = 2 * np.pi/w
times = np.linspace(0, 8 * T, 81, endpoint=False)

ea, eb = 0.03, 0.9
ft = 10    # fintsize
g = np.pi/T
Jvalue = 0.2/T     # finite interaction strong coupling
#Jvalue = 0.072/T     # finite interaction weak coupling
betas = [0.0, 1.5, 2.5, float('inf')]
asps = [0.3,0.3,0.3,0.3]
hpts = [jn_zeros(0,3)[0]]
labels = ['a','b', 'c', 'd', 'e', 'f', 'g', 'h']

lamb = 0.0 
lambd_x = lamb
lambd_y = lamb

figname = 'mz_long_nonfr_weakJ.jpeg'
nprocs = N
p = Pool(processes = nprocs) 
opts = Options(nsteps=1e5, num_cpus=1, openmp_threads=1)

spinposition_A = [0,1,2,3]
spinposition_B = [4,5,6,7] 

h0 = 0.0
ll = 0
fig = plt.figure()
grid = ImageGrid(fig, 111,nrows_ncols = (1,4),axes_pad = 0.04,
                 cbar_location = "right",
                 cbar_mode="single",cbar_size="4%", cbar_pad=0.05
                )
sx,sy,sz = sigmax(), sigmay(), sigmaz()
xx = 0.0
for hh,hpt in enumerate(hpts):
    h = hpt * w/4
    for b, beta in enumerate(betas):
        sz_os = [] 
        xx = int(xx)
        print('beta=',beta,'xx=',xx)
        datap = np.zeros((N, len(times)))
        for m,i in enumerate(spinposition_A):
            id = qeye(2**i)    
            dim12 = N-1-i
            id1 = qeye(2**dim12)
            sz_os.append(Qobj(tensor(id,tensor(sz,id1)).full()))
            
        for m,i in enumerate(spinposition_B):
            id = qeye(2**i)    
            dim12 = N-1-i
            id1 = qeye(2**dim12)
            sz_os.append(Qobj(tensor(id,tensor(sz,id1)).full()))

        params = [{'h0':0, 'h':h, 'omega':omega, 'N':N,'N1':N1,\
                   'opts':opts, 'sz_o':sz_o, 'lambd_x':lambd_x,\
                   'lambd_y':lambd_y, 'Jvalue':Jvalue,'beta':beta,\
                   'g':g,'ea':ea,'eb':eb, 'times':times} for sz_o in sz_os]

        #data = p.map(run_dynm,tqdm(params, position=0, leave=True))
        data = p.map(run_dynm,params)
        for i in range(N):
            datap[i] = data[i][0]

        imc = grid[xx].imshow(datap.T, aspect=asps[b],interpolation="nearest",\
                    cmap='bwr', origin='lower',extent = [0 , N, times[0]/T , times[-1]/T],\
                    vmax=1, vmin=-1)
        grid[xx].text(6, 40, labels[xx], fontsize=10, bbox=dict(facecolor='white', alpha=0.95))
        for i in range(N):
            grid[xx].axvline(i,color = 'black', linewidth=0.7)
        grid[xx].set_xlabel(r'$Site(i)$', fontsize=11, labelpad=0.0)
        grid[xx].set_ylabel(r'$t/T$', fontsize=10, labelpad=0.0)
        
        #grid[xx].tick_params(which='both', axis="x", direction="in")
        grid[xx].tick_params(which='both', axis="y", direction="in")
        xx=xx+1
        
cbticks = np.linspace(-1,1,5)
#plt.colorbar(imc, cax=grid.cbar_axes[0], label=r"$\langle\hat{S_z}\rangle$") 
clb = plt.colorbar(imc, cax=grid.cbar_axes[0], ticks= cbticks) 
clb.ax.set_title(label=r"$\langle\hat{S_z}\rangle$", fontsize = 10)     
figname = "sz_t_strongJ_N_" + str(N) +'.jpeg'
#plt.savefig(figname, bbox_inches='tight', pad_inches=0.0, dpi=600)
