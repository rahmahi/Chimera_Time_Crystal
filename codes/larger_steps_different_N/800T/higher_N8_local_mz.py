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
import h5py

freezing_pts = jn_zeros(0, 5)


N = 8
N1= int(N/2)
N2= N-N1
omega = 20.0
w = omega
T = 2 * np.pi/w
steps = 8001
times = np.linspace(0, 800 * T, steps, endpoint=False)

ea, eb = 0.03, 0.9
ft = 10    # fintsize
g = np.pi/T
#Jvalue = 0.2/T     # finite interaction strong coupling
Jvalue = 0.072/T     # finite interaction weak coupling
#Jvalue = 0.0       # infinite interaction
betas = [0.0, 1.5, 2.5, float('inf')]
hpts = [jn_zeros(0,3)[0]]
labels = ['a','b', 'c', 'd', 'e', 'f', 'g', 'h']

lamb = 0.0 
lambd_x = lamb
lambd_y = lamb

nprocs = 8
p = Pool(processes = nprocs) 
opts = Options(nsteps=1e5, num_cpus=1, openmp_threads=1)

spinposition_A = [0,1,2,3]
spinposition_B = [4,5,6,7] 
datas = []

h0 = 0.0
ll = 0

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

        datas.append(data)
             
    fname = "localmz_N"+str(N)+"800T_J_"+str(Jvalue)+".hdf5"
    with h5py.File(fname, 'w') as hf:
        hf.create_dataset('mz', np.shape(datas), data=datas)
        hf.create_dataset('times', np.shape(times), data=times)
        hf.create_dataset('betas', np.shape(betas), data=betas)
        hf.attrs['N'] = N  
        hf.attrs['Jvalue'] = Jvalue
        hf.attrs['steps'] = steps
        hf.attrs['w'] = w
