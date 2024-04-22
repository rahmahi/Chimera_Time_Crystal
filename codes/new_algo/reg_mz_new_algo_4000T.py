from qutip import *
from scipy import signal
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.special import jn_zeros
from itertools import combinations
from multiprocessing import Pool
from tqdm import tqdm
import h5py

##--- Energy per spin for interaction range order---------
def j_ij(Jvalue, i,j, beta):
    return Jvalue/(np.abs(i-j))**beta

##-- Drive-------------------------------------------------

def drive1(t, args):
    w = args['omega']
    T = 2 * np.pi/w

    sqr1 = signal.square(2 * np.pi/T * t)
    if sqr1 == -1:
        sqr1= 0
    return sqr1

def drive2(t, args):
    w = args['omega']
    T = 2 * np.pi/w

    sqr1 = -signal.square(2 * np.pi/T * t)
    if sqr1 == -1:
        sqr1= 0
    return sqr1

def drive3(t, args):    # square cos wave transverse
    w = args['omega']
    h0 = args['h0']
    h = args['h']
    T = 2 * np.pi/w

    sint = -np.sin(w*t)
    if sint<=0:
        sint = 0
    sqrsin = h0 + h * sint
    return sqrsin

def get_hamiltonian(N,N1, lambd_x, lambd_y, Jvalue, beta, g, ea, eb):
    sx,sy,sz = sigmax(), sigmay(), sigmaz()
    empt = qzero(2**N) + 1j * qzero(2**N)
    H10, H11, H12, H21, H22, H23, H24 =  empt, empt, empt, empt, empt, empt, empt
    
    ##-- Hamiltonian ------------------------------------------- 
    ##-- First half

    for i in range(N1):  
        id = qeye(2**i)    
        dim11 = N-1-i
        id1 = qeye(2**dim11)
        H11 = H11 + Qobj(tensor(id,tensor(sx,id1)).full()) * g * (1-ea)

    for i in range(N1,N):  
        id = qeye(2**i)    
        dim12 = N-1-i
        id1 = qeye(2**dim12)
        H12 = H12 + Qobj(tensor(id,tensor(sx,id1)).full()) * g * (1-eb)   

    ##-- Second half

    comb = combinations(np.arange(N), 2)
    for nm in list(comb):
        i,j= np.array(nm)
        id = qeye(2**i)
        dim11 = j-i-1
        id1 = qeye(2**dim11)
        dim12 = N-1-j
        id2 = qeye(2**dim12)
        H21 = H21 + Qobj(tensor(id, tensor(sy, tensor(id1, tensor(sy,id2)))).full()) * j_ij(Jvalue, i,j, beta)
        
    for i in range(N):  
        id = qeye(2**i)    
        dim22 = N-1-i
        id1 = qeye(2**dim22)
        H22 = H22 + Qobj(tensor(id,tensor(sz,id1)).full()) 

    for i in range(N):  
        id = qeye(2**i)    
        dim22 = N-1-i
        id1 = qeye(2**dim22)
        H23 = H23 + Qobj(tensor(id,tensor(sx,id1)).full()) * lambd_x

    for i in range(N):  
        id = qeye(2**i)    
        dim22 = N-1-i
        id1 = qeye(2**dim22)
        H24 = H24 + Qobj(tensor(id,tensor(sy,id1)).full()) * lambd_y
    
    return H11, H12, H21, H22, H23, H24

##-- Dynamics
def run_dynm(args):
    N,N1,lambd_x,lambd_y,Jvalue=args['N'],args['N1'],args['lambd_x'],args['lambd_y'],args['Jvalue']
    beta,g,ea,eb,w = args['beta'],args['g'],args['ea'],args['eb'],args['omega']
    h0,h,times,opts,sz_s = args['h0'],args['h'],args['times'],args['opts'], args['sz_s']
    
    H11, H12, H21, H22, H23, H24 =  get_hamiltonian(N,N1, lambd_x, lambd_y, Jvalue, beta, g, ea, eb)
    
    params = args
    
    H = [[H11,drive1], [H12,drive1],[H21,drive2], [H22,drive3], [H23,drive2], [H24,drive2]]
    grket = basis(2**N,0)        
    out = mesolve(H, grket, times, [], sz_s, args = params)
    return out.expect, beta, Jvalue



freezing_pts = jn_zeros(0, 3)

N = 10
N1= int(N/2)
N2= N-N1
omega = 20.0
w = omega
T = 2 * np.pi/w
times = np.linspace(0, 5000 * T, 100000, endpoint=False)

ea, eb = 0.03, 0.9
lambd_y = 0
lambd_x = 0

ft = 10   
g = np.pi/T
Jvalues = [0.072/T, 0.2/T]   
Jlbl = [r'$J_0 = 0.072/T$',r'$J_0=0.2/T$'] 
betas = [0, 1.5, 2.5, float('inf')]

nprocs = 4
p = Pool(processes = nprocs) 
opts = Options(nsteps=1e5, num_cpus=1, openmp_threads=1)

     
spinposition_A = [0,1,2,3,4]
spinposition_B = [5,6,7,8,9]  
  
h0 = 0.0
h = freezing_pts[0] * w/4 
sx,sy,sz = sigmax(), sigmay(), sigmaz()

  
for jh,Jvalue in tqdm(enumerate(Jvalues)):             
        # Region A
        sz_os = []
        
        for m,i in enumerate(spinposition_A):
            id = qeye(2**i)    
            dim12 = N-1-i
            id1 = qeye(2**dim12)
            sz_os.append(tensor(id,tensor(sz,id1)).full())
        sz_oa = Qobj(np.sum(sz_os, axis=0))


        # Region B
        sz_os = []
        
        for m,i in enumerate(spinposition_B):
            id = qeye(2**i)    
            dim12 = N-1-i
            id1 = qeye(2**dim12)
            sz_os.append(tensor(id,tensor(sz,id1)).full())
        sz_ob = Qobj(np.sum(sz_os, axis=0))

        sz_s = [sz_oa, sz_ob]
        
        params = [{'h0':0, 'h':h, 'omega':omega, 'N':N,'N1':N1,\
                   'opts':opts, 'sz_s':sz_s, 'lambd_y':lambd_y,\
                   'lambd_x':lambd_x, 'Jvalue':Jvalue,'beta':beta,\
                   'g':g,'ea':ea,'eb':eb, 'times':times} for beta in betas]   
        
        data = p.map(run_dynm,params)

        for b in range(len(betas)):
            mz_data_a = data[b][0][0] * 2/N
            mz_data_b = data[b][0][1] * 2/N
            beta = data[b][1]
            Jval = data[b][2]

            fname = "apr22_RegMz_"+str(N)+"_j_"+str(Jval)+"_beta_"+str(betas[b])+".hdf5"
            with h5py.File(fname, 'w') as hf:
                hf.create_dataset('mza', np.shape(mz_data_a), data=mz_data_a)
                hf.create_dataset('mzb', np.shape(mz_data_b), data=mz_data_b)
                hf.create_dataset('times', np.shape(times), data=times)
                hf.attrs['N'] = N  
                hf.attrs['Jvalue'] = Jval
                hf.attrs['beta'] = betas[b]
                hf.attrs['w'] = w