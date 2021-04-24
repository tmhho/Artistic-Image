from numpy import linalg as npl 
import scipy 
import numpy as np 

def H(x):
	return abs(x)

def Hp(x):
	return x/sqrt(x**2)

def convolution(pi, grid):
    n = pi.shape[0]
    n_padded = 2*n +1 # à compléter
    pi_padded = np.zeros(n_padded)
    pad = (n+1) // 2 # à compléter
    pi_padded[pad:-pad] = pi
    S = grid[-1] + (grid[1]-grid[0])*(n_padded-n)/2.
    grid_padded = np.linspace(-S, S, n_padded)
    Hgrid = H(grid_padded)
    conv = convol(pi_padded, Hgrid)
    conv = conv[pad:-pad]
    return conv.real

def convolution_prime(pi, grid):
    n = pi.shape[0]
    n_padded = 2*n +1
    pi_padded = np.zeros(n_padded)
    pad = (n+1) // 2 
    pi_padded[pad:-pad] = pi
    S = grid[-1] + (grid[1]-grid[0])*(n_padded-n)/2.
    grid_padded = np.linspace(-S, S, n_padded)
    Hgrid = Hp(grid_padded)
    conv = convol(pi_padded, Hgrid)
    conv = conv[pad:-pad]
    return conv.real

def interp(grid, ky):
	return scipy.interpolate.interp1d(grid,ky)

def F(p):
	N = len(p) 
	i,j = 0 
	H = npl.norm(p[0])
	i,j = 0 
	while i < N: 
		while j < N: 
			H += npl.norm(p[i]-p[j])
			j += 1
		i += 1
	return 1/(2*n**2)*H 


def gradF(p): 
	N = len(p)
	grad = np.zeros(N)
	for i in range(N):
		grad[i] = np.sum(Hp(p[i]-p))
	return grad/(N**2)

def G(p,pi,grid): 
	ky = convolution(pi, grid)
	f = interp(grid,ky)
	return np.sum(f(p))/len(p)

def gradG(p,pi,grid):
	ky = convolution_prime(pi, grid)
	f = interp(grid,ky)
	return np.sum(f(p))/len(p)	

def J(p):
	return F(p) - G(p)

def gradJ(p):
	return gradF(p) - gradG(p)

def Gradient(J, gradJ,h=1e-1,xini=np.array([0,0])): 
	x = np.copy(xini) 
	y = [x] 
	eps = 1e-10 
	itermax = 1000 
	err = 2*eps 
	iter = 0 
	while err>eps and iter<itermax:
		x = x - h*gradJ(p)
		y.append(x) 
		err = np.linalg.norm(gradJ) 
		iter += 1 
	xiter=np.array(y) 
	return x,xiter,iter

N = 67 
pi = np.array([1/N]*N) 
grid = (np.arange(n)+0.5)/N - 0.5



