import numpy as np 
import scipy
from scipy import interpolate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import PIL
import math
import random

def importIMG(fig, N, show = False):
    img = PIL.Image.open(fig).convert('L')
    img = img.resize((N,N),PIL.Image.ANTIALIAS)
    img = np.array(img,dtype=float)
    img = np.rot90(np.rot90(np.rot90(img)))
    if show:
        PIL.Image.fromarray(img).show() 
    return img

def plot_points(samples,scale,col = 'black'):
    p = scale*np.array(samples)
    plt.scatter(p[:,0],p[:,1],s = 1, color=col)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim([0,scale])
    plt.ylim([0,scale])
    plt.axis('off')
    plt.show()
    plt.clf()

# square of |p1-p2|
def distance(p1,p2):
    p = [p1[0] - p2[0], p1[1] - p2[1]]
    return p[0]**2 + p[1]**2

# random a point p2 s.t. r < |p-p2| < 2r
def randPoint(p, r):
    l = random.uniform(r,2*r)
    a = random.random()*2*math.pi
    p2 = [l*math.cos(a)+p[0], l*math.sin(a)+p[1]]
    while p2[0] > 1 or p2[0] < 0 or p2[1] > 1 or p2[1] < 0:
        l = random.uniform(r,2*r)
        a = random.random()*2*math.pi
        p2 = [l*math.cos(a)+p[0], l*math.sin(a)+p[1]]
    return p2

def placeGrid(p, index, grid):
    N = len(grid)-4
    i = math.floor(p[0]*N)
    j = math.floor(p[1]*N)
    grid[i+2][j+2] = index

def neighbor_list(i,j,grid):
    neighbors = [grid[i][j+1],grid[i][j-1],grid[i+1][j+1],grid[i+1][j],\
    grid[i+1][j-1],grid[i-1][j+1],grid[i-1][j],grid[i-1][j-1],\
    grid[i-2][j-1],grid[i-2][j],grid[i-2][j+1],\
    grid[i+2][j-1],grid[i+2][j],grid[i+2][j+1],\
    grid[i-1][j+2],grid[i][j+2],grid[i+1][j+2],\
    grid[i-1][j-2],grid[i][j-2],grid[i+1][j-2]]
    return list(filter(lambda a: a != 0, neighbors))

def PoissonDisc(r,maxiter):
    r2 = r**2
    N = math.ceil(math.sqrt(2)/r) # here
    grid = np.zeros([N+4, N+4],dtype = int)
    for i in range(N+4):
        for j in range(N+4):
            grid[i][j] = 0

    samples=[[random.random(), random.random()]]
    active_list = [1]
    placeGrid(samples[0],1,grid)
    index = 2
    while active_list:
        it = random.randint(0,len(active_list)-1)
        p = samples[active_list[it]-1]
        notfound = True
        for iter in range(maxiter):
            p2 = randPoint(p,r)
            flag = True
            i = math.floor(p2[0]*N)+2
            j = math.floor(p2[1]*N)+2
            if grid[i][j] != 0:
                continue
            neighbors = neighbor_list(i,j,grid)
            for neighbor in neighbors:
                if distance(p2, samples[neighbor-1]) < r2:
                    flag = False
                    break
            if flag:
                notfound = False
                active_list.append(index)
                samples.append(p2)
                placeGrid(p2,index,grid)
                index += 1
        if (iter >= maxiter-1) and notfound:
            del active_list[it]
    return samples

def placeGrid2(p, pindex, grid):
    N = len(grid)
    i = math.floor(p[0]*N)
    j = math.floor(p[1]*N)
    grid[i][j].append(pindex)

def neighbor_list2(px,py,r,grid):
    N = len(grid)
    near = math.ceil(r*N) + 1
    neighbors = []
    for i in range(max(px-near,0),min(px+near,N)):
        for j in range(max(py-near,0),min(py+near,N)):
#             print([i,j])
            neighbors = neighbors + grid[i][j]
    return neighbors

# return the distance threshold for the point p
# def PoissonParam(p,rmin,rmax,pi):
#     return rmax

# def PoissonParam(p,rmin,rmax,pi):
#     delta = 0.15
#     gamma = 150
#     dist = p[0]**2 + p[1]**2
#     out = (dist + delta ) / gamma
#     return out

def PoissonParam(p,rmin,rmax,pi):
    N =len(pi)
    i = math.floor(p[0]*N)
    j = math.floor(p[1]*N)
    return rmin + pi[i][j] * (rmax - rmin) / 255.0

def PoissonDisc_tulleken(rmin,rmax,maxiter,pi):
    N = math.ceil(math.sqrt(2)/rmax)
    grid = []
    for i in range(N):
        grid.append([])
        for j in range(N):
            grid[i].append([])
    p = [0.5,0.5] # p = [random.random(), random.random()]
    r = PoissonParam(p,rmin,rmax,pi)
    samples=[p]
    lRadius = [r]
    active_list = [0]
    placeGrid2(samples[0],0,grid)
    index = 1
    while active_list:
        it = random.randint(0,len(active_list)-1)
        p = samples[active_list[it]]
        r = lRadius[active_list[it]]
        for iter in range(maxiter):
            p2 = randPoint(p,r)
            rp2 = PoissonParam(p2,rmin,rmax,pi)
            flag = True
            i = math.floor(p2[0]*N)
            j = math.floor(p2[1]*N)
            neighbors = neighbor_list2(i,j,rp2,grid)
            for neighbor in neighbors:
                if distance(p2, samples[neighbor]) <= rp2**2:
                    flag = False
                    break
            if flag:
                active_list.append(index)
                samples.append(p2)
                lRadius.append(rp2)
                placeGrid2(p2,index,grid)
                index += 1
        if (iter >= maxiter-1):
            del active_list[it]
    return samples

def placeGrid3(p, r, pindex, grid):
    N = len(grid)
    px = math.floor(p[0]*N)
    py = math.floor(p[1]*N)
    near = math.ceil(r*N) + 1
    for i in range(max(px-near,0),min(px+near,N)):
        for j in range(max(py-near,0),min(py+near,N)):
            grid[i][j].append(pindex)

def PoissonDisc_dwork(rmin,rmax,maxiter,pi):
    N = math.ceil(math.sqrt(2)/rmin)
    grid = []
    for i in range(N):
        grid.append([])
        for j in range(N):
            grid[i].append([])
    p = [0.5,0.5] # p = [random.random(), random.random()]
    r = PoissonParam(p,rmin,rmax,pi)
    samples=[p]
    lRadius = [r]
    active_list = [0]
    placeGrid3(samples[0],r,0,grid)
    index = 1
    while active_list:
        it = random.randint(0,len(active_list)-1)
        p = samples[active_list[it]]
        r = lRadius[active_list[it]]
        for iter in range(maxiter):
            p2 = randPoint(p,r)
            rp2 = PoissonParam(p2,rmin,rmax,pi)
            flag = True
            i = math.floor(p2[0]*N)
            j = math.floor(p2[1]*N)
            neighbors = grid[i][j]
            for neighbor in neighbors:
                if distance(p2, samples[neighbor]) <= rp2**2:
                    flag = False
                    break
            if flag:
                active_list.append(index)
                samples.append(p2)
                lRadius.append(rp2)
                placeGrid3(p2,r,index,grid)
                index += 1
        if (iter >= maxiter-1):
            del active_list[it]
    return samples