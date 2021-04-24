import numpy as np 
import scipy
from scipy import interpolate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import PIL
import math

def convol(a,b):
    return np.fft.ifftshift(np.fft.ifft2(np.fft.fft2(a)*np.fft.fft2(b)))

# H(x) = sqrt(x^2 + y^2 + e^2)

def H(x,eps=1e-7):
    return np.sqrt(x[0]**2 + x[1]**2 + eps**2)

def Hvect(X,Y,eps=1e-7): 
    return np.sqrt(X**2 + Y**2 +eps**2)

# gradH(X,Y) = [diff(H,x),diff(H,y)]    
# Remark: diff(H, x) = x / H(x)

def gradH(x,eps=1e-7):   
    return [x[0]/H(x,eps), x[1]/H(x,eps)]

def gradHvect(X,Y,eps=1e-7):
    return [X/Hvect(X,Y,eps), Y/Hvect(X,Y,eps)]

def F(p, eps=1e-7):
    N = len(p)  
    F = 0
    for i in range(N):
        for j in range(N):
            F += H(p[i]-p[j],eps)
    return F/(2*N**2)

# diff(F,xi) = sum_{j <> i} diff(H(pi-pj),xi) / N^2 = sum_{j <> i} (xi-xj)/H(pi-pj) / N^2
# Input: p = (p1,...,pN) Output: gradF at p : shape (N*2)
def gradF(p, eps=1e-7): 
    N = len(p)
    gradF_ = np.zeros([N,2])
    for i in range(N):
        s = 0
        for j in range(i):
            s = s + (p[i]-p[j])/H(p[i]-p[j],eps)
        for j in range(i+1,N):
            s = s + (p[i]-p[j])/H(p[i]-p[j],eps)
        gradF_[i] = s / N**2
    return gradF_

# G2, gradG2: functions used for testing
def G2(p, pi, xgrid, ygrid, eps = 1e-7):
    #     Input: (p1,...,pN) (matrix of size N*2) :
    #     Output: G at (p1,...,pN) is a real number 
    N = len(p)
    pi_ = pi/pi.sum()
    G_ = 0
    for i in range(N):
        X, Y = np.meshgrid(xgrid - p[i][0], ygrid - p[i][1])
        Hgrid = Hvect(X,Y,eps)
        Z = Hgrid * pi_
        G_ = G_ + Z.sum()
    return G_/N

def gradG2(p, pi, xgrid, ygrid, eps = 1e-7):
    N = len(p)
    pi_ = pi/pi.sum()
    gradG_ = np.zeros([N,2])
    for i in range(N):
        X, Y = np.meshgrid(xgrid - p[i][0], ygrid - p[i][1])
        gradH_ = gradHvect(X,Y,eps)
        Z = gradH_[0] * pi_
        gradG_[i][0] = - Z.sum()
        Z = gradH_[1] * pi_
        gradG_[i][1] = - Z.sum()
    return gradG_/N

def compute_integral(pi, xgrid, ygrid, eps = 1e-7): # grid.shape = (r,c,2)
    r,c = pi.shape
    k = 2
    r_padded = k*r + 1
    c_padded = k*c + 1
    pi_padded = np.zeros([r_padded,c_padded])
    padr = ((k-1)*r+1) // 2
    padc = ((k-1)*c+1) // 2
    pi_padded[padr:-padr][:,padc:-padc] = pi/pi.sum()
    
    Sx = xgrid[-1] + (xgrid[1]-xgrid[0])*padr
    Sy = ygrid[-1] + (ygrid[1]-ygrid[0])*padc
    
    x = np.linspace(-Sx,Sx,r_padded)
    y = np.linspace(-Sy,Sy,c_padded)
    x_padded, y_padded=  np.meshgrid(x, y)
    
    Hgrid= Hvect(x_padded,y_padded,eps)
    
    conv = convol(pi_padded, Hgrid)
    conv = conv[padr:-padr][:,padc:-padc]
    return conv.real

def interp(xgrid, ygrid, val): 
    # interpolate a bivariate function on with values vale on xgrid * ygrid
    # ygrid and xgrid need to be in this order
    return interpolate.RectBivariateSpline(ygrid, xgrid, val,kx=1, ky=1)

def G(p, pi, xgrid, ygrid, eps = 1e-7):
    #     Input: (p1,...,pN) (matrix of size N*2) :
    #     Output: G at (p1,...,pN) is a real number 
    N = len(p)
    conv = compute_integral(pi, xgrid, ygrid, eps) # value of the convolution of H and pi over grid
    conv_func = interp(xgrid, ygrid, conv) # interpolate the convolution of H and pi
    G_ = sum(conv_func(p[:,1],p[:,0], grid = False))
    return G_/N

# Now we need to compute the gradient of G with respect to the variables x1, y1,..., xN, yN. 
# Derivative of the convolution leads to compute the convolutions of diff(H, x) and diff(H,y) with pi

# this function below computes the values of two functions conv(diff(H,x),pi) and conv(diff(H,y),pi) over grid

def compute_integral_prime(pi, xgrid, ygrid, eps = 1e-7): # pi.shape = (r,c), grid.shape = (r,c,2)
    # Output: two matrices represent two grids of shape (r,c)
    # Each grid contains values of diff(H,x) and diff(H,y) with pi over grid = xgrid * ygrid
    r,c = pi.shape
    k = 2
    r_padded = k*r + 1
    c_padded = k*c + 1
    pi_padded = np.zeros([r_padded,c_padded])
    padr = ((k-1)*r+1) // 2
    padc = ((k-1)*c+1) // 2
    pi_padded[padr:-padr][:,padc:-padc] = pi/pi.sum()
    
    Sx = xgrid[-1] + (xgrid[1]-xgrid[0])*padr
    Sy = ygrid[-1] + (ygrid[1]-ygrid[0])*padc
    
    x = np.linspace(-Sx,Sx,r_padded)
    y = np.linspace(-Sy,Sy,c_padded)
    x_padded, y_padded=  np.meshgrid(x, y)
    
    gradH_ = gradHvect(x_padded,y_padded,eps) # gradHgrid.shape = (r_padded, c_padded, 2)
    
    conv1 = convol(pi_padded, gradH_[0]) # conv( diff(H, x) , pi)
    conv2 = convol(pi_padded, gradH_[1]) # conv( diff(H, y) , pi)
    
    # remove padding
    conv1 = conv1[padr:-padr][:,padc:-padc]
    conv2 = conv2[padr:-padr][:,padc:-padc]
    
    return conv1.real, conv2.real
    
def gradG(p, pi, xgrid, ygrid, eps = 1e-7):
#     Input: p = (p1,...,pN) (matrix of size N*2) 
#     Output: gradG(p) = [G1(p1), G2(p1), G1(p2), G2(p2),..., G1(pN), G2(pN)] (matrix of size N*2)
    conv1, conv2 = compute_integral_prime(pi, xgrid, ygrid, eps)

    G1 = interp(xgrid,ygrid,conv1)
    G2 = interp(xgrid,ygrid,conv2)
    
    N = len(p)
    grad = np.zeros([N,2])
    grad[:,1] = G1(p[:,0],p[:,1],grid=False)
    grad[:,0] = G2(p[:,0],p[:,1],grid=False)
    return grad/N

def J(p,pi,xgrid,ygrid,eps=1e-7):
    return - F(p) + G(p,pi,xgrid,ygrid,eps) 

def gradJ(p,pi,xgrid,ygrid,eps=1e-7):
    return - gradF(p) + gradG(p,pi,xgrid,ygrid,eps)

def myplot_curve(p,iter):
    plt.plot(p[:,0],p[:,1],markersize = 2)
    plt.scatter(p[:,0],p[:,1],color='red',s = 8)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim([-0.5,0.5])
    plt.ylim([-0.5,0.5])
    plt.axis('off')
    # plt.title("it = %d"%iter)
    plt.show()
    plt.clf()

def myplot(p,iter,col = 'black'):
    plt.scatter(p[:,0],p[:,1],s = 8, color=col)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim([-0.5,0.5])
    plt.ylim([-0.5,0.5])
    plt.axis('off')
    # plt.title("it = %d"%iter)
    plt.show()
    plt.clf()

def GradDescent(J,gradJ,pini,pi,xgrid,ygrid,itermax,learning_rate =0.1): 
    p = np.copy(pini)
    # y = [p]
    eps = 1e-10
    err = 2*eps
    iter = 0
    myplot(p,iter)
    # plt.savefig('video_img/plot_%03d.png'%iter)
    while err>eps and iter<itermax:
        grad = gradJ(p, pi, xgrid,ygrid)
        p = p - learning_rate*grad
        # y.append(p)
        err = np.linalg.norm(grad)
        iter += 1
        if iter%100==0:
            myplot(p,iter)
    # piter=np.array(y)
    return p

def importIMG(fig, N, show = False):
    img = PIL.Image.open(fig).convert('L')
    img = img.resize((N,N),PIL.Image.ANTIALIAS)
    img = np.array(img,dtype=float)
    img = np.max(img)-img
    img = np.rot90(np.rot90(np.rot90(img)))
    if show:
        PIL.Image.fromarray(img).show() 
    return img