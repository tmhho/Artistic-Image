import numpy as np 
import scipy
from scipy import interpolate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import PIL
import math

def importIMG(fig, N, show = False):
    img = PIL.Image.open(fig).convert('L')
    img = img.resize((N,N),PIL.Image.ANTIALIAS)
    img = np.array(img,dtype=float)
    img = np.max(img)-img
    img = np.rot90(np.rot90(np.rot90(img)))
    if show:
        PIL.Image.fromarray(img).show() 
    return img
    
def plot_curves(p,):
    plt.plot(p[:,0],p[:,1],markersize = 2)
    plt.scatter(p[:,0],p[:,1],color='red',s = 8)
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.xlim([-0.5,0.5])
    # plt.ylim([-0.5,0.5])
    plt.axis('off')
    # plt.title("it = %d"%iter)
    plt.show()
    plt.clf()

def plot_points(p,col = 'black'):
    plt.scatter(p[:,0],p[:,1],s = 5, color=col)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim([-0.5,0.5])
    plt.ylim([-0.5,0.5])
    plt.axis('off')
    # plt.savefig('video_img/plot_%03d.png'%iter)
    # plt.title("it = %d"%iter)
    plt.show()
    plt.clf()

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

def Stippling(pini,pi,xgrid,ygrid,itermax,learning_rate =0.1): 
    p = np.copy(pini)
    eps = 1e-10
    err = 2*eps
    iter = 0
    while err>eps and iter<itermax:
        if iter%50==0:
            plot_points(p,iter)
        iter += 1
        grad = gradJ(p, pi, xgrid,ygrid)
        p = p - learning_rate*grad
        err = np.linalg.norm(grad)
    return p

# def Wolfe_learning_rate(f, gradf, x, descent, pi,xgrid,ygrid, step, eps1 = 1e-4, eps2 = 0.99, itermax = 5):
#     eps1 = 1e-4
#     eps2 = 0.99
#     iter,s, sn , sp,d = 0,step,0., np.inf,descent
#     #     sn, sp are the minorant and majorant of learning rate 
#     f_x = f(x,pi,xgrid,ygrid)
#     grad_x = gradf(x,pi,xgrid,ygrid).ravel()
#     d_1col = d.ravel()
#     cond1 = f(x+s*d,pi,xgrid,ygrid) <= f_x + eps1*s*grad_x.T.dot(d_1col)
#     grad_xsd = gradf(x+s*d,pi,xgrid,ygrid).ravel()
#     cond2 = grad_xsd.T.dot(d_1col) >= eps2*grad_x.T.dot(d_1col)
#     while not cond1 or not cond2 :
#         if not cond1: 
#             sp = s
#             s = (sn + sp)/2.
#         else:
#             sn = s
#             if sp < np.inf:
#                 s = (sn + sp)/2.
#             else:
#                 s = 1.2*s 
#         cond1 = f(x+s*d,pi,xgrid,ygrid) <= f_x + eps1*s*grad_x.T.dot(d_1col)
#         grad_xsd = gradf(x+s*d,pi,xgrid,ygrid).ravel()
#         cond2 = grad_xsd.T.dot(d_1col) >= eps2*grad_x.T.dot(d_1col)
#         iter += 1
#         if iter > itermax:
#             break
#     return s

def Wolfe_learning_rate(f, gradf, x, descent, pi,xgrid,ygrid, step, eps1 = 1e-4, eps2 = 0.99, itermax = 10):
    eps1 = 1e-4
    eps2 = 0.99
    iter,s, sn , sp,d = 0,step,0., np.inf,descent
    #     sn, sp are the minorant and majorant of learning rate 
    f_x = f(x,pi,xgrid,ygrid)
    grad_x = gradf(x,pi,xgrid,ygrid).T.ravel()
    d_1col = d.T.ravel()
    cond1 = f(x+s*d,pi,xgrid,ygrid) <= f_x + eps1*s*grad_x.T.dot(d_1col)
    grad_xsd = gradf(x+s*d,pi,xgrid,ygrid).T.ravel()
    cond2 = grad_xsd.T.dot(d_1col) >= eps2*grad_x.T.dot(d_1col)
    while not cond1 or not cond2 :
        if not cond1: 
            sp = s
            s = (sn + sp)/2.
        else:
            sn = s
            if sp < np.inf:
                s = (sn + sp)/2.
            else:
                s = 1.2*s 
        cond1 = f(x+s*d,pi,xgrid,ygrid) <= f_x + eps1*s*grad_x.T.dot(d_1col)
        grad_xsd = gradf(x+s*d,pi,xgrid,ygrid).T.ravel()
        cond2 = grad_xsd.T.dot(d_1col) >= eps2*grad_x.T.dot(d_1col)
        iter += 1
        if iter > itermax:
            break
    return s

def GradDescent_wolfe(J,gradJ,pini,pi,xgrid,ygrid,itermax,step =0.1): 
    p = np.copy(pini)
    eps = 1e-10
    err = 2*eps
    iter = 0
    myplot(p,iter)
    num_points = p.shape[0]

    while err>eps and iter<itermax:
        t00 = time.process_time()
        grad = gradJ(p, pi, xgrid,ygrid)
        print(iter)
        print("learning_rate :", step)
        step =  Wolfe_learning_rate(J, gradJ, p, -gradJ(p,pi,xgrid,ygrid), pi,xgrid,ygrid, step)
        p = p - step*grad
        err = np.linalg.norm(grad)
        print("gradient: ", err)
        iter += 1
        if iter%10==0:
            # myplot(p,iter, True, "wolfe_%d"%num_points, step)
            myplot(p,iter)
        t = time.process_time() - t00
        file1 = open("GradDescent_%d_wolfe.txt"%num_points,"a")
        file1.write("Iter: %03d Erreur: %.7f Running_time %.5f\n"%(iter,err,t))
        file1.close()
    return p, iter 

# def BFGS(J, gradJ, pini, B0, pi, xgrid, ygrid, step = 2, eps = 1e-8, itermax = 100):
# #     H0 is random symetric definite positive matrix
#     B = B0
#     I = np.eye(B.shape[0])
#     iter = 0
#     p = pini
#     num_points = p.shape[0]
#     while iter < itermax : 
#         grad = gradJ(p, pi, xgrid,ygrid).ravel('A')
#         if np.linalg.norm(grad) < eps:
#             break 
#         print(iter)
#         d1col = -B.dot(grad)
#         d = d1col.reshape(2,num_points).T
#         # step =  Wolfe_learning_rate(J, gradJ, p, d, pi,xgrid,ygrid, step)
#         # print(step)
#         p = p + step*d
#         s = step*d1col 
#         y =  gradJ(p, pi, xgrid,ygrid).ravel('A') - grad 
#         M = I - (s.dot(y.T))/(y.T.dot(s))
#         B = M.T.dot(B).dot(M) + (s.dot(s.T))/(y.T.dot(s))
#         iter += 1
        
#         print(np.linalg.norm(grad))
#         myplot(p,iter)
#     return p
def BFGS(J, gradJ, pini, B0, pi, xgrid, ygrid, step = 10, eps = 1e-8, itermax = 100):
    B = B0
    iter = 0
    p = pini
    num_points = p.shape[0]
    I = np.eye(num_points*2)
    while iter < itermax : 
        print(iter)
        grad = gradJ(p, pi, xgrid,ygrid).T.ravel()
        if np.linalg.norm(grad) < eps:
            break 
        d1col = -np.linalg.solve(B,grad)
        d = d1col.reshape(2,num_points).T
        step =  Wolfe_learning_rate(J, gradJ, p, d, pi,xgrid,ygrid, step)
        p1 = p + step*d
        s = (p1 - p).T.ravel()
        y =  gradJ(p1, pi, xgrid,ygrid).T.ravel() - grad 
        M = I - (s.dot(y.T))/(y.T.dot(s))
        B = M.T.dot(B).dot(M) + (s.dot(s.T))/(y.T.dot(s))
        p = p1
        myplot(p,iter)
        print(np.linalg.norm(grad))
        iter += 1
    return p

# def BFGS(J, gradJ, pini, B0, pi, xgrid, ygrid, step = 10, eps = 1e-8, itermax = 100):
# #     H0 is random symetric definite positive matrix
#     B = B0
#     # H = H0
#     # H = np.linalg.inv(B)
#     iter = 0
#     p = pini
#     num_points = p.shape[0]
#     I = np.eye(num_points*2)
#     while iter < itermax : 
#         print(iter)
#         print("grad ",gradJ(p, pi, xgrid,ygrid) )
#         grad = gradJ(p, pi, xgrid,ygrid).T.ravel()
#         print("grad ravel ", grad)
#         if np.linalg.norm(grad) < eps:
#             break 
#         # d1col = -B.dot(grad)
#         # d1col = -np.linalg.solve(H,grad)
#         # d1col = -H.dot(grad)
#         d1col = -np.linalg.solve(B,grad)
#         # d1col = H.dot(grad)
#         # print("d11col ", d1col)
#         d = d1col.reshape(2,num_points).T
#         step =  Wolfe_learning_rate(J, gradJ, p, d, pi,xgrid,ygrid, step)
#         # print(step)
#         print("d ", d)
#         p1 = p + step*d
#         s = (p1 - p).T.ravel()
#         # print("s ", s)
#         y =  gradJ(p1, pi, xgrid,ygrid).T.ravel() - grad 
#         # print("y ",y)
#         # M = B.dot(s.dot(s.T)).dot(B)
#         # N = s.T.dot(B).dot(s)
#         # B = B + (y.dot(y.T))/(y.T.dot(s)) - M/N
#         M = I - (s.dot(y.T))/(y.T.dot(s))
#         B = M.T.dot(B).dot(M) + (s.dot(s.T))/(y.T.dot(s))
#         # S = s.dot(s.T)
#         # print('here', H.dot(S).dot(H))
#         # print(s.T.dot(H).dot(s))
#         # H = H + (y.dot(y.T))/(y.T.dot(s)) - (H.dot(S).dot(H))/s.T.dot(H).dot(s)
        
#         # H = np.linalg.inv(B)
#         p = p1
#         # print('H',H)
#         # print(np.linalg.norm(grad))
#         myplot(p,iter)
#         iter += 1
#     return p

# def ls_wolfe(step, f, gradf, x, descent, pi,xgrid,ygrid):
#     e1, e2 = 1e-4, 0.99
#     k, s, s_inf, s_sup, d = 0, step, 0, np.inf, descent
#     f_x, df_x = f(x,pi,xgrid,ygrid,eps=1e-7), gradf(x,pi,xgrid,ygrid,eps=1e-7).ravel()
#     f_xplus_sd, df_xplus_sd= f(x + s * d,pi,xgrid,ygrid,eps=1e-7), gradf(x + s * d,pi,xgrid,ygrid,eps=1e-7).ravel()
#     cd1_OK = f_xplus_sd <= f_x + e1 * s * df_x.T.dot(d.ravel())
#     cd2_OK = df_xplus_sd.T.dot(d.ravel()) >= e2 * df_x.T.dot(d.ravel())    
#     while not(cd1_OK and cd2_OK):
#         if not cd1_OK:
#             s_sup = s
#             s = 0.5 * (s_inf + s_sup)
#             f_x, df_x = f(x,pi,xgrid,ygrid,eps=1e-7), gradf(x,pi,xgrid,ygrid,eps=1e-7).ravel()
#             cd1_OK = f_xplus_sd <= f_x + e1 * s * df_x.T.dot(d.ravel())
#         else:
#             s_inf = s
#             s = 2 * s if (s_sup == np.inf) else 0.5 * (s_inf + s_sup)
#             f_xplus_sd, df_xplus_sd = f(x + s * d,pi,xgrid,ygrid,eps=1e-7), gradf(x + s * d,pi,xgrid,ygrid,eps=1e-7).ravel()
#             cd2_OK = df_xplus_sd.T.dot(d.ravel()) >= e2 * df_x.T.dot(d.ravel()) 
#         k +=1
#         print(cd1_OK)
#         print(cd2_OK)
#     return s

# def GradDescent_ls_wolfe(J,gradJ,pini,pi,xgrid,ygrid,itermax,step =0.1): 
#     p = np.copy(pini)
#     # y = [p]
#     eps = 1e-10
#     err = 2*eps
#     iter = 0
#     myplot(p,iter)
#     # plt.savefig('video_img/plot_%03d.png'%iter)

#     while err>eps and iter<itermax:
#         t00 = time.process_time()
#         grad = gradJ(p, pi, xgrid,ygrid)
#         t0 = time.process_time()
#         step = ls_wolfe(step, J, gradJ, p, -grad, pi,xgrid,ygrid)
#         print("learning_rate :", step)
#         # print("Wolfe_learning_rate calculation time: ", time.process_time() - t0)
#         p = p - step*grad
#         # y.append(p)
#         err = np.linalg.norm(grad)
#         print("gradient: ", err)
#         iter += 1
#         if iter%100==0:
#             myplot(p,iter)
#     # piter=np.array(y)
#         # print("GradDescent_wolfe one loop running time: ", time.process_time() -t00)
#     return p

# from scipy.optimize import line_search

# def GradDescent_linesearch_scipy(J,gradJ,pini,pi,xgrid,ygrid,itermax,step =0.1): 
#     p = np.copy(pini)
#     # y = [p]
#     eps = 1e-10
#     err = 2*eps
#     iter = 0
#     myplot(p,iter)
#     # plt.savefig('video_img/plot_%03d.png'%iter)
#     def func_J(p):
#         return J(p,pi,xgrid,ygrid)
#     def grad_J(p):
#         return gradJ(p,pi,xgrid,ygrid).ravel()
#     while err>eps and iter<itermax:
#         t00 = time.process_time()
#         grad = gradJ(p, pi, xgrid,ygrid)
#         t0 = time.process_time()
#         # learning_rate = Wolfe_learning_rate(4, oracle_J,p, -gradJ(p,pi,xgrid,ygrid),pi,xgrid,ygrid)
#         # step =  Wolfe_learning_rate(J, gradJ, p, -gradJ(p,pi,xgrid,ygrid), pi,xgrid,ygrid, step)
#         step = line_search(func_J, grad_J, p, -gradJ(p,pi,xgrid,ygrid).ravel())
#         print("learning_rate :", step)
#         # print("Wolfe_learning_rate calculation time: ", time.process_time() - t0)
#         p = p - step*grad
#         # y.append(p)
#         err = np.linalg.norm(grad)
#         print("gradient: ", err)
#         iter += 1
#         if iter%100==0:
#             myplot(p,iter)
#     # piter=np.array(y)
#         # print("GradDescent_wolfe one loop running time: ", time.process_time() -t00)
#     return p