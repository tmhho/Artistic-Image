import numpy as np
import matplotlib.pyplot as plt


def upSampleByTwo(xi):
    xiRes = np.zeros((2*xi.shape[0],xi.shape[1]))
    n = xi.shape[0]
    theta=np.array([0,.5])
    for i in range(n):
        xiRes[2*i:2*(i+1),:] = np.outer(1-theta,xi[i,:])+np.outer(theta,xi[(i+1)%n,:])
    return xiRes


resMax = 4
M = 16
alpha = 5e5
beta = 2e9

Mfinal = M*2**resMax

class Projector:
    def __init__(self, M, alpha1, alpha2, k):
        self.bounds=[alpha1*2**k,alpha2*2**(2*k),np.pi,np.pi]
        # Initialiser ici votre projecteur
        # A MODIFIER ICI
    def project(self, xi):
        # Projette xi sur l'ensemble R^(Kx2) tels que que si on note t=proj(xi)
        # pour chaque i ||A_1 t[i]||_2 <= self.bounds[0]
        # pour chaque i ||A_2 t[i]||_2 <= self.bounds[1]
        t = xi # A MODIFIER ICI
        return t



xi = np.zeros((M, 2))
xi[:,0] = np.linspace(0.2,0.5,M)*np.cos(np.linspace(0,2*np.pi,M,endpoint=False))
xi[:,1] = np.linspace(0.2,0.5,M)*np.sin(np.linspace(0,2*np.pi,M,endpoint=False))


print('M=',M)
print('Mfinal=',Mfinal)


Niter = 50


for k in range(resMax,-1,-1):
    if k<resMax:
        xi = upSampleByTwo(xi)
        M *= 2

    dt=2*np.pi/(alpha*Mfinal)
    alpha1 = alpha*dt
    alpha2 = beta*dt
    proj = Projector(M, alpha1, alpha2, k)

    for niter in range(Niter):

        # Calculer la valeur de votre fonction coût et son gradient ici
        # A MODIFIER ICI

        # Projeter sur l'ensemble des contraintes admissibles ici
        xi = proj.project(xi)

        plt.figure(1)
        plt.clf()
        plt.plot(xi[:,0],xi[:,1],linewidth=0.2,marker='o',markersize=0.5)
        plt.scatter(xi[:,0],xi[:,1],s=1,color='red')
        plt.axis("equal")
        plt.title("k=%i - nit=%i"%(k,niter))
        plt.pause(0.01)

plt.show()
