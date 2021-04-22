import numpy as np
import matplotlib.pyplot as plt

class McoModifie:
    def __init__(self, cloud):
        self.cloud = cloud
        self.a0, self.a1 = self.fit()
		

        
    def fit(self,G): 
        #Moindre carrés avec perturbation
        N = len(self.cloud.X)
        A = np.zeros((N,2))
        b = np.zeros(2)
        b[0] = 0
        b[1] = 0
        for i in range(N):
            A[i,0] = 1/self.cloud.esp*G[i]**0.5
            A[i,1] = self.cloud.X[i]/self.cloud.eps[i]*G[i]**0.5
            b[0] += self.cloud.Y[i]/self.cloud.eps[i]**2*G[i]
            b[1] += self.cloud.Y[i]*self.cloud.X[i]/self.cloud.eps[i]**2*G[i]
        
        #Matrice des écarts-types (? à check)
        C = np.linalg.inv(np.dot(A.T,A))
        
        #Calcul des coefficients
        a = np.dot(C,b)
        
        return (a[0],a[1])
    	
    @property
    def coeffs(self):
		#Getter des coefficients de la regression
        self.a0, self.a1 = self.fit()
        return self.a0, self.a1
	
    def plot(self):
		#Fonction permettant de tracer le nuage ainsi que la droite de regression
        self.a0, self.a1 = self.fit()
        self.cloud.plot()
        plt.plot([self.cloud.x_min, self.cloud.x_max], [self.a0 + self.a1*self.cloud.x_min, self.a0 + self.a1*self.cloud.x_max], 'r--')
