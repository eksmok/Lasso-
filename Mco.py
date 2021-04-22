import numpy as np
import matplotlib.pyplot as plt

class Mco:
    
	def __init__(self, cloud):
		self.cloud = cloud
		self.a0, self.a1 = self.fit()
		
	def fit(self):
		if self.cloud.is_empty():
			return 0, 0
		else:
			#On crée la matrice X en ajoutant une colonne de 1 à gauche pour prendre en compre la constante
			X = np.c_[np.ones(self.cloud.N), np.matrix(self.cloud.X).T]
			
			#Calcul de bêta chapeau B_hat = (X'*X)^(-1)*X'*Y 
			beta_hat = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)), X.T), self.cloud.Y)
			
			#On retourne les coefficients a0 et a1
			return beta_hat[0,0], beta_hat[0,1]
	
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
