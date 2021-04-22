import numpy as np
import matplotlib.pyplot as plt

class LinearCloud:
	def __init__(self, a0, a1, sd, N=0, x_min=0, x_max=1):
		#Coefficients de la droite
		self.a0 = a0
		self.a1 = a1
		
		#Standard deviation pour les eps_i
		self.sd = sd
		
		#Limites des abcisses des points du cloud linéaire
		self.x_min = x_min
		self.x_max = x_max
		self.N = N
		
		#Abssices et Ordonnées des points du cloud
		self.X = None
		self.Y = None
		self.eps = None
		if N:
			self.new_features(N)
	
	def new_features(self, N):
		#Cette fonction permet de construire de nouveaux points (et enlève les anciens points) pour constituer le cloud linéaire
		self.N = N
		self.X = np.random.uniform(self.x_min, self.x_max, N)
		self.eps = np.random.normal(0, self.sd, N)
		self.Y = self.a0 + self.a1*self.X + self.eps
	
	def is_empty(self):
		return self.X is None or len(self.X) == 0
	
	def plot(self):
		#Cette fonction permet de tracer le cloud de point
		plt.scatter(self.X, self.Y)
		#plt.plot([self.x_min, self.x_max], [self.a0 + self.a1*self.x_min, self.a0 + self.a1*self.x_max], 'b--')
		plt.xlabel("x")
		plt.ylabel("y")
		plt.grid()
