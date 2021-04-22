# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 10:52:59 2021

@author: Maëlys
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 08:04:32 2021

@author: Maëlys
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.linear_model import lasso_path

from SourceLasso import LassoRegression

class LassoModifie:
    
    NOMBRE_CV = 10
    
    def __init__(self, cloud_train, cloud_test, G):
        
        self.cloud_train = cloud_train
        self.cloud_test = cloud_train
        
        self.G = G
        
        #Calcul du modèle Lasso
        self.model = self.model_definition()
        
        #Calcul de l'hyperparamètre alpha à partir des données d'entrainement
        self.alpha = 0
        
        #Calcul des coefficients
        self.a0, self.a1 = self.fit()
        
    
    def model_definition(self) :
        
        #On prépare les données en les normalisant
        #Pour que lambda agisse de façon homogène sur les coefficients

        #D'abord, on crée un array avec nos données X et Y
        
        matrice_donnees = np.empty((0, 3))
        
        for i in range(len(self.cloud_train.X)) :
            nouvelle_ligne = np.array([[0,self.cloud_train.X[i],self.cloud_train.Y[i]]])
            
            matrice_donnees = np.append(matrice_donnees,nouvelle_ligne, axis=0)
        
    
        #Puis on normalise nos données : (x_train - moy_{X_train})/sigma_{X_train}
        
        donnees_normalisees_train = matrice_donnees
        
        #On fait en sorte d'avoir un array (nb_observations, nb_param (=1))
        
        X =  donnees_normalisees_train[:,0:2]
        #On récupère notre vecteur 1D des résultats y
        y = donnees_normalisees_train[:,2]
        
        model = LassoRegression( iterations = 1000, learning_rate = 0.01, l1_penality = 500, G = self.G )

        model.fit(X,y)
	

        #On renvoie le modèle
        return model
         
    def fit(self):
        if self.cloud_test.is_empty():
            return 0, 0
        else:         
            return (self.model.b,self.model.W[1])
    
    @property
    def coeffs(self):
        #Getter des coefficients de la regression
        self.model = self.model_definition()
        
        self.a0, self.a1 = self.fit()
        return self.a0, self.a1
    
    def plot(self):
        #Fonction permettant de tracer le nuage ainsi que la droite de regression
        self.model = self.model_definition()
        self.a0, self.a1 = self.fit()
        self.cloud_test.plot()
        plt.plot([self.cloud_test.x_min, self.cloud_test.x_max], [self.a0 + self.a1*self.cloud_test.x_min, self.a0 + self.a1*self.cloud_test.x_max], 'r--')


#Tests
        

from LinearCloud import LinearCloud

        
        
N = 300
#G = np.random.exponential(1,N)
G = abs(np.random.normal(1,1,N))

print(sum(G)/len(G))
a0 = 2
a1 = 7 #Ok de 12 à 22, pour N = 300 xD
sigma = 1
lc = LinearCloud(a0,a1,sigma,N=N)
lc_test = LinearCloud(a0,a1,sigma,N=150)

Res = LassoModifie(lc,lc_test, G)

for i in range(150) :
    
    G = abs(np.random.normal(1,1,N))

    Res.G = G
    
    Res.plot()
   
