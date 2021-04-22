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

class Lasso:
    
    #Le nombre de fold pour la validation croisée
    NOMBRE_CV = 10
    
    def __init__(self, cloud_train, cloud_test):
        
        self.cloud_train = cloud_train
        self.cloud_test = cloud_test
        
        #Calcul du modèle Lasso
        self.model = self.model()
        
        #Calcul de l'hyperparamètre alpha à partir des données d'entrainement
        self.alpha = self.model.alpha_
        
        #Calcul des coefficients
        self.a0, self.a1 = self.fit()
        
    
    def model(self) :
        if self.cloud_train.is_empty():
            return None
        else:
            
            #On prépare les données en les normalisant
            #Pour que lambda agisse de façon homogène sur les coefficients
    
            #D'abord, on crée un array avec nos données X et Y
            
            matrice_donnees = np.empty((0, 3))
            
            for i in range(len(self.cloud_train.X)) :
                
                nouvelle_ligne = np.array([[1,self.cloud_train.X[i],self.cloud_train.Y[i]]])
                
                matrice_donnees = np.append(matrice_donnees,nouvelle_ligne, axis=0)
            
        
            #Puis on normalise nos données : (x_train - moy_{X_train})/sigma_{X_train}
            sc = StandardScaler()
            donnees_normalisees_train = sc.fit_transform(matrice_donnees)
            
            donnees_normalisees_train = matrice_donnees
            
            #Lasso en validation croisée
            
            #D'abord on crée le modèle
            model = LassoCV(normalize=False,fit_intercept=False,random_state=0,cv= Lasso.NOMBRE_CV)
            
            #On fait en sorte d'avoir un array (nb_observations, nb_param (=1))
            
            X =  donnees_normalisees_train[:,0:2]
            #On récupère notre vecteur 1D des résultats y
            y = donnees_normalisees_train[:,2]
            
            #Puis on entraîne le modèle sur nos données d'entraînement
            model.fit(X,y)
            
            #On renvoie le modèle
            return model
         
    def fit(self):
        if self.cloud_test.is_empty():
            return 0, 0
        else:
            #On prépare les données en les normalisant
            #Pour que lambda agisse de façon homogène sur les coefficients
    
            #D'abord, on crée un array avec nos données X et Y
            
            matrice_donnees = np.empty((0, 3))
        
            for i in range(len(self.cloud_test.X)) :
            
                nouvelle_ligne = np.array([[1,self.cloud_test.X[i],self.cloud_test.Y[i]]])
                
                matrice_donnees = np.append(matrice_donnees,nouvelle_ligne, axis=0)
            
            
            #Puis on normalise nos données : (x_train - moy_{X_train})/sigma_{X_train}
            sc = StandardScaler()
            donnees_normalisees_test = sc.fit_transform(matrice_donnees)
            
            donnees_normalisees_test = matrice_donnees
            #On fait en sorte d'avoir un array (nb_observations, nb_param (=1))
            X = donnees_normalisees_test[:,0:2]
            #On récupère notre vecteur 1D des résultats y
            y = donnees_normalisees_test[:,2]
            
            
            #
            mon_alpha = [self.alpha]
            alpha_path, coeffs_lasso, _ = lasso_path(X,y,alphas=mon_alpha)
            
            return (coeffs_lasso[0,0], coeffs_lasso[1,0])
    
    @property
    def coeffs(self):
        #Getter des coefficients de la regression
        self.a0, self.a1 = self.fit()
        return self.a0, self.a1
    
    def plot(self):
        #Fonction permettant de tracer le nuage ainsi que la droite de regression
        self.a0, self.a1 = self.fit()
        self.cloud_test.plot()
        plt.plot([self.cloud_test.x_min, self.cloud_test.x_max], [self.a0 + self.a1*self.cloud_test.x_min, self.a0 + self.a1*self.cloud_test.x_max], 'r--')


#
##Tests
#        
#
#from LinearCloud import LinearCloud
#
#liste_pourcentages = []
#nombre_total_iterations = 100
#liste_a0 = [1,0.1,0.01,0.001,0.000000001]
#
#
##Avec un échantillon de training et de test, sigma = 1
##[0.07, 0.11, 0.10, 0.13, 0.10]
#
##Avec un seul échantillon, sigma = 1
##[0.26, 0.33, 0.26, 0.34, 0.34]
#
#
##Avec un échantillon de training et de test, sigma = 12
##[0.15, 0.21, 0.09, 0.16, 0.2]
#
##Avec un seul échantillon, sigma = 12
##[0.42, 0.39, 0.36, 0.42, 0.37]
#
#
#for a0 in liste_a0 :
#    nombre_nuls = 0
#    for iteration in range(nombre_total_iterations) :
#        a0 = 0.01
#        a1 = 6
#        sigma = 12
#        lc = LinearCloud(a0,a1,sigma,N=300)
#        lc_test = LinearCloud(a0,a1,sigma,N=150)
#        
#        Res = Lasso(lc,lc)
#        
#        if Res.a0 == 0 or Res.a1 == 0 :
#            nombre_nuls +=1
#    liste_pourcentages.append(nombre_nuls/nombre_total_iterations)
#
#Res.plot()


