# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 17:15:01 2021

@author: Maëlys
"""

from LinearCloud import LinearCloud

from Mco import Mco
from McoModifie import McoModifie

from Lasso import Lasso
from LassoModifie import LassoModifie



import time
import matplotlib.pyplot as plt
import numpy as np

class ProductionResultats() :
    
    def temps_calcul_mco(a0,a1,sd,liste_n) :
        T = []
        for N in liste_n :
            t_depart = time.clock()
            
            cloud = LinearCloud(a0, a1, sd, N=N)
            mco = Mco(cloud)
            
            t_fin = time.clock()
            
            T.append(t_fin - t_depart)
        
            print(str(round(N/liste_n[-1],2)) + " %")
        
        plt.plot(liste_n, T)
        
        #plt.plot([self.x_min, self.x_max], [self.a0 + self.a1*self.x_min, self.a0 + self.a1*self.x_max], 'b--')
        plt.xlabel("nombre d'observations N")
        plt.ylabel("temps de calcul des coefficients par la MCO")
        plt.grid()
        
        
    def temps_calcul_lasso(a0,a1,sd,liste_n) :
        T = []
        for N in liste_n :
            t_depart = time.clock()
            
            cloud_train = LinearCloud(a0, a1, sd, N=N)
            cloud_test = LinearCloud(a0, a1, sd, N=N)
            
            lasso = Lasso(cloud_train, cloud_test)
            
            t_fin = time.clock()
            
            T.append(t_fin - t_depart)
        
            print(str(round(N/liste_n[-1],2)) + " %")
        
        plt.plot(liste_n, T)
        
        #plt.plot([self.x_min, self.x_max], [self.a0 + self.a1*self.x_min, self.a0 + self.a1*self.x_max], 'b--')
        plt.xlabel("nombre d'observations N")
        plt.ylabel("temps de calcul des coefficients par la MCO")
        plt.grid()
        
    
    def temps_calcul_lasso_p(a0,a1,sd,liste_n) :
        T = []
        for N in liste_n :
            t_depart = time.clock()
            
            cloud_train = LinearCloud(a0, a1, sd, N=N)
            cloud_test = LinearCloud(a0, a1, sd, N=N)
            G = np.random.exponential(1,N)
            
            lasso = LassoModifie(cloud_train, cloud_test, G)
            
            t_fin = time.clock()
            
            T.append(t_fin - t_depart)
        
            print(str(round(N/liste_n[-1],2)) + " %")
        
        plt.plot(liste_n, T)
        
        #plt.plot([self.x_min, self.x_max], [self.a0 + self.a1*self.x_min, self.a0 + self.a1*self.x_max], 'b--')
        plt.xlabel("nombre d'observations N")
        plt.ylabel("temps de calcul des coefficients par la MCO")
        plt.grid()
        
        
    #Création des histogrammes pour les différentes méthodes
        
    def confidence_area_Mco(a0, a1, sd, res=100, nombre_points_lc = 100) :
        lc = LinearCloud(a0, a1, sd)
        mco = Mco(lc)

        
    	#Taille de la fenêtre (Si a0 = 1 et delta_a0 = 0.6, la fenêtre ira de 0.7 à 1.3)
        delta_a0 = 1
        delta_a1 = 1
    	
    	#Nombre de de fois où on applique la Mco (ici 10 fois par pixels)
        N = res**2 * 10
    	
    	#Matrice de l'histogramme
        T = np.zeros((res, res))
    	
        a0_min = a0-delta_a0/2
        a1_min = a1-delta_a1/2
        da0 = delta_a0/res
        da1 = delta_a1/res
    	
        for k in range(N):
    		#On affiche la progression
            print('{}\{}'.format(k,N))
    		
    		#On tire de nouvelles valeurs puis on fait une Mco pour déterminer a0_Mco et a1_Mco
            lc.new_features(nombre_points_lc)
            a0_mco, a1_mco = mco.coeffs
    		
    		#En fonction de la valeur de a0_Mco et a1_Mco, on peut ajuster l'histogramme (la matrice T)
            i = int((a0_mco-a0_min)/da0)
            j = int((a1_mco-a1_min)/da1)
            if (i>=0) and (i<res) and (j>=0) and (j<res):
                T[i,j] += 1
    	
    	#Affichage de l'histogramme
        plt.imshow(T, cmap=plt.cm.gray, extent=[a0 - delta_a0/2, a0 + delta_a0/2, a1 - delta_a1/2, a1 + delta_a1/2])
        plt.xlabel("a0")
        plt.ylabel("a1")
        plt.show()
        
    def confidence_area_mco_p(a0, a1, sd, res=100, nombre_points_lc = 100) : #A coder ?
        lc = LinearCloud(a0, a1, sd, N = nombre_points_lc)
        G = np.random.exponential(1,lc.N)
        
        lasso = Lasso(lc, G)

        
    	#Taille de la fenêtre (Si a0 = 1 et delta_a0 = 0.6, la fenêtre ira de 0.7 à 1.3)
        delta_a0 = 0.8
        delta_a1 = 0.8
    	
    	#Nombre de de fois où on applique la Mco (ici 10 fois par pixels)
        N = res**2 * 10
    	
    	#Matrice de l'histogramme
        T = np.zeros((res, res))
    	
        a0_min = a0-delta_a0/2
        a1_min = a1-delta_a1/2
        da0 = delta_a0/res
        da1 = delta_a1/res
    	
        for k in range(N):
    		#On affiche la progression
            print('{}\{}'.format(k,N))
    		
    		#On récupère d'autres coefficients G
            G = np.random.exponential(1,lc.N)
            
            a0_lasso, a1_lasso = lasso.coeffs
    		
    		#En fonction de la valeur de a0_Mco et a1_Mco, on peut ajuster l'histogramme (la matrice T)
            i = int((a0_lasso-a0_min)/da0)
            j = int((a1_lasso-a1_min)/da1)
            if (i>=0) and (i<res) and (j>=0) and (j<res):
                T[i,j] += 1
    	
    	#Affichage de l'histogramme
        plt.imshow(T, cmap=plt.cm.gray, extent=[a0 - delta_a0/2, a0 + delta_a0/2, a1 - delta_a1/2, a1 + delta_a1/2])
        plt.xlabel("a0")
        plt.ylabel("a1")
        plt.show()
        
        
    
    def confidence_area_lasso(a0, a1, sd, res=100, nombre_points_lc = 100) :
        lc_train = LinearCloud(a0, a1, sd, N = nombre_points_lc)
        lc_test = LinearCloud(a0, a1, sd, N= nombre_points_lc)
        
        lasso = Lasso(lc_train, lc_test)

        
    	#Taille de la fenêtre (Si a0 = 1 et delta_a0 = 0.6, la fenêtre ira de 0.7 à 1.3)
        delta_a0 = 0.8
        delta_a1 = 0.8
    	
    	#Nombre de de fois où on applique la Mco (ici 10 fois par pixels)
        N = res**2 * 10
    	
    	#Matrice de l'histogramme
        T = np.zeros((res, res))
    	
        a0_min = a0-delta_a0/2
        a1_min = a1-delta_a1/2
        da0 = delta_a0/res
        da1 = delta_a1/res
    	
        for k in range(N):
    		#On affiche la progression
            print('{}\{}'.format(k,N))
    		
    		#On tire de nouvelles valeurs puis on fait une Mco pour déterminer a0_Mco et a1_Mco
            lc_test.new_features(nombre_points_lc)
            lc_train.new_features(nombre_points_lc)
            
            a0_lasso, a1_lasso = lasso.coeffs
    		
    		#En fonction de la valeur de a0_Mco et a1_Mco, on peut ajuster l'histogramme (la matrice T)
            i = int((a0_lasso-a0_min)/da0)
            j = int((a1_lasso-a1_min)/da1)
            if (i>=0) and (i<res) and (j>=0) and (j<res):
                T[i,j] += 1
    	
    	#Affichage de l'histogramme
        plt.imshow(T, cmap=plt.cm.gray, extent=[a0 - delta_a0/2, a0 + delta_a0/2, a1 - delta_a1/2, a1 + delta_a1/2])
        plt.xlabel("a0")
        plt.ylabel("a1")
        plt.show()
        
        
    def confidence_area_lasso_p(a0, a1, sd, res=100, nombre_points_lc = 100) :
        lc_train = LinearCloud(a0, a1, sd, N = nombre_points_lc)
        lc_test = LinearCloud(a0, a1, sd, N= nombre_points_lc)
        G = np.random.exponential(1,lc_train.N)
        
        lasso_m = LassoModifie(lc_train, lc_test, G)

        print("On s'en sort ?")
        
    	#Taille de la fenêtre (Si a0 = 1 et delta_a0 = 0.6, la fenêtre ira de 0.7 à 1.3)
        delta_a0 = 10
        delta_a1 = 10
    	
    	#Nombre de de fois où on applique la Mco (ici 10 fois par pixels)
        N = res**2 * 1
    	
    	#Matrice de l'histogramme
        T = np.zeros((res, res))
    	
        a0_min = a0-delta_a0/2
        a1_min = a1-delta_a1/2
        da0 = delta_a0/res
        da1 = delta_a1/res
    	
        for k in range(N):
    		#On affiche la progression
            print('{}\{}'.format(k,N))
    		
    		#On récupère d'autres coefficients G
            G = np.random.exponential(1,lc_train.N)
            lasso_m.G = G
            
            a0_lasso, a1_lasso = lasso_m.coeffs
    		
            print(a0_lasso,a1_lasso)
            
    		#En fonction de la valeur de a0_Mco et a1_Mco, on peut ajuster l'histogramme (la matrice T)
            i = int((a0_lasso-a0_min)/da0)
            j = int((a1_lasso-a1_min)/da1) + 5
            print(i, j, res)
            if (i>=0) and (i<res) and (j>=0) and (j<res):
                T[i,j] += 1
    	
    	#Affichage de l'histogramme
        plt.imshow(T, cmap=plt.cm.gray, extent=[a0 - delta_a0/2, a0 + delta_a0/2, a1 - delta_a1/2, a1 + delta_a1/2])
        plt.xlabel("a0")
        plt.ylabel("a1")
        plt.show()
        
#test
        
#ProductionResultats.temps_calcul_mco(3,8,1,range(10,10000,50))
#ProductionResultats.temps_calcul_lasso(3,8,1,range(10,10000,500))
ProductionResultats.confidence_area_lasso_p(2, 7, 1, res = 50, nombre_points_lc = 50)


#mco_m = McoModifie(lc)
        
#G = np.random.exponential(1,lc.N)
#print(Mco_m.moindre_carre_lineaire_modifie(G))


#lc = LinearCloud(3, 8, 1, N = 10)
#mco = Mco(lc)
##lc.new_features(10)
#print(mco.coeffs)
