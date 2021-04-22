# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from LinearCloud import LinearCloud
from Mco import Mco
from Mco_modifie import McoModifie

def main():
	confidence_area_Mco(1, 2, 0.1)

def confidence_area_Mco(a0, a1, sd, res=100):
    lc = LinearCloud(a0, a1, sd)
    mco = Mco(lc)
    mco_m = McoModifie(lc)
    
    G = np.random.exponential(1,lc.N)
    #print(Mco_m.moindre_carre_lineaire_modifie(G))
    
	#Taille de la fenêtre (Si a0 = 1 et delta_a0 = 0.6, la fenêtre ira de 0.7 à 1.3)
    delta_a0 = 0.6
    delta_a1 = 0.6
	
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
        lc.new_features(10)
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
	
    if __name__ == '__main__':
        main()
