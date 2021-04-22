# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 18:08:20 2021

@author: Maëlys
"""

Kate
VisualStudio

import numpy

from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import Axes3D

import random
import numpy.linalg
def f0(x):

                

    return 1

def f1(x):
    return x

def generation(a0,a1,f0,f1,N,s):
    x=numpy.linspace(0,1,N)
    sigma=numpy.ones(N)*s
    y=numpy.zeros(N)
    for i in range(N):
        yi=a0*f0(x[i])+a1*f1(x[i])
        y[i] = random.gauss(yi,sigma[i])
    return (x,y,sigma)

N=10
a0=1
a1=2
s=0.1
(x,y,sigma)=generation(a0,a1,f0,f1,N,s)
#
#figure()
#plot(x,y,"ko")
#xlabel("x")
#ylabel("y")
#grid()


def moindre_carre_lineaire(f0,f1,x,y,sigma):
    N=len(x)
    A=numpy.zeros((N,2))
    b=numpy.zeros(2)
    b[0] = 0
    b[1] = 0
    for i in range(N):
        A[i,0] = f0(x[i])/sigma[i]
        A[i,1] = f1(x[i])/sigma[i]
        b[0] += y[i]*f0(x[i])/sigma[i]**2
        b[1] += y[i]*f1(x[i])/sigma[i]**2
    C=numpy.linalg.inv(numpy.dot(A.T,A))
    a = numpy.dot(C,b)
    return(a,C)
    
(a,C)=moindre_carre_lineaire(f0,f1,x,y,sigma)
                     
print(a)
print(C)

list_x=numpy.linspace(0,1,500)
plot(list_x,a[0]*f0(list_x)+a[1]*f1(list_x),"k-")
xlabel("x")
ylabel("y")

def hist_coefficients(a0,a1,delta_a0,delta_a1,M,Nt,f0,f1,s):
    h=numpy.zeros((M,M))
    a0_min=a0-delta_a0/2
    a1_min=a1-delta_a1/2
    da0 = delta_a0/M
    da1 = delta_a1/M
    for t in range(Nt):
        (x,y,sigma) = generation(a0,a1,f0,f1,N,s)
        (a,C)=moindre_carre_lineaire(f0,f1,x,y,sigma)
        i = int((a[0]-a0_min)/da0)
        j = int((a[1]-a1_min)/da1)
        if (i>=0) and (i<M) and (j>=0) and (j<M):
            h[j,i] += 1
    list_a0 = numpy.linspace(a0_min,a0_min+delta_a0,M)
    list_a1 = numpy.linspace(a1_min,a1_min+delta_a1,M)
    return (list_a0,list_a1,h)

delta_a0 = 0.5
delta_a1 = 0.5
M=30
Nt = M**2*10
(list_a0,list_a1,h) = hist_coefficients(a0,a1,delta_a0,delta_a1,M,Nt,f0,f1,s)

imshow(h,cmap=cm.gray,extent=[a0-delta_a0/2,a0+delta_a0/2,a1-delta_a1/2,a1+delta_a1/2])
xlabel("a0")
ylabel("a1")


def moindre_carre_lineaire_modifie(f0,f1,x,y,sigma,G) : 
    #Moindre carrés avec perturbation
    N=len(x)
    A=numpy.zeros((N,2))
    b=numpy.zeros(2)
    b[0] = 0
    b[1] = 0
    for i in range(N):
        A[i,0] = f0(x[i])/sigma[i]*G[i]**0.5
        A[i,1] = f1(x[i])/sigma[i]*G[i]**0.5
        b[0] += y[i]*f0(x[i])/sigma[i]**2*G[i]
        b[1] += y[i]*f1(x[i])/sigma[i]**2*G[i]
    C=numpy.linalg.inv(numpy.dot(A.T,A))
    a = numpy.dot(C,b)
    return (a,C)


def mean(L) :
    return(sum(L)/len(L))
    
M_pour_Gi = 1000000

ens_I = []
ens_J = []

h=numpy.zeros((M,M))

#(x,y,sigma) = generation(a0,a1,f0,f1,N,s)

for k in range(M_pour_Gi) :
    
    G = numpy.random.exponential(1,N)
    
    #Histogramme
    #def histogramme2(a0,a1,delta_a0,delta_a1,M,Nt,f0,f1,s) :
    #Crée une matrice vide
    a0_min=a0-delta_a0/2 
    a1_min=a1-delta_a1/2
    da0 = delta_a0/M
    da1 = delta_a1/M
    #for t in range(Nt):
    
    (a,C)=moindre_carre_lineaire_modifie(f0,f1,x,y,sigma, G)
    i = int((a[0]-a0_min)/da0)
    j = int((a[1]-a1_min)/da1)
    #print(i,j)
    
    ens_I.append(i)
    ens_J.append(j)
    
    if (i>=0) and (i<M) and (j>=0) and (j<M):
        h[j,i] = h[j,i] + 1
        
moy_I = mean(ens_I)
moy_J = mean(ens_J)

for k in range(len(ens_I)) :
    ens_I[k] = ens_I[k] + int(moy_I)
    ens_J[k] = ens_J[k] + int(moy_J)
    i = ens_I[k]
    j = ens_J[k]
    if (i>=0) and (i<M) and (j>=0) and (j<M):
        h[j,i] = h[j,i] + 1

list_a0 = numpy.linspace(a0_min,a0_min+delta_a0,M)
list_a1 = numpy.linspace(a1_min,a1_min+delta_a1,M)

    
imshow(h,cmap=cm.gray,extent=[a0-delta_a0/2,a0+delta_a0/2,a1-delta_a1/2,a1+delta_a1/2])
xlabel("a0")
ylabel("a1")


    
print("Affichage")
print(mean(ens_I))
print(mean(ens_J))
print(h)