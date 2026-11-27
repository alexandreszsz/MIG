"""Ce code sert à construire la fonction qui va coder la matrice de contrainte en fonction de l'espace et du temps."""


#-------------------------------------#
#Définitions des constantes
#-------------------------------------#

Q = 0.04 #Le débit volumique d'injection de fluide

lambdaa = 16E9 #coefficient de Lamé drainé

lambdaa_u = 37E9 #coefficient de Lamé non drainé

mu = 16E9 #Module de cisaillement

D = 0.1 #diffusivité hydraulique

alpha = 0.8 #coefficient de Biot

#-------------------------------------#
#Définitions de fonctions utiles
#-------------------------------------#
import numpy as np
import scipy as sp

def erfc(x):
    #Définition de la fonction erfc
    return (2/np.sqrt(np.pi))*sp.integrate.quad(lambda y : np.exp(-y**2),x,np.inf) 

def g(x):
    #Définition de la fonction g
    return (2/np.pi)*sp.integrate.quad(lambda y : np.exp(-y**2),0,x/2) - (x/np.pi)*np.exp(-x**2/4)

def delta(i,j):
    #Symbole de Kronecker
    if i==j:
        return 1
    else:
        return 0

def fun_sigma(r,t):
    """Calcul de la matrice de contrainte en tout point de l'espace-temps."""
    norme_r = np.sqrt(r[0]**2 + r[1]**2 + r[2]**2)
    xi = r/np.sqrt(D * t)
    sigma = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            #grosse formule bourrine de calcul de la matrice de contrainte
            sigma[i,j] = - (Q*mu*(lambdaa_u-lambdaa))/(4*np.pi*D*norme_r*alpha*(lambdaa_u + 2*mu)) * (delta(i,j)*(erfc(xi/2)-2*g(xi)/(xi**2)) + r[i]*r[j]/norme_r**2 *(erfc(xi/2)+6*g(xi)/(xi**2)))
    return sigma

def fun_sigma_normale(sigma,n):
    """Retourne la contrainte normale à partir d'une contrainte sigma et d'un vecteur normal n."""
    return np.dot(np.dot(sigma,n),n)