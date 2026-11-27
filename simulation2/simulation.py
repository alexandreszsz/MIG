import numpy as np
import matplotlib.pyplot as plt


def solve(P,n,L,A,sens_v_p,v_p,d_c,sigma,rho,a,b,mu, eps=1E-10, N=1000):
    """Cette fonction prend en compte les caractéristiques d'une faille et renvoie la simulation pour N itération avec une erreur epsilon. """

    #P = np.array([0,0,1000]) coordonnées du centre de la faille. On a pris la convention où la profondeur est notée positivement.

    #n = np.array([0,0,1]) #Vecteur normal du plan de la faille sous la forme (x,y,z)

    #L 1.00 #Taille de la faille en m

    #A = 1.00 #Surface de la faille orthogonal à n en m²

    #sens_v_p le sens et la direction dans laquelle coulisse la faille à la vitesse v_p. (Précondition : il est de norme 1.)

    #v_p = 3.15E-5  #de l'ordre du cm/an, ici exprimé en m/s

    #d_c = 1E-6 #distance caractéristique du frottement

    #sigma = (0,0,1E6) #Contrainte tridimensionnelle

    #eta = 1E-8 #terme de viscosité

    #a , b = 0.5 , 0.1 #Coefficients caractéristiques de la loi de frottement

    #mu = 10E10 #module de cisaillement

    #eps erreur acceptable

    #N nombre d'itérations maximum dans le solveur

    #-------------------------------------#
    #Paramètres ajustés
    #-------------------------------------#

    sigma_normale=np.dot(sigma,n)
    
    k = mu/L

    #-------------------------------------#
    #Paramètres adimensionnnés
    #-------------------------------------#

    eta = np.sqrt(mu*rho)/2

    kappa = k*d_c/(b*sigma_normale)

    alpha = a/b

    #-------------------------------------#
    #Paramètres initiaux
    #-------------------------------------#

    v_i= 0.1*v_p #vitesse initiale


    #-------------------------------------#
    #Résolution
    #-------------------------------------#
    """
    Created on Wed Dec  9 14:23:37 2020

    @author: pierredublanchet
    """


    import numpy as np
    import matplotlib.pyplot as plt


    class NdParamMec:
        "Non dimensional Mechanical parameters"

    class ParamComp:
        "Computational parameters"
        
    #-------------------------------------#
    # ND Mechanical parameter definition
    #-------------------------------------#
    pnd=NdParamMec()
    pnd.a=alpha
    pnd.eta=eta
    pnd.k=kappa

    #-------------------------------------------#
    # Computational parameter definition
    #-------------------------------------------#
    pc=ParamComp()
    pc.tol=eps                        # error tolerance
    pc.nitrkmax=30                       # maximum number of iteration in a rkf step
    pc.nitmax=N                        # maximum number of iterations 
    pc.hmin=1.0E-12                      # minimum time step
    pc.hmax=1.0E10                       # maximum time step (CFL for diffusion equation)
    pc.safe=0.8   

    #-------------------------------------------#
    # Initial conditions (ND variables)
    #-------------------------------------------#
    v=v_i/v_p      # initial normalized slip rate
    th=1/v      # initial normalized state variable
    t=0.0       # initial time
    h=v*1E-3     # initial time step

    phi=np.log(v)
    nu=np.log(th)

    def frns(phi,nu,pnd):

        F=-pnd.k*(np.exp(phi)-1)+np.exp(phi)-np.exp(-nu)
        F=F/(pnd.a+pnd.eta*np.exp(phi))

        return F

    def grns(phi,nu):

        F=np.exp(-nu)-np.exp(phi)

        return F

    def rkf(phi,nu,h,pnd,pc):
        

        c21=1/4
        c31=3/32
        c32=9/32
        c41=1932/2197
        c42=-7200/2197
        c43=7296/2197
        c51=439/216
        c52=-8
        c53=3680/513
        c54=-845/4104
        c61=-8/27
        c62=2
        c63=-3544/2565
        c64=1859/4104
        c65=-11/40

        r1  =  1/360
        r3  = -128/4275
        r4  = -2197/75240
        r5  =  1/50
        r6  =  2/55

        c1  =  25/216
        c3  =  1408/2565
        c4  =  2197/4104
        c5  = -1/5
        c6 = 2/55

        passtep=0
        iterrk=0
        
        

        while passtep==0 and iterrk <= pc.nitrkmax:
            iterrk+=1


            # Compute values needed to compute truncation error estimate and
            # the 4th order RK estimate.
                
            
            k1 = h * frns(phi,nu,pnd)
            l1 = h * grns(phi,nu)


            dphi = c21*k1
            dnu = c21*l1
            
            k2 = h * frns(phi+dphi,nu+dnu,pnd)
            l2 = h * grns(phi+dphi,nu+dnu)


            dphi = c31*k1+c32*k2
            dnu = c31*l1+c32*l2
            
            k3 = h * frns(phi+dphi,nu+dnu,pnd)
            l3 = h * grns(phi+dphi,nu+dnu)


            dphi = c41*k1+c42*k2+c43*k3
            dnu = c41*l1+c42*l2+c43*l3
            
            k4 = h * frns(phi+dphi,nu+dnu,pnd)
            l4 = h * grns(phi+dphi,nu+dnu)



            dphi = c51*k1+c52*k2+c53*k3+c54*k4
            dnu = c51*l1+c52*l2+c53*l3+c54*l4

            k5 = h * frns(phi+dphi,nu+dnu,pnd)
            l5 = h * grns(phi+dphi,nu+dnu)


            dphi = c61*k1+c62*k2+c63*k3+c64*k4+c65*k5
            dnu = c61*l1+c62*l2+c63*l3+c64*l4+c65*l5
            
            k6 = h * frns(phi+dphi,nu+dnu,pnd)
            l6 = h * grns(phi+dphi,nu+dnu)


            
            # Error estimation
            err=r1*np.array([k1, l1])+r3*np.array([k3, l3])+r4*np.array([k4, l4])+r5*np.array([k5, l5])+r6*np.array([k6, l6])
            errmax=np.max(abs(err))

            if errmax <= pc.tol and errmax>=0:
                passtep=1
                
                phi=phi+c1 * k1 + c3 * k3 + c4 * k4 + c5 * k5 +c6 * k6
                nu=nu+c1 * l1 + c3 * l3 + c4 * l4 + c5 * l5 + c6*l6
                
                
                if errmax>0:
                    h=np.min(np.array([pc.safe*h*((pc.tol/errmax)**0.25), pc.hmax]))


            if errmax>pc.tol:
                h=np.max(np.array([pc.safe*h*((pc.tol/errmax)**0.25), pc.hmin]))

    

        return phi,nu,h

    #-------------------------------------------#
    # Iterations
    #-------------------------------------------#
    T=np.array([t])
    Phi=np.array([phi])
    Nu=np.array([nu])
    for iter in range(0,pc.nitmax,1):
        
        #--update phi, nu and h
        phi,nu,h = rkf(phi,nu,h,pnd,pc)
        
        #--update time
        t+=h

        #--store results
        T=np.append(T,[t])
        Phi=np.append(Phi,[phi])
        Nu=np.append(Nu,[nu])



    #-------------------------------------#
    #Redimensionnement du résultat
    #-------------------------------------#
    Temps = T*d_c/v_p

    Vitesse = v_p*np.exp(phi)

    Thêta = d_c/v_p*np.exp(nu)

    #-------------------------------------#
    #Replongement du résultat et traitement supplémentaire.
    #-------------------------------------#

    U = deplacement(Temps,Vitesse) #calcul du déplacement 

    U_plongé = np.array([sens_v_p[0]*U , sens_v_p[1]*U , sens_v_p[2]*U]) #calcul du déplacement relatif plongé dans l'espace 3d où évolue la faille

    V_plongé = np.array([sens_v_p[0]*Vitesse , sens_v_p[1]*Vitesse , sens_v_p[2]*Vitesse]) #calcul de la vitesse plongée dans l'espace 3d où évolue la faille

    Position = np.array([U_plongé[0] + P[0] , U_plongé[1] + P[1] , U_plongé[2] + P[2]])  #On replace ce déplacement pour obtenir la position de la faille


    return (Temps, V_plongé, Thêta, Position) #Renvoi du résultat



#-------------------------------------#
#Le but maintenant va être de traiter plus finement ce résultat pour le plongement en 3D, et des surpressions.
#-------------------------------------#


def deplacement(T,V):
    """Cette fonction va prendre en entrée le temps et la vitesse et en sortie donner le déplacement. """
    U=np.zeros(len(T))
    for i in range(1,len(T)):
        U[i]=V[i]*(T[i+1]-T[i])+U[i-1]
    return U

