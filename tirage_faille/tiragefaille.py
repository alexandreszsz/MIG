# ---------------------------------------------------------------------
# 1.Définition du Volume d'espace 
# ---------------------------------------------------------------------
import numpy as np

dx, dy, dz = 0.001 , 0.001 , 0.001 #paramètres infinitésimaux (précision spatiale entre deux cases d'un tableau)

n = 10000 # Nombre de colonne d'un tableau 

X , Y , Z = n*dx , n*dy, n*dz #Dimensions du volume

Volume = X*Y*Z #Calcul du volume occupé

# ---------------------------------------------------------------------
# 2.Tirage au sort du nombre de failles
# ---------------------------------------------------------------------

"""Ici on s'appuie sur la loi de Gutenberg-Richter qui donne la proportion
de séismes selon leur magnitude, et sur la loi liant magnitude d'un séisme
et longueur de la faille qui a cédé, pour en déduire la nombre(et donc la 
proportion de faille) selon leur longueur. """

alpha = 1.35 #paramètre d'évolution caractéristique de la loi de Gutenberg-Richter.
N0 =100 #paramètre caractéristique de la sismicité 


def N(L,alph = alpha,N_0 = N0):
    """Cette fonction calcule le nombre de faille à partir de leur longueur."""
    return int(N_0 * L**(-alph))

L_longueurs = [0.1,0.2,0.5,1,2,3,5,7,10,20] #Les tailles de failles possibles

N_longueurs = np.vectorize(N)(L_longueurs)#leurs nombres associés
# ---------------------------------------------------------------------
# 3.Tirage au sort de la position et de l'orientation des failles
# ---------------------------------------------------------------------
import random as rd
def tirage(L):
    n = 100000 # Nombre de colonne d'un tableau 
    """Cette fonction renvoie aléatoirement une position dans le cube de volume
    et une orientation sous la forme d'un triplet (x,y,z) et d'un vecteur normal et de la longueur."""
    x , y , z = rd.randint(0,n)*dx , rd.randint(0,n)*dy , rd.randint(0,n)*dz
    nx,ny,nz = (rd.random()*2-1),(rd.random()*2-1),(rd.random()*2-1)
    nx,ny,nz=nx/np.sqrt(nx**2+ny**2+nz**2),ny/np.sqrt(nx**2+ny**2+nz**2),nz/np.sqrt(nx**2+ny**2+nz**2)
    return [L,x,y,z,nx,ny,nz]

L_faille =[]
for i in range(len(L_longueurs)):
    for j in range(N_longueurs[i]):
        L_faille.append(tirage(L_longueurs[i]))


#Dans L_faille on a maintenant la liste des failles, avec leur position, leur orientation
#et leur longueur.


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

# ---------------------------------------------------------------------
# 4.plot des failles
# ---------------------------------------------------------------------

faults = np.array(L_faille)

# ---------------------------------------------------------------
# 1.fonctions
def orthonormal_basis(n):
    """Renvoie (u, v) orthonormaux et orthogonaux à n."""
    n = n / np.linalg.norm(n)
    # vecteur auxiliaire : axe le moins colinéaire à n
    aux = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(aux, n)) > 0.9:
        aux = np.array([0.0, 1.0, 0.0])
    u = np.cross(n, aux)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)          # déjà unitaire car u ⟂ n
    return u, v

def fault_square(L, pos, n):
    """Retourne les 4 coins du carré de côté L, normal à n, centré en pos."""
    pos, n = np.asarray(pos, float), np.asarray(n, float)
    u, v = orthonormal_basis(n)
    half = L / 2.0
    corners = np.array([
        pos + half*u + half*v,
        pos - half*u + half*v,
        pos - half*u - half*v,
        pos + half*u - half*v,
    ])
    return corners

def fault_mesh(L, pos, n, h=0.01):
    """Échantillonne le plan en grille pas h (discrétisation spatiale)."""
    pos, n = np.asarray(pos, float), np.asarray(n, float)
    u, v = orthonormal_basis(n)
    s = np.arange(-L/2, L/2 + h/2, h)          # respecte le pas h sur [−L/2, L/2]
    U, V = np.meshgrid(s, s)
    X = pos[0] + U*u[0] + V*v[0]
    Y = pos[1] + U*u[1] + V*v[1]
    Z = pos[2] + U*u[2] + V*v[2]
    return X, Y, Z

# ---------------------------------------------------------------
# 2. Figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

cmap = plt.colormaps['tab20']                    # 20 couleurs distinctes
all_pts = []

for i, f in enumerate(faults):
    color = cmap(i % 20)
    # --- rendu rapide : polygone ---
    corners = fault_square(L=f[0], pos=[f[1],f[2],f[3]], n=[f[4],f[5],f[6]])
    poly = plt.Polygon(corners[:, :2], closed=True)  # faux, remplacé ci-dessous
    # collection 3D :
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    pc = Poly3DCollection([corners], facecolors=color,
                          edgecolors='k', linewidths=0.5, alpha=0.6)
    ax.add_collection3d(pc)
    # --- version discrétisée (décommenter si besoin d'une grille de points) ---
    # X, Y, Z = fault_mesh(f['L'], f['pos'], f['n'], h=0.01)
    # ax.plot_wireframe(X, Y, Z, color=color, lw=0.1)
    all_pts.append(corners)

# bornes de la scène
pts = np.vstack(all_pts)
ax.set_xlim(pts[:,0].min(), pts[:,0].max())
ax.set_ylim(pts[:,1].min(), pts[:,1].max())
ax.set_zlim(pts[:,2].min(), pts[:,2].max())

ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
ax.set_box_aspect((1, 1, 1))   # échelle isotrope 
plt.tight_layout()
plt.show()

