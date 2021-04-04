#%matplotlib inline
#%autosave 300
import numpy as np
import scipy as sp
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 14
from IPython.core.display import HTML
from IPython.display import display,Image
from matplotlib import animation
from sympy.physics.vector import init_vprinting
init_vprinting(use_latex='mathjax', pretty_print=False)

#Modélisation avec sympy
from sympy.physics.mechanics import dynamicsymbols, Point, ReferenceFrame

print("Définition des paramêtres (sous forme de symbol)")
l1, l2 = sp.symbols('l1 l2')
print("Parametres: ",l1,l2)
l1,l2

print("Définition des DDL, variables fonction du temps") 
theta1, theta2 = dynamicsymbols('theta1 theta2')
print("DDL: ",theta1,theta2)
theta1,theta2

#Repères et points -----------------------------------------------------

#repères Repère du robot, du point p et du Q
RO = ReferenceFrame('R_O')
RP = ReferenceFrame('R_P')
RQ = ReferenceFrame('R_Q')

RP.orient(RO, 'Axis', [theta1, RO.z])
RQ.orient(RP, 'Axis', [theta2, RO.z])

# Matrice de passage du repère de la base au repère de la pince 

T02 = RO.dcm(RQ)
T02.simplify()

#Points
O = Point('O')
P = Point('P')
Q = Point('Q')

P.set_pos(O, l1 * RP.x)
Q.set_pos(P, l2 * RQ.x)

# ----------------------------------------------------------------------

#position de Q dans le repère de base : calcul du MGD
Q.pos_from(O)

qxy = (Q.pos_from(O).express(RO)).simplify()
qxy



#composantes de Q suivant x et y dans le repère RO
qx=qxy.dot(RO.x)
qx

qy=qxy.dot(RO.y)
qy



#idem pour le point P
pxy = (P.pos_from(O).express(RO)).simplify()
pxy

#composante de P dans RO
px = pxy.dot(RO.x)
py = pxy.dot(RO.y)

# ----------------------------------------------------------------------



# ----------- Simulation géométrique -----------

# lambdify permet la conversion des formules analytiques en fonction numérique python.

Qx = sp.lambdify((l1, l2, theta1, theta2), qx, 'numpy')
Qy = sp.lambdify((l1, l2, theta1, theta2), qy, 'numpy')
Px = sp.lambdify((l1, theta1), px, 'numpy')
Py = sp.lambdify((l1, theta1), py, 'numpy')

#application numérique

# nbre de valeurs de theta
Np=100
# longueur du bras en m
L1 = 0.5
L2 = 0.5
# variation des angles
theta1s = np.linspace(0, np.pi/2,Np)
theta2s = np.linspace(-1.571, np.pi/2,Np)
# calcul de la positions fct angle
QX = np.array(Qx(L1, L2, theta1s, theta2s))
QY = np.array(Qy(L1, L2, theta1s, theta2s))
PX = np.array(Px(L1,theta1s))
PY = np.array(Py(L1,theta1s))

"""#tracer de position
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,6))
ax1.plot(np.rad2deg(theta1s), QX, label = r'$Q_x$')
ax1.plot(np.rad2deg(theta1s), QY, label = r'$Q_y$')
ax1.set_xlabel(r'($\theta_1$, $\theta_2$) [deg]')
ax1.set_ylabel(r' position [m]')
ax1.set_title('Position du point Q')
ax1.legend()
ax1.grid()
ax2.plot(QX,QY,label='Q')
ax2.plot(PX,PY,label='P')
ax2.set_title("Trajectoire")
ax2.legend();"""

#Animation de la trajectoire
Fig=plt.figure(figsize=(8,8))
ax = Fig.add_subplot(111, aspect='equal')
ax.set_axis_off()
ax.set_xlim((-1.2*(L1+L2),1.2*(L1+L2)))
ax.set_ylim((-1.2*(L1+L2),1.2*(L1+L2)))
ax.set_title('mouvement du bras de robot',fontsize=30)
#fig.set_facecolor("#ffffff")
line1, = ax.plot([0.,L1], [0.,0.], 'o-b', lw=18 , markersize=25)
line2, = ax.plot([L1,L1+L2], [0.,0.], 'o-', lw=18 , markersize=25)
pt1    = ax.scatter([L1+L2+0.05],[0.],marker="$\in$",s=800,c="black",zorder=3)
def animate(i):
    global PX,PY,QX,QY
    line1.set_data([0.,PX[i]],[0.,PY[i]])
    line2.set_data([PX[i],QX[i]],[PY[i],QY[i]])
    pt1.set_offsets([QX[i]+0.05,QY[i]])
    return line1,line2,pt1
anim = animation.FuncAnimation(Fig, animate, np.arange(1, Np), interval=50, blit=True)
plt.show()