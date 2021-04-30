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

class Robot_manipulator : 

    def __init__(self,th1,th2):
        self.theta1s = th1
        self.theta2s = th2
        self.L1 = 0.5
        self.L2 = 0.5


    def set_theta(self,w1,w2) : 
        self.theta1s = w1*60
        self.theta2s = w2*60
    
    def get_theta(self): 
        return self.theta1s, self.theta2s


    #Repères et points
    def set_paramater(self,Theta1s,Theta2s):

        print("Définition des paramêtres (sous forme de symbol)")
        l1, l2 = sp.symbols('l1 l2')
        print("Parametres: ",l1,l2)
        l1,l2

        print("Définition des DDL, variables fonction du temps") 
        theta1, theta2 = dynamicsymbols('theta1 theta2')
        print("DDL: ",theta1,theta2)
        theta1,theta2

        #repères
        RO = ReferenceFrame('R_O')
        RP = ReferenceFrame('R_P')
        RQ = ReferenceFrame('R_Q')

        RP.orient(RO, 'Axis', [theta1, RO.z])
        RQ.orient(RP, 'Axis', [theta2, RO.z])

        T02 = RO.dcm(RQ)
        T02.simplify()

        #Points
        O = Point('O')
        P = Point('P')
        Q = Point('Q')

        P.set_pos(O, l1 * RP.x)
        Q.set_pos(P, l2 * RQ.x)

        #position de Q dans le repère de base
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

        #composante de Q dans RO
        px = pxy.dot(RO.x)
        py = pxy.dot(RO.y)

        ######Simulation géométrique
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
        theta1s = Theta1s
        theta2s = Theta2s
        # calcul de la positions fct angle
        QX = np.array(Qx(L1, L2, theta1s, theta2s))
        QY = np.array(Qy(L1, L2, theta1s, theta2s))
        PX = np.array(Px(L1,theta1s))
        PY = np.array(Py(L1,theta1s))
        return PX,PY,QX,QY


    def get_coord_pince(self) : 
        P1,P2,P3,P4 = self.set_paramater(self.theta1s,self.theta2s)
        return P3,P4

    #Animation de la trajectoire
    def env_animation(self,L1,L2) : 

        #fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,6))
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
        return Fig,line1,line2,pt1


    def train(self): 
        #Np =100
        
        x1,x2,x3,x4 = self.env_animation(self.L1,self.L2)
        

        def animate(i):
            global PX,PY,QX,QY
            #THeta1s = np.linspace(0, np.pi,100)
            #THeta2s = np.linspace(0, np.pi/2,100)
            PX,PY,QX,QY = self.set_paramater(self.theta1s,self.theta2s)
            x2.set_data([0.,PX[i]],[0.,PY[i]])
            x3.set_data([PX[i],QX[i]],[PY[i],QY[i]])
            x4.set_offsets([QX[i]+0.05,QY[i]])
            return x2,x3,x4

        anim = animation.FuncAnimation(x1, animate, np.arange(1, 100), interval=50, blit=True)
        plt.show()

#"""th1 = np.linspace(0, np.pi,100)   # valeur rad
#th2 = np.linspace(0, np.pi/2,100)
#longg = 0.5
#bra = Robot_manipulator(th1,th2,longg)
#bra.train()"""