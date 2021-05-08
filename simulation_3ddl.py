#%matplotlib inline
#%autosave 300
import numpy as np
import scipy as sp
import sympy as sp
import math
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 14
from IPython.core.display import HTML
from IPython.display import display,Image
from matplotlib import animation
from matplotlib.animation import PillowWriter
from sympy.physics.vector import init_vprinting
init_vprinting(use_latex='mathjax', pretty_print=False)

#Modélisation avec sympy
from sympy.physics.mechanics import dynamicsymbols, Point, ReferenceFrame

class Robot_manipulator_3ddl : 

    def __init__(self):
        self.theta1s = 0
        self.theta2s = 0
        self.theta3s = 0        
        self.L1 = 0.5
        self.L2 = 0.5
        self.L3 = 0.5   
        self.save = False 
        self.animation_name = "Animation_002"    

    def set_theta(self,nv_th1,nv_th2,nv_th3) : 
        self.theta1s = nv_th1
        self.theta2s = nv_th2
        self.theta3s = nv_th3
        
    
    def get_theta(self): 
        return self.theta1s, self.theta2s, self.theta3s

    def get_coord_coude(self) : 
        return [self.L1 * math.cos(self.theta1s), self.L1 * math.sin(self.theta1s)]
    
    def get_coord_moit(self) : 
        return [self.L1 * math.cos(self.theta1s)+
                self.L2 * math.cos(self.theta1s+self.theta2s),
                self.L1 * math.sin(self.theta1s)+
                self.L2 * math.sin(self.theta1s+self.theta2s)]    
    
    def get_coord_pince(self):
        return [self.L1 * math.cos(self.theta1s)+
                self.L2 * math.cos(self.theta1s + self.theta2s)+
                self.L3 * math.cos(self.theta1s + self.theta2s + self.theta3s),
                self.L1 * math.sin(self.theta1s)+
                self.L2 * math.sin(self.theta1s + self.theta2s)+
                self.L3 * math.sin(self.theta1s + self.theta2s + self.theta3s)]

    def generate_liste_of_coord(self,th1,th2,th3) : 
        liste1 = []
        liste2 = []
        liste3 = []        
        for i in range(len(th1)) : 
            self.set_theta(th1[i],th2[i],th3[i])
            liste1.append(self.get_coord_coude())
            liste2.append(self.get_coord_moit())            
            liste3.append(self.get_coord_pince())
        return liste1, liste2, liste3

    def draw_Trajectory(self,th1,th2,th3,target) : 
        fig, ax2 = plt.subplots(figsize=(7,7))
        positionQ, positionX, positionP = self.generate_liste_of_coord(th1,th2,th3)
        positionP_x =[]
        positionP_y =[]
        positionQ_x =[]
        positionQ_y =[]
        positionX_x =[]
        positionX_y =[]
        for i in range(len(th1)): 
            positionP_x.append(positionP[i][0])
            positionP_y.append(positionP[i][1])
            positionQ_x.append(positionQ[i][0])
            positionQ_y.append(positionQ[i][1])
            positionX_x.append(positionX[i][0])
            positionX_y.append(positionX[i][1])
        ax2.plot(positionP_x,positionP_y,label='Pince')
        ax2.plot(positionQ_x,positionQ_y,label='Coude1')
        ax2.plot(positionX_x,positionX_y,label='Coude2')
        ax2.scatter([target[0]],[target[1]],marker='+',s=800,c="red",label='cible')
        ax2.set_title("Trajectoire")
        ax2.legend();
        plt.show()
    
    
        
    def draw_env(self,target) : 
        Fig=plt.figure(figsize=(7,7))
        ax = Fig.add_subplot(111, aspect='equal')
        ax.set_xlim((-1.2*(self.L1+self.L2+self.L3),1.2*(self.L1+self.L2+self.L3)))
        ax.set_ylim((-1.2*(self.L1+self.L2+self.L3),1.2*(self.L1+self.L2+self.L3)))
        ax.set_title('mouvement du bras de robot',fontsize=30)
        ax.scatter([target[0]],[target[1]],marker='+',s=800,c="red")
        return Fig,ax
        
    def draw_robot(self,fig,ax) :
        line1, = ax.plot([0.,self.L1], [0.,0.], 'o-b', lw=10 , markersize=20)
        line2, = ax.plot([self.L1,self.L1+self.L2], [0.,0.], 'o-', lw=10 , markersize=20)
        line3, = ax.plot([self.L1+self.L2,self.L1+self.L2+self.L3], [0.,0.], 'o-', lw=10 , markersize=20)        
        pt1    = ax.scatter([self.L1+self.L2+self.L3],[0.],marker="$\in$",s=800,c="black",zorder=3)
        return line1,line2,line3,pt1
        

    def train(self,th1,th2,th3,line1,line2,line3,pt1,Fig):
        
        P,Q,X = self.generate_liste_of_coord(th1,th2,th3)
        
        def animate(i):
            line1.set_data([0.,P[i][0]],[0.,P[i][1]])
            line2.set_data([P[i][0],Q[i][0]],[P[i][1],Q[i][1]])
            line3.set_data([Q[i][0],X[i][0]],[Q[i][1],X[i][1]])            
            pt1.set_offsets([X[i][0],X[i][1]])
            return line1,line2,line3,pt1

        anim = animation.FuncAnimation(Fig, animate, np.arange(1, len(th1)), interval=50, blit=True,repeat = False)
        if self.save : 
            anim.save(self.animation_name + '.gif',writer=PillowWriter(fps=30))
        plt.show()

