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

class Robot_manipulator_2ddl : 

    def __init__(self):
        self.theta1s = 0
        self.theta2s = 0
        self.L1 = 0.5
        self.L2 = 0.5
        self.save = False
        self.name_file = "Animation_001"

    def set_theta(self,nv_th1,nv_th2) : 
        self.theta1s = nv_th1
        self.theta2s = nv_th2
    
    def get_theta(self): 
        return self.theta1s, self.theta2s

    def get_coord_coude(self) : 
        return [self.L1 * math.cos(self.theta1s), self.L1 * math.sin(self.theta1s)]
    
    def get_coord_pince(self):
        return [self.L1 * math.cos(self.theta1s)+
                self.L2 * math.cos(self.theta1s + self.theta2s),
                self.L1 * math.sin(self.theta1s)+
                self.L2 * math.sin(self.theta1s + self.theta2s)]

    def generate_liste_of_coord(self,th1,th2) : 
        liste1 = []
        liste2 = []
        for i in range(len(th1)) : 
            self.set_theta(th1[i],th2[i])
            liste1.append(self.get_coord_coude())
            liste2.append(self.get_coord_pince())
        return liste1, liste2

    """def draw_graph_grad(grad,time):

        data2 = {'Grad': [1920,1930,1940,1950,1960,1970,1980,1990,2000,2010],
         'Unemployment_Rate': [9.8,12,8,7.2,6.9,7,6.5,6.2,5.5,6.3]
        }
        df2 = DataFrame(data2,columns=['Year','Unemployment_Rate'])
        figure2 = plt.Figure(figsize=(5,4), dpi=100)
        ax2 = figure2.add_subplot(111)
        ax2.set_xlim((-1.2*(self.L1+self.L2),1.2*(self.L1+self.L2)))
        ax2.set_ylim((-1.2*(self.L1+self.L2),1.2*(self.L1+self.L2)))
        df2 = df2[['Year','Unemployment_Rate']].groupby('Year').sum()
        df2.plot(kind='line', legend=True, ax=ax2, color='r',marker='o', fontsize=10)
        ax2.set_title('Year Vs. Unemployment Rate') """
    
    def draw_Trajectory(self,th1,th2,target) : 
        fig, ax2 = plt.subplots(figsize=(7,7))
        positionQ, positionP = self.generate_liste_of_coord(th1,th2)
        positionP_x =[]
        positionP_y =[]
        positionQ_x =[]
        positionQ_y =[]
        for i in range(len(th1)): 
            positionP_x.append(positionP[i][0])
            positionP_y.append(positionP[i][1])
            positionQ_x.append(positionQ[i][0])
            positionQ_y.append(positionQ[i][1])
        ax2.plot(positionP_x,positionP_y,label='Pince')
        ax2.plot(positionQ_x,positionQ_y,label='Coude')
        ax2.scatter([target[0]],[target[1]],marker='+',s=800,c="red",label='cible')
        ax2.set_title("Trajectoire")
        ax2.legend();
        plt.show()

    def draw_grad_graph(self,grad0,grad1,t) : 
        fig, ax2 = plt.subplots(figsize=(12,8))
        """positionQ, positionP = self.generate_liste_of_coord(th1,th2)
        positionP_x =[]
        positionP_y =[]
        positionQ_x =[]
        positionQ_y =[]
        for i in range(len(th1)): 
            positionP_x.append(positionP[i][0])
            positionP_y.append(positionP[i][1])
            positionQ_x.append(positionQ[i][0])
            positionQ_y.append(positionQ[i][1])"""
        ax2.plot(t,grad0,label=' grad0 ')
        #ax2.plot(t,grad1,label=' grad1 ')
        ax2.set_title("Trajectoire")
        ax2.legend();
        plt.show()
    
        
    def draw_env(self,target) : 
        Fig=plt.figure(figsize=(7,7))
        ax = Fig.add_subplot(111, aspect='equal')
        ax.set_xlim((-1.2*(self.L1+self.L2),1.2*(self.L1+self.L2)))
        ax.set_ylim((-1.2*(self.L1+self.L2),1.2*(self.L1+self.L2)))
        ax.set_title('mouvement du bras de robot',fontsize=30)
        ax.scatter([target[0]],[target[1]],marker='+',s=800,c="red")
        return Fig,ax
        
    def draw_robot(self,fig,ax) :
        line1, = ax.plot([0.,self.L1], [0.,0.], 'o-b', lw=10 , markersize=25)
        line2, = ax.plot([self.L1,self.L1+self.L2], [0.,0.], 'o-', lw=10 , markersize=25)
        pt1    = ax.scatter([self.L1+self.L2],[0.],marker="$\in$",s=800,c="black",zorder=3)
        return line1,line2,pt1
        

    def train(self,th1,th2,line1,line2,pt1,Fig):
        
        P,Q = self.generate_liste_of_coord(th1,th2)
        
        def animate(i):
            line1.set_data([0.,P[i][0]],[0.,P[i][1]])
            line2.set_data([P[i][0],Q[i][0]],[P[i][1],Q[i][1]])
            pt1.set_offsets([Q[i][0],Q[i][1]])
            return line1,line2,pt1

        anim = animation.FuncAnimation(Fig, animate, np.arange(1, len(th1)), interval=50, blit=True,repeat = False)
        if self.save : 
            anim.save(self.name_file + '.gif',writer=PillowWriter(fps=30))
        plt.show()


        
