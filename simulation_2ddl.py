#%matplotlib inline
#%autosave 300

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import PillowWriter
from matplotlib import style
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 10
from matplotlib.patches import Rectangle

class Robot_manipulator_2ddl : 

    def __init__(self):
        self.theta1s = 0
        self.theta2s = 0
        self.m1 = 1000  
        self.m2 = 1000
        self.m3 = 1000
        self
        self.L1 = 0.5
        self.L2 = 0.5

        # to save the animation as Gif file
        self.save = False
        self.name_file = "Animation_001"
        
        #Obstacle_sol
        self.x = -1.8
        self.y = 0
        self.w = 3.6
        self.h = -1.8        

    # Apply new values to self.theta1s and self.theta2s
    def set_theta(self,nv_th1,nv_th2) :
        self.theta1s = nv_th1
        self.theta2s = nv_th2

    # get self.theta1s and theta2s values.
    def get_theta(self): 
        return self.theta1s, self.theta2s

    # return the coordinates of the first joint point.
    def get_coord_coude(self) : 
        return [self.L1 * math.cos(self.theta1s), self.L1 * math.sin(self.theta1s)]
    
    # return the coordinates of the second joint point.
    def get_coord_pince(self):
        return [self.L1 * math.cos(self.theta1s)+
                self.L2 * math.cos(self.theta1s + self.theta2s),
                self.L1 * math.sin(self.theta1s)+
                self.L2 * math.sin(self.theta1s + self.theta2s)]

    # return a list of coordinates of a given joint point.
    def generate_liste_of_coord(self,th1,th2) : 
        liste1 = []
        liste2 = []
        for i in range(len(th1)) : 
            self.set_theta(th1[i],th2[i])
            liste1.append(self.get_coord_coude())
            liste2.append(self.get_coord_pince())
        return liste1, liste2

    def compute_weight_center(self): 
        return (1/(self.m1 + self.m2 + self.m3))*((self.m1+self.m3)*self.l1*math.cos(self.theta1s)+(self.m2+self.m3)*self.l2*math.cos(self.theta1s+self.theta2s))

    
    # draw the Trajectory taken by the robot to reach the goal
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
        ax2.set_title("Trajectory")
        ax2.legend();
        plt.show()

    # draw the variations of the gradient as funnction of time 
    def draw_grad_graph(self,grad0,grad1,t) : 
        fig, ax3 = plt.subplots(figsize=(11,7))
        ax3.plot(t,grad0,label=' \u2207J1')
        ax3.plot(t,grad1,label=' \u2207J2')
        ax3.grid(True)
        ax3.set_xlabel(r'Time (seconde)')
        ax3.set_ylabel(r' Gradient ')
        ax3.set_title("Gradient \u2207J1,\u2207J2")
        ax3.legend();
        plt.show()
        
    # draw the variations of the velocity as funnction of time 
    def draw_velocity_graph(self,velocity0,velocity1,t):
        fig, ax4 = plt.subplots(figsize=(11,7))
        ax4.plot(t,velocity0,label=" \u03C9 1 ")
        ax4.plot(t,velocity1,label=" \u03C9 2 ")
        ax4.set_title("Velocity \u03C9 1 , \u03C9 2 ")
        ax4.set_xlabel(r'Time (seconde)')
        ax4.set_ylabel(r' Velocity ')
        ax4.grid(True)
        ax4.legend();
        plt.show()

    # draw the variations of the gradient as function of time 
    def draw_error_graph(self,error_x,error_y,t):
        fig, ax5 = plt.subplots(figsize=(11,7))
        ax5.plot(t,error_x,label=' error_x')
        ax5.plot(t,error_y,label=' error_y')
        ax5.set_xlabel(r'Time (seconde)')
        ax5.set_ylabel(r' Error ')
        ax5.grid(True)
        ax5.set_title("error_x,error_y")
        ax5.legend();
        plt.show()


    
    # define and return the environnement parameters of the simulation : figure, axes, title ...  
    def draw_env(self,target) : 
        Fig=plt.figure(figsize=(8,8))
        style.use('fivethirtyeight')
        #plt.grid(False)
        ax = Fig.add_subplot(111, aspect='equal')
        ax.set_xlim((-1.2*(self.L1+self.L2),1.2*(self.L1+self.L2)))
        ax.set_ylim((-1.2*(self.L1+self.L2),1.2*(self.L1+self.L2)))
        ax.set_title('mouvement du bras de robot',fontsize=16)
        ax.scatter([target[0]],[target[1]],marker='+',s=400,c="red")
        return Fig,ax

    # define and return the robot parameters of the simulation : robot's arm, goal ...   
    def draw_robot(self,fig,ax) :
        line1, = ax.plot([0.,self.L1], [0.,0.], '#EC764A', lw=12 , marker='.', markersize=50, markerfacecolor='#EC764A', markeredgecolor='#444140')
        line2, = ax.plot([self.L1,self.L1+self.L2], [0.,0.], '#FF7F50', lw=12 , marker='.', markersize=30, markerfacecolor='#FF7F50', markeredgecolor='#444140')
        pt1    = ax.scatter([self.L1+self.L2],[0.],marker='3', s=1500,c="#444140",zorder=6)        
        return line1,line2,pt1
    
    def draw_obstacle(self,fig,ax):
        rectangle=ax.add_patch(Rectangle((self.x, self.y), self.w, self.h,color="#383433"))
        
    # initiate the animation 
    def train(self,th1,th2,line1,line2,pt1,Fig):
        
        P,Q = self.generate_liste_of_coord(th1,th2)
        
        def animate(i):
            line1.set_data([0.,P[i][0]],[0.,P[i][1]])
            line2.set_data([P[i][0],Q[i][0]],[P[i][1],Q[i][1]])
            pt1.set_offsets([Q[i][0],Q[i][1]])
            return line1,line2,pt1

        # Start animation
        anim = animation.FuncAnimation(Fig, animate, np.arange(1, len(th1)), interval=50, blit=True,repeat = False)
        if self.save : 
            anim.save(self.name_file + '.gif',writer=PillowWriter(fps=30))
        plt.show()


        
