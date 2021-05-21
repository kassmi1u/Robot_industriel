import time
import math

"""def theta_s(x,y):
    if x>0:
        return 1*math.atan(1*y)
    if x<=0:
        return 1*math.atan(-1*y)"""

class OnlineTrainer_3ddl:
    
    def __init__(self,robot,NN):

        """
        Args:
            robot (Robot): a robot instance following the pattern of
                Simulation
            target (list): the target position [x,y,theta]
        """
        self.robot = robot
        self.network = NN
        self.alpha = [1,1]  # normalition avec limite du monde cartesien = -3m � + 3m
        self.training = False
        self.running = True
        self.pas = 0.5
        self.moment = 0
        self.grad0 = []
        self.grad1 = []
        self.grad2 = []
        self.velocity0 = []
        self.velocity1 = []
        self.error_x = []
        self.error_y = []
        self.t = []
        self.t2 = []

    def train(self, targett):
        
        position = self.robot.get_coord_pince()
        alpha_1 = self.alpha[0]
        alpha_2 = self.alpha[1]
        th1 = []
        th2 = []
        th3 = []        
        target = [float(targett[0].get()),float(targett[1].get())]
        print(target)
        """position_x = []
        position_y = []
        position_x.append(position[0])
        position_x.append(0)
        position_y.append(position[1])
        position_y.append(1) """
        
        #network_input[0] = 
        #network_input[1] = 
        robot_a_bouge = time.time()
        i=0
        temp = 0
        temp1 = 0
        temp3 = 0
        temp4 = 0
        while abs(position[0]-target[0]) > 0.001 or  abs(position[1]-target[1]) > 0.001 : 
            debut = time.time()
            network_input = [(position[0]-target[0])*self.alpha[0], (position[1]-target[1])*self.alpha[1]]
            command = self.network.runNN(network_input) # propage erreur et calcul vitesses roues instant t  # Fonction à changer

            # useful to draw velocity graph 
            temp4 = temp4 + temp3
            temp3 = time.time() - robot_a_bouge
            self.t2.append(temp3+temp4)
            self.velocity0.append(command[0])
            self.velocity1.append(command[1])
            self.error_x.append(position[0]-target[0])
            self.error_y.append(position[1]-target[1])

            crit_av= alpha_1*(position[0]-target[0])*(position[0]-target[0]) + alpha_2*(position[1]-target[1])*(position[1]-target[1]) 

            #alpha_x = 1/6 #"""(max(position_x) - min(position_x))"""
            #alpha_y = 1/6 #"""(max(position_y) - min(position_y))"""   
            
                                 
            diff = time.time() - robot_a_bouge
            thetas = self.robot.get_theta()
            theta_temp1 = thetas[0] + command[0] * diff
            theta_temp2 = thetas[1] + command[1] * diff
            theta_temp3 = thetas[2] + command[1] * diff            
            self.robot.set_theta(theta_temp1,theta_temp2,theta_temp3)    
            robot_a_bouge = time.time()  
            th1.append(theta_temp1)
            th2.append(theta_temp2)
            th3.append(theta_temp3)            
            #self.robot.train(th1,th2,th3)
            # applique vitesses roues instant t,       # Fonction à changer               
            time.sleep(0.050) # attend delta t

            position = self.robot.get_coord_pince() #  obtient nvlle pos robot instant t+1       # Fonction à changer   
            #position_x.append(position[0])
            #position_y.append(position[1])            
            network_input[0] = (position[0]-target[0])*self.alpha[0]
            network_input[1] = (position[1]-target[1])*self.alpha[1]
            #network_input[2] = (position[2]-target[2]-theta_s(position[0], position[1]))*self.alpha[2]
            i+=1
            print(i)
            crit_ap= alpha_1*(position[0]-target[0])*(position[0]-target[0]) + alpha_2*(position[1]-target[1])*(position[1]-target[1]) 
            selfthetas1,selfthetas2,selfthetas3 = self.robot.get_theta()

            if self.training:
                delta_t = (time.time()-debut)
                temp1 = temp1 + temp
                temp = time.time()-robot_a_bouge
                self.t.append(temp+temp1)
                grad = [
                    2*(-1)*alpha_1*delta_t*(self.robot.L1*math.sin(selfthetas1)+self.robot.L2*math.sin(selfthetas1 + selfthetas2)+self.robot.L3*math.sin(selfthetas1 + selfthetas2 + selfthetas3))*(target[0] - position[0])
                    -2*(-1)*alpha_2*delta_t*(self.robot.L1*math.cos(selfthetas1)+ self.robot.L2*math.cos(selfthetas1 + selfthetas2) + self.robot.L3*math.cos(selfthetas1 + selfthetas2 + selfthetas3))*(target[1] - position[1]),
                    
                    2*(-1)*alpha_1*delta_t*(self.robot.L2*math.sin(selfthetas1+ selfthetas2)+self.robot.L3*math.sin(selfthetas1+ selfthetas2+ selfthetas3))*(target[0] - position[0])
                    -2*(-1)*alpha_2*delta_t*(self.robot.L2*math.cos(selfthetas1+ selfthetas2)+self.robot.L3*math.cos(selfthetas1+ selfthetas2+selfthetas3))*(target[1] - position[1]),
                    
                    2*(-1)*alpha_1*delta_t*(self.robot.L3*math.sin(selfthetas1+ selfthetas2+selfthetas3))*(target[0] - position[0])
                    -2*(-1)*alpha_2*delta_t*(self.robot.L3*math.cos(selfthetas1+ selfthetas2+selfthetas3))*(target[1] - position[1])                    
                ]
                # The two args after grad are the gradient learning steps for t+1 and t
                # si critere augmente on BP un bruit fction randon_update, sion on BP le gradient
                self.grad0.append(grad[0])
                self.grad1.append(grad[1])
                self.grad2.append(grad[2])
                if (crit_ap <= crit_av) :
                    self.network.backPropagate(grad, self.pas,self.moment)# grad, pas d'app, moment
                else :
                    #self.network.random_update(0.001)
                    self.network.backPropagate(grad, self.pas,self.moment)
                    
        #self.robot.train(th1,th2) 
        #self.robot.set_theta([0,0]) # stop  apres arret  du prog d'app                 # Fonction à changer 
        #self.robot.train()
        #position = self.robot.get_position() #  obtient nvlle pos robot instant t+1
        #Teta_t=position[2]
             
                
        return th1,th2,th3
