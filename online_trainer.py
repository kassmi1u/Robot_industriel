import time
import math

"""def theta_s(x,y):
    if x>0:
        return 1*math.atan(1*y)
    if x<=0:
        return 1*math.atan(-1*y)"""

class OnlineTrainer:
    def __init__(self,NN):
        """
        Args:
            robot (Robot): a robot instance following the pattern of
                VrepPioneerSimulation
            target (list): the target position [x,y,theta]
        """
        self.robot = robot
        self.network = NN

        self.alpha = [1/6,1/6]  # normalition avec limite du monde cartesien = -3m � + 3m

    def train(self, target):
        position = self.robot.get_coord_pince()
        position_angulaire = self.robot.get_theta()
        """position_x = []
        position_y = []
        position_x.append(position[0])
        position_x.append(0)
        position_y.append(position[1])
        position_y.append(1) """
        network_input = [0, 0]
        network_input[0] = (position[0]-target[0])*self.alpha[0]
        network_input[1] = (position[1]-target[1])*self.alpha[1]
        #network_input[2] = (position[2]-target[2]-theta_s(position[0], position[1]))*self.alpha[2]
        #Teta_t = 0
        #"""position_intial = position"""

        while self.running:
            debut = time.time()
            command = self.network.runNN(network_input) # propage erreur et calcul vitesses roues instant t  # Fonction à changer 
            
                      
            alpha_x = 1/6 #"""(max(position_x) - min(position_x))"""
            alpha_y = 1/6 #"""(max(position_y) - min(position_y))"""
            #alpha_teta = math.sqrt(alpha_x*alpha_x + alpha_y*alpha_y)
                        
            crit_av= alpha_1*alpha_1*(position[0]-target[0])*(position[0]-target[0]) + alpha_2*alpha_2*(position[1]-target[1])*(position[1]-target[1]) 

            
            self.robot.set_theta(command[0],command[1])           
            self.robot.train() # applique vitesses roues instant t,       # Fonction à changer               
            time.sleep(0.050) # attend delta t
            position = self.robot.get_coord_pince() #  obtient nvlle pos robot instant t+1       # Fonction à changer   
            #position_x.append(position[0])
            #position_y.append(position[1])            
            network_input[0] = (position[0]-target[0])*self.alpha[0]
            network_input[1] = (position[1]-target[1])*self.alpha[1]
            #network_input[2] = (position[2]-target[2]-theta_s(position[0], position[1]))*self.alpha[2]
            
            crit_ap= alpha_1*alpha_1*(position[0]-target[0])*(position[0]-target[0]) + alpha_2*alpha_2*(position[1]-target[1])*(position[1]-target[1]) 

            if self.training:
                delta_t = (time.time()-debut)

                grad = [
                    2*alpha_1*alpha_1*delta_t*(l1*math.sin(position_angulaire[0])+l2*math.sin(position_angulaire[0] + position_angulaire[1]))*(target(0) - position[0])
                    -2*alpha_2*alpha_2*delta_t*(l2*math.cos(position_angulaire[0])+ l2*math.cos(position_angulaire[0] + position_angulaire[1]))*(target(1) - position[1]),
                    
                    2*alpha_1*alpha_1*delta_t*(l2*math.sin(position_angulaire[0]+ position_angulaire[1]))*(target(0) - position[0])
                    -2*alpha_2*alpha_2*delta_t*(l2*math.cos(position_angulaire[0]+ position_angulaire[1]))*(target(1) - position[1])
                ]

                # The two args after grad are the gradient learning steps for t+1 and t
                # si critere augmente on BP un bruit fction randon_update, sion on BP le gradient
                
                if (crit_ap <= crit_av) :
                    self.network.backPropagate(grad, 0.5,0)# grad, pas d'app, moment
                else :
                    #self.network.random_update(0.001)
                    self.network.backPropagate(grad, 0.5 ,0)
                
        self.robot.set_theta([0,0]) # stop  apres arret  du prog d'app                 # Fonction à changer 
        self.robot.train()
        #position = self.robot.get_position() #  obtient nvlle pos robot instant t+1
                #Teta_t=position[2]
             
                
        
        self.running = False
