import time
import math

class OnlineTrainer_3ddl:
    
    def __init__(self,robot,NN):

        self.robot = robot
        self.network = NN
        self.alpha = [1,1,1/16,1/16,1/16]  
        self.training = False
        self.running = True
        self.pas = 0.5
        self.moment = 0

        # to draw the variations of the gradient, the velocity and the error
        self.grad0 = []
        self.grad1 = []
        self.grad2 = []
        self.velocity0 = []
        self.velocity1 = []
        self.error_x = []
        self.error_y = []
        self.t = []
        self.t2 = []

        # to define articulation values 
        self.articulation = False 
        self.articulation1_max = 0 
        self.articulation1_min = 0
        self.articulation2_max = 0
        self.articulation2_min = 0
        self.articulation3_max = 0
        self.articulation3_min = 0
        self.articulation1_moy = (self.articulation1_max + self.articulation1_min)/2
        self.articulation2_moy = (self.articulation2_max + self.articulation2_min)/2
        self.articulation3_moy = (self.articulation3_max + self.articulation3_min)/2

    def train(self, targett):
        
        # get postions
        position = self.robot.get_coord_pince()
        th1 = []
        th2 = []
        th3 = []        

        # get the target 
        target = [float(targett[0].get()),float(targett[1].get())]
        
        # to update alpha1 and alpha2 values
        position_x = []
        position_y = []
        position_x.append(position[0])
        position_x.append(0)
        position_y.append(position[1])
        position_y.append(1)

        # to count number of iterations 
        i=0

        # to measure time 
        robot_a_bouge = time.time()
        temp = 0
        temp1 = 0
        temp3 = 0
        temp4 = 0

        while (abs(position[0]-target[0]) > 0.006 or  abs(position[1]-target[1]) > 0.006 ) and i<1500 : 

            # weighting coefficient of the elements of the criterion 
            alpha_1 = 1/(max(position_x) - min(position_x))
            alpha_2 = 1/(max(position_y) - min(position_y))
            alpha_3 = self.alpha[2]
            alpha_4 = self.alpha[3]
            alpha_5 = self.alpha[4]

            # initate the network inputs
            debut = time.time()
            selfthetas11,selfthetas22,selfthetas33 = self.robot.get_theta()
            if self.articulation : 
                network_input = [(position[0]-target[0])*self.alpha[0], (position[1]-target[1])*self.alpha[1], \
                    self.alpha[2]*((selfthetas11-self.articulation1_moy)/(self.articulation1_max-self.articulation1_min)**2), \
                        self.alpha[3]*((selfthetas22-self.articulation2_moy)/(self.articulation2_max-self.articulation2_min)**2),\
                            self.alpha[4]*((selfthetas33-self.articulation3_moy)/(self.articulation3_max-self.articulation3_min)**2)]
            else : 
                network_input = [(position[0]-target[0])*self.alpha[0], (position[1]-target[1])*self.alpha[1]]

            # Velocity
            command = self.network.runNN(network_input)


            # useful to draw velocity graph 
            temp4 = temp4 + temp3
            temp3 = time.time() - robot_a_bouge
            self.t2.append(temp3+temp4)
            self.velocity0.append(command[0])
            self.velocity1.append(command[1])
            self.error_x.append(position[0]-target[0])
            self.error_y.append(position[1]-target[1])

            # generate angle values                  
            diff = time.time() - robot_a_bouge
            thetas = self.robot.get_theta()
            theta_temp1 = thetas[0] + command[0] * diff
            theta_temp2 = thetas[1] + command[1] * diff
            theta_temp3 = thetas[2] + command[1] * diff  

            # useful to animate the robot             
            self.robot.set_theta(theta_temp1,theta_temp2,theta_temp3)    
            robot_a_bouge = time.time()  
            th1.append(theta_temp1)
            th2.append(theta_temp2)
            th3.append(theta_temp3)            
                     
            time.sleep(0.050) 

            # to update alpha values
            position = self.robot.get_coord_pince() 
            position_x.append(position[0])
            position_y.append(position[1]) 
            print(position[0],position[1])
            
            i+=1
            print(i)
            selfthetas1,selfthetas2,selfthetas3 = self.robot.get_theta()

            # training
            if self.training:
                delta_t = (time.time()-debut)
                temp1 = temp1 + temp
                temp = time.time()-robot_a_bouge
                self.t.append(temp+temp1)
                if self.articulation : 

                    # Ariticulations case
                    grad = [
                        2*(-1)*alpha_1*delta_t*(self.robot.L1*math.sin(selfthetas1)+self.robot.L2*math.sin(selfthetas1 + selfthetas2)+self.robot.L3*math.sin(selfthetas1 + selfthetas2 + selfthetas3))*(target[0] - position[0])
                        -2*(-1)*alpha_2*delta_t*(self.robot.L1*math.cos(selfthetas1)+ self.robot.L2*math.cos(selfthetas1 + selfthetas2) + self.robot.L3*math.cos(selfthetas1 + selfthetas2 + selfthetas3))*(target[1] - position[1])
                        + 2*alpha_3*delta_t*(1/(self.articulation1_max-self.articulation1_min)**2)*(selfthetas1 + delta_t*command[0] -self.articulation1_moy),

                        2*(-1)*alpha_1*delta_t*(self.robot.L2*math.sin(selfthetas1+ selfthetas2)+self.robot.L3*math.sin(selfthetas1+ selfthetas2+ selfthetas3))*(target[0] - position[0])
                        -2*(-1)*alpha_2*delta_t*(self.robot.L2*math.cos(selfthetas1+ selfthetas2)+self.robot.L3*math.cos(selfthetas1+ selfthetas2+selfthetas3))*(target[1] - position[1])
                        - 2*alpha_4*delta_t*(1/(self.articulation2_max-self.articulation2_min)**2)*(selfthetas2 + delta_t*command[1] -self.articulation2_moy),

                        2*(-1)*alpha_1*delta_t*(self.robot.L3*math.sin(selfthetas1+ selfthetas2+selfthetas3))*(target[0] - position[0])
                        -2*(-1)*alpha_2*delta_t*(self.robot.L3*math.cos(selfthetas1+ selfthetas2+selfthetas3))*(target[1] - position[1])
                        - 2*alpha_5*delta_t*(1/(self.articulation3_max-self.articulation3_min)**2)*(selfthetas3 + delta_t*command[1] -self.articulation3_moy)                    
                    ]

                else : 
                    # without articulations case
                    grad = [
                        2*(-1)*alpha_1*delta_t*(self.robot.L1*math.sin(selfthetas1)+self.robot.L2*math.sin(selfthetas1 + selfthetas2)+self.robot.L3*math.sin(selfthetas1 + selfthetas2 + selfthetas3))*(target[0] - position[0])
                        -2*(-1)*alpha_2*delta_t*(self.robot.L1*math.cos(selfthetas1)+ self.robot.L2*math.cos(selfthetas1 + selfthetas2) + self.robot.L3*math.cos(selfthetas1 + selfthetas2 + selfthetas3))*(target[1] - position[1]),
                        
                        2*(-1)*alpha_1*delta_t*(self.robot.L2*math.sin(selfthetas1+ selfthetas2)+self.robot.L3*math.sin(selfthetas1+ selfthetas2+ selfthetas3))*(target[0] - position[0])
                        -2*(-1)*alpha_2*delta_t*(self.robot.L2*math.cos(selfthetas1+ selfthetas2)+self.robot.L3*math.cos(selfthetas1+ selfthetas2+selfthetas3))*(target[1] - position[1]),
                        
                        2*(-1)*alpha_1*delta_t*(self.robot.L3*math.sin(selfthetas1+ selfthetas2+selfthetas3))*(target[0] - position[0])
                        -2*(-1)*alpha_2*delta_t*(self.robot.L3*math.cos(selfthetas1+ selfthetas2+selfthetas3))*(target[1] - position[1])                    
                    ]

                self.network.backPropagate(grad, self.pas,self.moment)

                # useful to draw gradient graph
                self.grad0.append(grad[0])
                self.grad1.append(grad[1])
                self.grad2.append(grad[2])
                

        return th1,th2,th3
