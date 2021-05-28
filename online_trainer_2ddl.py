import time
import math

class OnlineTrainer_2ddl:
    
    def __init__(self,robot,NN):

        self.robot = robot
        self.network = NN
        self.alpha = [1,1,1/16,1/16,1/6,1/6]  
        self.training = False
        self.running = True
        self.moment = 0
        self.pas = 0.5

        # to draw the variations of the gradient, the velocity and the error
        self.grad0 = []
        self.grad1 = []
        self.velocity0 = []
        self.velocity1 = []
        self.error_x = []
        self.error_y = []

        # to define articulation values 
        self.articulation = False
        self.articulation1_max = math.pi
        self.articulation1_min = - math.pi
        self.articulation2_max = math.pi
        self.articulation2_min = - math.pi
        self.articulation1_moy = (self.articulation1_max + self.articulation1_min)/2
        self.articulation2_moy = (self.articulation2_max + self.articulation2_min)/2
        
        # time intervalls
        self.t1 = []
        self.t2 = []

    def train(self, targett):
        
        # get positions
        position = self.robot.get_coord_pince()
        th1 = []
        th2 = []

        # to update alpha1 and alpha2 values
        position_x = []
        position_y = []
        position_x.append(position[0])
        position_x.append(3)
        position_y.append(position[1])
        position_y.append(1)

        # get the target
        target = [float(targett[0].get()),float(targett[1].get())]
        robot_a_bouge = time.time()
        i=0
        yp=0
        b=0

        # time variables: useful for measuring time to plot graphs
        temp1 = 0
        temp2 = 0
        temp3 = 0
        temp4 = 0
        command=[0,0]
        delta_t = 0.05

        while  (abs(position[0]-target[0]) > 0.006 or  abs(position[1]-target[1]) > 0.006) and i<1500:
            
            # weighting coefficient of the elements of the criterion 
            alpha_1 = 1/(max(position_x) - min(position_x))
            alpha_2 = 1/(max(position_y) - min(position_y))
            alpha_3 = self.alpha[2]
            alpha_4 = self.alpha[3]
            alpha_5 = self.alpha[4]

            # initate the network inputs
            debut = time.time()
            selfthetas11,selfthetas22 = self.robot.get_theta()
            yp =  self.robot.L1*math.sin(selfthetas11 + delta_t*command[0])+self.robot.L2*math.sin(selfthetas11+selfthetas22+delta_t*(command[0]+command[1]))
            if self.articulation : 
                network_input = [(position[0]-target[0])*self.alpha[0], (position[1]-target[1])*self.alpha[1], \
                    self.alpha[2]*((selfthetas11-self.articulation1_moy)/(self.articulation1_max-self.articulation1_min)**2),\
                         self.alpha[3]*((selfthetas22-self.articulation2_moy)/(self.articulation2_max-self.articulation2_min)**2)]
            else : 
                network_input = [(position[0]-target[0])*self.alpha[0], (position[1]-target[1])*self.alpha[1],yp*alpha_5*b]
            
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

            # useful to animate the robot     
            self.robot.set_theta(theta_temp1,theta_temp2)    
            robot_a_bouge = time.time()  
            th1.append(theta_temp1)
            th2.append(theta_temp2)
                      
            time.sleep(0.050)
            
            # to update alpha values
            position = self.robot.get_coord_pince() 
            position_x.append(position[0])
            position_y.append(position[1]) 
            print(position[0],position[1])

            i+=1
            print(i)
            selfthetas1,selfthetas2 = self.robot.get_theta()

            if position[1] < 0: 
                b = 1
            else : 
                b = 0

            

            # Training
            if self.training:

                delta_t = (time.time()-debut)
               

                if self.articulation : 

                    # Articulations case
                    grad = [
                        2*(-1)*alpha_1*delta_t*(self.robot.L1*math.sin(selfthetas1)+self.robot.L2*math.sin(selfthetas1 + selfthetas2))*(target[0] - position[0])
                        -2*(-1)*alpha_2*delta_t*(self.robot.L1*math.cos(selfthetas1)+ self.robot.L2*math.cos(selfthetas1 + selfthetas2))*(target[1] - position[1])
                        + 2*alpha_3*delta_t*(1/(self.articulation1_max-self.articulation1_min)**2)*(selfthetas1 + delta_t*command[0] -self.articulation1_moy),
                        
                        2*(-1)*alpha_1*delta_t*(self.robot.L2*math.sin(selfthetas1+ selfthetas2))*(target[0] - position[0])
                        -2*(-1)*alpha_2*delta_t*(self.robot.L2*math.cos(selfthetas1+ selfthetas2))*(target[1] - position[1])
                        - 2*alpha_4*delta_t*(1/(self.articulation2_max-self.articulation2_min)**2)*(selfthetas2 + delta_t*command[1] -self.articulation2_moy)
                    ]
                else : 

                    # Withtout articulations case
                    grad = [
                        2*(-1)*alpha_1*delta_t*(self.robot.L1*math.sin(selfthetas1)+self.robot.L2*math.sin(selfthetas1 + selfthetas2))*(target[0] - position[0])
                        -2*(-1)*alpha_2*delta_t*(self.robot.L1*math.cos(selfthetas1)+ self.robot.L2*math.cos(selfthetas1 + selfthetas2))*(target[1] - position[1])
                        + 2*alpha_5*b*(self.robot.L1*self.robot.L1*delta_t*math.cos(selfthetas1 + delta_t*command[0])*math.sin(selfthetas1+delta_t*command[0]) +self.robot.L2*self.robot.L2*delta_t*math.cos(selfthetas1 + selfthetas2 + delta_t*(command[0]+command[1]))*math.sin(selfthetas1 + selfthetas2 + delta_t*(command[0]+command[1]))+self.robot.L1*self.robot.L2*delta_t*math.sin(2*(selfthetas1+delta_t*command[0])+selfthetas2+delta_t*command[1])),
                        
                        2*(-1)*alpha_1*delta_t*(self.robot.L2*math.sin(selfthetas1+ selfthetas2))*(target[0] - position[0])
                        -2*(-1)*alpha_2*delta_t*(self.robot.L2*math.cos(selfthetas1+ selfthetas2))*(target[1] - position[1])
                        + 2*alpha_5*b*(self.robot.L2*self.robot.L2*delta_t*math.cos(selfthetas1 + selfthetas2 + delta_t*(command[0]+command[1]))*math.sin(selfthetas1 + selfthetas2 + delta_t*(command[0]+command[1]))+self.robot.L1*self.robot.L2*delta_t*math.sin(selfthetas1+delta_t*command[0])*math.cos(selfthetas1+selfthetas2+delta_t*(command[0]+command[1])))
                    ]

                self.network.backPropagate(grad, self.pas ,self.moment)

                # useful to draw the gradient graph
                temp1 = temp1 + temp2
                temp2 = time.time()-robot_a_bouge
                self.t1.append(temp2+temp1)
                self.grad0.append(grad[0])
                self.grad1.append(grad[1])


        return th1,th2
    
