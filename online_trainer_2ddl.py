import time
import math

def deg_to_rad(angle):
    return (angle*math.pi)/180

class OnlineTrainer_2ddl:
    
    def __init__(self,robot,NN):

        self.robot = robot
        self.network = NN
        self.alpha = [1,1]  
        self.training = False
        self.running = True
        self.moment = 0
        self.pas = 0.5
        self.grad0 = []
        self.grad1 = []
        self.velocity0 = []
        self.velocity1 = []
        self.error_x = []
        self.error_y = []
        # time intervalls
        self.t1 = []
        self.t2 = []

    def train(self, targett):
        
        position = self.robot.get_coord_pince()
        alpha_1 = self.alpha[0]
        alpha_2 = self.alpha[1]
        th1 = []
        th2 = []
        # to update alpha1 and alpha2 values
        position_x = []
        position_y = []
        position_x.append(position[0])
        position_x.append(3)
        position_y.append(position[1])
        position_y.append(1)

        # temporaire
        target = [float(targett[0].get()),float(targett[1].get())]
        print(target)
        robot_a_bouge = time.time()
        i=0

        # time variables: useful for measuring time to plot graphs
        temp1 = 0
        temp2 = 0
        temp3 = 0
        temp4 = 0

        while abs(position[0]-target[0]) > 0.001 or  abs(position[1]-target[1]) > 0.001 : 
            
            alpha_1 = 1/(max(position_x) - min(position_x))
            alpha_2 = 1/(max(position_y) - min(position_y))
            
            debut = time.time()

            network_input = [(position[0]-target[0])*self.alpha[0], (position[1]-target[1])*self.alpha[1]]
            command = self.network.runNN(network_input)

            # useful to draw velocity graph 
            temp4 = temp4 + temp3
            temp3 = time.time() - robot_a_bouge
            self.t2.append(temp3+temp4)
            self.velocity0.append(command[0])
            self.velocity1.append(command[1])
            self.error_x.append(position[0]-target[0])
            self.error_y.append(position[1]-target[1])

            crit_av= alpha_1*(position[0]-target[0])*(position[0]-target[0]) + alpha_2*(position[1]-target[1])*(position[1]-target[1]) 

            # get velocity and generate angle values       
            diff = time.time() - robot_a_bouge
            thetas = self.robot.get_theta()
            theta_temp1 = thetas[0] + command[0] * diff
            theta_temp2 = thetas[1] + command[1] * diff
            self.robot.set_theta(theta_temp1,theta_temp2)    
            robot_a_bouge = time.time()  
            th1.append(theta_temp1)
            th2.append(theta_temp2)
                      
            time.sleep(0.050)

            position = self.robot.get_coord_pince() #  obtient nvlle pos robot instant t+1       # Fonction à changer 
  
            position_x.append(position[0])
            position_y.append(position[1]) 

                  
            network_input[0] = (position[0]-target[0])*self.alpha[0]
            network_input[1] = (position[1]-target[1])*self.alpha[1]

            i+=1
            print(i)

            crit_ap= alpha_1*(position[0]-target[0])*(position[0]-target[0]) + alpha_2*(position[1]-target[1])*(position[1]-target[1]) 
            selfthetas1,selfthetas2 = self.robot.get_theta()

            if self.training:

                delta_t = (time.time()-debut)
                grad = [
                    2*(-1)*alpha_1*delta_t*(self.robot.L1*math.sin(selfthetas1)+self.robot.L2*math.sin(selfthetas1 + selfthetas2))*(target[0] - position[0])
                    -2*(-1)*alpha_2*delta_t*(self.robot.L1*math.cos(selfthetas1)+ self.robot.L2*math.cos(selfthetas1 + selfthetas2))*(target[1] - position[1]),
                    
                    2*(-1)*alpha_1*delta_t*(self.robot.L2*math.sin(selfthetas1+ selfthetas2))*(target[0] - position[0])
                    -2*(-1)*alpha_2*delta_t*(self.robot.L2*math.cos(selfthetas1+ selfthetas2))*(target[1] - position[1])
                ]

                # useful to draw the gradient graph
                temp1 = temp1 + temp2
                temp2 = time.time()-robot_a_bouge
                self.t1.append(temp2+temp1)
                self.grad0.append(grad[0])
                self.grad1.append(grad[1])

                if (crit_ap <= crit_av) :
                    self.network.backPropagate(grad,self.pas,self.moment)
                else :
                    self.network.backPropagate(grad, self.pas ,self.moment)
                

        return th1,th2
    
