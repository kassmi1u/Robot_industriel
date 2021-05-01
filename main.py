from online_trainer import OnlineTrainer
from BackProp_Python_v2 import NN
from Python_simulation import Robot_manipulator
import numpy as np
#import json
#import threading

robot = Robot_manipulator()
target = [0.7,0.9]
Network = NN(2,20,2)
training = OnlineTrainer(robot,Network)
training.training = True
thetas1,thetas2 = training.train(target)
robot.train(thetas1,thetas2)