from online_trainer import OnlineTrainer
from BackProp_Python_v2 import NN
from Python_simulation import Robot_manipulator
import numpy as np
import time 
import math
import random
import json
import tkinter as tk
from tkinter import ttk


# Interface

root = tk.Tk()
root.title('Azure')

window_height = 530
window_width = 800

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

x_cordinate = int((screen_width/2) - (window_width/2))
y_cordinate = int((screen_height/2) - (window_height/2))

root.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))

style = ttk.Style(root)
root.tk.call('source', 'azure.tcl')
style.theme_use('azure')

options = ['', 'OptionMenu', 'Value 1', 'Value 2']

# Variable values
load_model = tk.IntVar()
learning = tk.IntVar()
target_x = tk.DoubleVar()
target_y = tk.DoubleVar()


# Network zone
frame1 = ttk.LabelFrame(root, text=' Network ', width=370, height=110)
frame1.place(x=23, y=45)

# load_model box
switch = ttk.Checkbutton(root, text='Activate to load the previous network', style='Switch', variable=load_model, offvalue=0, onvalue=1)
switch.place(x=50, y=80)
switch.invoke()

# learning box
switch = ttk.Checkbutton(root, text='Activate to learn', style='Switch', variable=learning, offvalue=0, onvalue=1)
switch.place(x=50, y=114)
switch.invoke()

# Target zone
frame1 = ttk.LabelFrame(root, text=' Target ', width=370, height=130)
frame1.place(x=23, y=183)

# target_x text
Label1 = ttk.Label(root, text = 'Enter the coordinate x of the target ')
Label1.place (x=50,y=220)

# target_x value
boite = ttk.Spinbox(root,from_=0.05,to=0.95,increment=0.05,textvariable=target_x,width=5)
boite.place(x=300,y=215)

# target_y text
Label1 = ttk.Label(root, text = 'Enter the coordinate y of the target  ')
Label1.place (x=50,y=260)

# target_y box 
boite = ttk.Spinbox(root,from_=0.05,to=0.95,increment=0.05,textvariable=target_y,width=5)
boite.place(x=300,y=255)



robot = Robot_manipulator()
HL_size= 10
network = NN(2,HL_size,2)
training = OnlineTrainer(robot,network)
thetas1 =[]
thetas2 =[]
file='last_w.json'

def train_network() :
    global thetas1,thetas2,network,HL_size,file,training,load_model,learning,target_x,target_y
    #choice = input('Do you want to load previous network? (y/n) --> ')
    if load_model == 1:
        
        with open(file) as fp:
            json_obj = json.load(fp)
        for i in range(2):
            for j in range(HL_size):
                network.wi[i][j] = json_obj["input_weights"][i][j]
        for i in range(HL_size):
            for j in range(2):
                network.wo[i][j] = json_obj["output_weights"][i][j] 
    
    target = [target_x,target_y]
    
    if learning == 1:
        training.training = True
    elif learning == 0:
        training.training = False    
    print("départ")
    thetas1,thetas2 = training.train(target)
    print("arrivé")

def animate() : 
    global thetas1,thetas2,target_x,target_y
    tar = [float(target_x.get()),float(target_y.get())]
    Fig, ax = robot.draw_env(tar)
    line1, line2, pt1 = robot.draw_robot(Fig,ax)
    robot.train(thetas1,thetas2,line1,line2,pt1,Fig)    

button = ttk.Button(root, text='Train', command=train_network)
button.place(x=480, y=320)

button = ttk.Button(root, text='Animate', command=animate)
button.place(x=600, y=320)

"""
json_obj = {"input_weights": network.wi, "output_weights": network.wo}
with open('last_w.json', 'w') as fp:
    json.dump(json_obj, fp)

print("The last weights have been stored in last_w.json")"""
root.mainloop()