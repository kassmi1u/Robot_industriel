from online_trainer_2ddl import OnlineTrainer_2ddl
from online_trainer_3ddl import OnlineTrainer_3ddl
from BackProp_Python_v2 import NN
from simulation_2ddl import Robot_manipulator_2ddl
from simulation_3ddl import Robot_manipulator_3ddl
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
window_width = 1000

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
save_animation = tk.IntVar()
target_x = tk.DoubleVar()
target_y = tk.DoubleVar()
hidden = tk.IntVar()
choise = tk.IntVar()
step = tk.DoubleVar()
moment = tk.DoubleVar()
animation_name= tk.StringVar()

moment.set(0.00)
step.set(0.5)
choise.set(1)
hidden.set(20)
save_animation.set(1)
learning.set(1)
load_model.set(1)



# Network Settings zone
frame2 = ttk.LabelFrame(root, text=' Network Settings ', width=370, height=110)
frame2.place(x=23, y=45)

# Step text
Label1 = ttk.Label(root, text = ' Step ')
Label1.place (x=195,y=80)

# Step value
boite21 = ttk.Spinbox(root,from_=0.05,to=5,increment=0.05,textvariable=step,width=5)
boite21.place(x=180,y=106)

# Moment text
Label1 = ttk.Label(root, text = ' Moment ')
Label1.place (x=293,y=80)

# Moment value
boite21 = ttk.Spinbox(root,from_=0.0,to=5,increment=0.05,textvariable=moment,width=5)
boite21.place(x=285,y=106)

# Network zone
frame1 = ttk.LabelFrame(root, text=' Network ', width=370, height=110)
frame1.place(x=23, y=183)

# load_model box
switch = ttk.Checkbutton(root, text='Activate to load the previous network', style='Switch', variable=load_model, offvalue=0, onvalue=1)
switch.place(x=50, y=218)
switch.invoke()

# learning box
switch1 = ttk.Checkbutton(root, text='Activate to learn', style='Switch', variable=learning, offvalue=0, onvalue=1)
switch1.place(x=50, y=252)
switch1.invoke()

# Target zone
frame2 = ttk.LabelFrame(root, text=' Target ', width=370, height=130)
frame2.place(x=23, y=321)

# target_x text
Label3 = ttk.Label(root, text = 'Enter the coordinate x of the target ')
Label3.place (x=50,y=358)

# target_x value
boite3 = ttk.Spinbox(root,from_=0.05,to=0.95,increment=0.05,textvariable=target_x,width=5)
boite3.place(x=300,y=353)

# target_y text
Label2 = ttk.Label(root, text = 'Enter the coordinate y of the target  ')
Label2.place (x=50,y=398)

# target_y box 
boite4 = ttk.Spinbox(root,from_=0.05,to=0.95,increment=0.05,textvariable=target_y,width=5)
boite4.place(x=300,y=393)

# Robot model zone
frame21 = ttk.LabelFrame(root, text='  Robot model  ', width=170, height=110)
frame21.place(x=413, y=45)

# Robot model choise
choise1 = ttk.Radiobutton(frame21, text=' Bras à 2DDL', variable=choise, value=1)
choise1.place(x=20, y=15)
choise2 = ttk.Radiobutton(frame21, text=' Bras à 3DDL', variable=choise, value=2)
choise2.place(x=20, y=45)

# Obstace zone
frame22 = ttk.LabelFrame(root, text='  Obstacle ', width=170, height=110)
frame22.place(x=413, y=183)

# Animation zone
frame23 = ttk.LabelFrame(root, text='  Animation ', width=170, height=130)
frame23.place(x=413, y=321)

# Save Animation Button
switch2 = ttk.Checkbutton(frame23, text=' Activate to save\n the animation as\n a Gif image', style='Switch', variable=save_animation, offvalue=0, onvalue=1)
switch2.place(x=10, y=10)
switch2.invoke()

# Gif Image name
Answer = ttk.Entry(frame23,text = "File name ",width = 15,textvariable=animation_name)
Answer.insert(0,'File name')
Answer.place(x=10,y=70)

# Hidden Layer text
Label1 = ttk.Label(root, text = 'Hidden Layer size')
Label1.place (x=50,y=80)

# Hidden Layer size value
boite1 = ttk.Spinbox(root,from_=5,to=100,increment=1,textvariable=hidden,width=5)
boite1.place(x=65,y=106)

# Graph zone
Graph_frame = ttk.LabelFrame(root, text='  Graph  ', width=370, height=130)
Graph_frame.place(x=605, y=321)

#Trajectory Label 
tragectory_Label = ttk.Label(Graph_frame,text = "The trajectory of the clamp, the elbows .. ")
tragectory_Label.place(x=15,y=10)
 
     
HL_size = int(hidden.get())
network = NN(2,HL_size,2)
robot = Robot_manipulator_2ddl()
training= OnlineTrainer_2ddl(robot,network)
thetas1 =[]
thetas2 =[]
thetas3 =[]
file_2ddl='last_w_2ddl.json'
file_3ddl='last_w_3ddl.json'
state_train = False

def train_network() :

    global thetas1,thetas2,thetas3,network,HL_size,file_2ddl,file_3ddl,\
    training,load_model,learning,target_x,target_y,network,robot,step,moment,state_train

    HL_size = int(hidden.get())
    print(HL_size)
    network = NN(2,HL_size,2)
    robot_2ddl = Robot_manipulator_2ddl()
    robot_3ddl = Robot_manipulator_3ddl()   
    training_2ddl = OnlineTrainer_2ddl(robot_2ddl,network)
    training_3ddl= OnlineTrainer_3ddl(robot_3ddl,network) 

    if int(choise.get()) == 1 : 
        robot = robot_2ddl
        training = training_2ddl
        file = file_2ddl
        
    else :
        robot = robot_3ddl
        training = training_3ddl
        file = file_3ddl

    training.pas = float(step.get())
    training.moment = float(moment.get())

    if int(load_model.get()) == 1:
        with open(file) as fp:
            json_obj = json.load(fp)
        for i in range(2):
            for j in range(HL_size):
                network.wi[i][j] = json_obj["input_weights"][i][j]
        for i in range(HL_size):
            for j in range(2):
                network.wo[i][j] = json_obj["output_weights"][i][j] 
    
    target = [target_x,target_y]

    if int(learning.get()) == 1:
        training.training = True
    elif int(learning.get()) == 0:
        training.training = False  

    if int(choise.get()) == 1 :
        thetas1,thetas2 = training.train(target)
    else :
        thetas1,thetas2,thetas3 = training.train(target)
    state_train = True
    json_obj = {"input_weights": network.wi, "output_weights": network.wo}
    with open(file, 'w') as fp:
        json.dump(json_obj, fp)
    print("The last weights have been stored in last_w.json")
    print("arrivé")
    
def animate() : 
    global thetas1,thetas2,thetas3,target_x,target_y,robot
    if int(save_animation.get()) == 1 : 
        robot.save = True
        robot.name_file = str(animation_name.get())
    tar = [float(target_x.get()),float(target_y.get())]
    Fig, ax = robot.draw_env(tar)
    if int(choise.get()) == 1 :
        line1, line2, pt1 = robot.draw_robot(Fig,ax)
        robot.train(thetas1,thetas2,line1,line2,pt1,Fig) 
    else :
        line1,line2,line3, pt1 = robot.draw_robot(Fig,ax)
        robot.train(thetas1,thetas2,thetas3,line1,line2,line3,pt1,Fig) 
       
def quit():
    root.destroy()

def Reset():
    global target_x,target_y,learning,load_model
    target_x.set(0.00)
    target_y.set(0.00)
    learning.set(0)
    load_model.set(0)
    choise.set(1)
    hidden.set(20)

def plot_graph():
    global robot,thetas1,thetas2,thetas3,target_x,target_y,state_train
    tar = [float(target_x.get()),float(target_y.get())]
    if state_train : 
        if int(choise.get()) == 1 :
            robot.draw_Trajectory(thetas1,thetas2,tar)
        else :
            robot.draw_Trajectory(thetas1,thetas2,thetas3,tar)

button1= ttk.Button(root, text='Train', width=18, command=train_network)
button1.place(x=23, y=470)

button2 = ttk.Button(root, text='Animate',width=18,command=animate)
button2.place(x=215, y=470)

button3 = ttk.Button(root, text='Quit',width=18,command = quit)
button3.place(x=600, y=470)

button4 = ttk.Button(root, text='Reset',width = 18, command=Reset)
button4.place(x=408, y=470)

show_button1 = ttk.Button(Graph_frame, text=' Show ',width = 7, command=plot_graph)
show_button1.place(x=270, y=5)

root.mainloop()