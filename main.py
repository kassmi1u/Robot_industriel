

from online_trainer_2ddl import OnlineTrainer_2ddl
from online_trainer_3ddl import OnlineTrainer_3ddl
from BackProp_Python_v2 import NN
from simulation_2ddl import Robot_manipulator_2ddl
from simulation_3ddl import Robot_manipulator_3ddl
import numpy as np
import time 
import math
import webbrowser
import random
import json
import tkinter as tk
from tkinter import PhotoImage, ttk


# Interface -------------------------------------------------------------------------------

root = tk.Tk()

# Icone -------------------------------------------------------------------------------

root.title('ARM ROBOT SIMULATOR')
photo1 = PhotoImage(file = "test.png")
root.iconphoto(False,photo1)

# Logo -------------------------------------------------------------------------------

photo2 = PhotoImage(file = "test5.png")
photo_label = tk.Label(root, image = photo2)
photo_label.place(x=740,y=27)

# Window Settings ---------------------------------------------------------------------

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

# Button variables ------------------------------------------------------------------

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
articulation = tk.IntVar()
articulation1_max = tk.DoubleVar()
articulation1_min = tk.DoubleVar()
articulation2_max = tk.DoubleVar()
articulation2_min = tk.DoubleVar()
articulation3_max = tk.DoubleVar()
articulation3_min = tk.DoubleVar()
notification = "This simulator aims is to train a neural network to reach \na target with a \
robotic arm. \nVary parameters values ​​and test !"

# Initialization of button variables --------------------------------------------------

moment.set(0.00)
step.set(0.5)
choise.set(1)
hidden.set(20)
save_animation.set(1)
learning.set(1)
load_model.set(1)
articulation.set(1)
articulation1_max.set(3.14)
articulation1_min.set(-3.14)
articulation2_max.set(3.14)
articulation2_min.set(-3.14)
articulation3_max.set(3.14)
articulation3_min.set(-3.14)

# Buttons, text fields and labels -----------------------------------------------------

# ****************************** Network Settings zone ********************************

frame2 = ttk.LabelFrame(root, text=' Network Settings ', width=370, height=130)
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
# Hidden Layer text
Label1 = ttk.Label(root, text = 'Hidden Layer size')
Label1.place (x=50,y=80)
# Hidden Layer size value
boite1 = ttk.Spinbox(root,from_=1,to=100,increment=1,textvariable=hidden,width=5)
boite1.place(x=65,y=106)

# ************************************ Network zone ************************************

frame1 = ttk.LabelFrame(root, text=' Network ', width=370, height=130)
frame1.place(x=23, y=183)
# load_model box
switch = ttk.Checkbutton(root, text='Activate to load the previous network', style='Switch', \
variable=load_model, offvalue=0, onvalue=1)
switch.place(x=50, y=218)
switch.invoke()
# learning box
switch1 = ttk.Checkbutton(root, text='Activate to learn', style='Switch', variable=learning, offvalue=0, onvalue=1)
switch1.place(x=50, y=252)
switch1.invoke()

# ************************************ Target zone ************************************

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

# ************************************ Robot model zone ************************************

frame21 = ttk.LabelFrame(root, text='  Robot model  ', width=170, height=130)
frame21.place(x=413, y=183)
# Robot model choise
choise1 = ttk.Radiobutton(frame21, text=' 2DOF Arm', variable=choise, value=1)
choise1.place(x=20, y=15)
choise2 = ttk.Radiobutton(frame21, text=' 3DOF Arm', variable=choise, value=2)
choise2.place(x=20, y=45)

# ************************************ Articulation zone ************************************

frame22 = ttk.LabelFrame(root, text='  Articulation ', width=325, height=130)
frame22.place(x=413, y=45)
# Aritculation config
switch = ttk.Checkbutton(frame22, text=' On / Off ', style='Switch', variable=articulation, offvalue=0, onvalue=1)
switch.place(x=25, y=15)
switch.invoke()
# Articulation values
Answer1_max = ttk.Entry(frame22,text = "3.14" ,width = 4,textvariable=articulation1_max)
Answer1_max.place(x=45,y=68)
Answer1_min = ttk.Entry(frame22,text = "-3.14" ,width = 4,textvariable=articulation1_min)
Answer1_min.place(x=98,y=68)
Answer2_max = ttk.Entry(frame22,text = "3.14" ,width = 4,textvariable=articulation2_max)
Answer2_max.place(x=210,y=25)
Answer2_min = ttk.Entry(frame22,text = "-3.14" ,width = 4,textvariable=articulation2_min)
Answer2_min.place(x=263,y=25)
Answer3_max = ttk.Entry(frame22,text = "3.14" ,width = 4,textvariable=articulation3_max)
Answer3_max.place(x=210,y=68)
Answer3_min = ttk.Entry(frame22,text = "-3.14" ,width = 4,textvariable=articulation3_min)
Answer3_min.place(x=263,y=68)

# ************************************ Articulation label ************************************

articulation1_number = ttk.Label(frame22,text="N°1")
articulation1_number.place(x=10,y=73)
articulation_Label_1_MAX = ttk.Label(frame22,text = "MAX")
articulation_Label_1_MAX.place(x=55,y=48)
articulation_Label_1_MIN = ttk.Label(frame22,text = "MIN")
articulation_Label_1_MIN.place(x=108,y=48)
articulation2_number = ttk.Label(frame22,text="N°2")
articulation2_number.place(x=175,y=30)
articulation3_number = ttk.Label(frame22,text="N°3")
articulation3_number.place(x=175,y=73)
articulation_Label_3_MAX = ttk.Label(frame22,text = "MAX")
articulation_Label_3_MAX.place(x=220,y=3)
articulation_Label_3_MIN = ttk.Label(frame22,text = "MIN")
articulation_Label_3_MIN.place(x=273,y=3)

# ************************************ Animation zone ************************************

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

# ************************************  Graph zone ************************************ 

Graph_frame = ttk.LabelFrame(root, text='  Graphs  ', width=370, height=130)
Graph_frame.place(x=605, y=321)
# Trajectory Label 
tragectory_Label = ttk.Label(Graph_frame,text = "The trajectory of the clamp, the elbows .. ")
tragectory_Label.place(x=15,y=7)
# Gradient Label 
Gradient_Label = ttk.Label(Graph_frame,text = "Gradient \u2207Ji ")
Gradient_Label.place(x=24,y=40)
# Velocity Label 
Velocity_Label = ttk.Label(Graph_frame,text = "Velocity \u03C9 i ")
Velocity_Label.place(x=149,y=40)
# Error Label 
Error_Label = ttk.Label(Graph_frame,text = "Error \u03B5x, \u03B5y ")
Error_Label.place(x=270,y=40)

# ************************************  Notification Zone ************************************ 

Notification_frame = ttk.LabelFrame(root, text='  Notification  ', style = 'User.TLabelframe',width=370, height=130)
Notification_frame.place(x=605, y=183)
notification_Label = ttk.Label(Notification_frame, text = notification)
notification_Label.place (x=20,y=6)
remarque_Label = ttk.Label(Notification_frame, text = "You must wait for the end of the training before starting \
    the\n animation. This simulator does not do live animation !", font=('Times',12,'italic'))
remarque_Label.place (x=20,y=60)

# definition of neuron networks, robot ... ----------------------------------------------------

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

# display of error messages, end of simulation, animation ...
def notif(notification):
    global root,Notification_frame,notification_Label
    notification_Label.destroy()
    notification_Label = ttk.Label(Notification_frame, text = notification)
    notification_Label.place (x=20,y=6)
notif(notification)

# checks if the target is in the reachable zone 
def target_check() : 
    global target_x,target_y,choise
    tar_x = float(target_x.get())
    tar_y = float(target_y.get())
    if int(choise.get()) == 1 : 
        # le cas d'un robot 2.d.d.l
        if math.sqrt(tar_x*tar_x + tar_y*tar_y) > 1 : 
            return 0 
    else : 
        # le cas d'un robot 3.d.d.l 
        if math.sqrt(tar_x*tar_x + tar_y*tar_y) > 1.5 : 
            return 0
    if tar_y == 0 and tar_x == 0 :
        return 0 
    return 1 

# neural network training 
def train_network() :

    global thetas1,thetas2,thetas3,network,HL_size,file_2ddl,file_3ddl,\
    training,load_model,learning,target_x,target_y,network,robot,step,moment,state_train,notification

    # Welcome message
    notif(notification)
    # if the target is not in the reachable zone 
    if target_check() == 0 : 
        # if the target is not in the reachable zone 
        notification = " You must choose a target in the reachable zone.\n The reachable zone \
        is defined by a radius of 1 from position\n  (0,0) for 2 DOF case and of 1.5 for 3 DOF case "
        notif(notification)
    else : 

        # choice of neural networks according to the type of robot 
        if int(articulation.get()) == 1 :
            if int(choise.get()) == 2 : 
                network = NN(5,HL_size,2)
            else : 
                network = NN(4,HL_size,2)
        else : 
            network = NN(2,HL_size,2)
        robot_2ddl = Robot_manipulator_2ddl()
        robot_3ddl = Robot_manipulator_3ddl()   
        training_2ddl = OnlineTrainer_2ddl(robot_2ddl,network)
        training_3ddl= OnlineTrainer_3ddl(robot_3ddl,network) 

        # choice of type of robot
        if int(choise.get()) == 1 : 
            robot = robot_2ddl
            training = training_2ddl
            file = file_2ddl
            training.articulation1_max = float(articulation1_max.get())
            training.articulation1_min = float(articulation1_min.get())
            training.articulation2_max = float(articulation2_max.get())
            training.articulation2_min = float(articulation2_min.get())
        else :
            robot = robot_3ddl
            training = training_3ddl
            file = file_3ddl
            training.articulation1_max = float(articulation1_max.get())
            training.articulation1_min = float(articulation1_min.get())
            training.articulation2_max = float(articulation2_max.get())
            training.articulation2_min = float(articulation2_min.get())
            training.articulation3_max = float(articulation3_max.get())
            training.articulation3_min = float(articulation3_min.get())

        # if articulations are taken into account
        if int(articulation.get()) == 1 :
            training.articulation = True

        training.pas = float(step.get())
        training.moment = float(moment.get())

        # use of previous networks 
        if int(load_model.get()) == 1:
            with open(file) as fp:
                json_obj = json.load(fp)
            for i in range(2):
                for j in range(HL_size):
                    network.wi[i][j] = json_obj["input_weights"][i][j]
            for i in range(HL_size):
                for j in range(2):
                    network.wo[i][j] = json_obj["output_weights"][i][j] 

        # intiate the target
        target = [target_x,target_y]

        # activate or not the learning 
        if int(learning.get()) == 1:
            training.training = True
        elif int(learning.get()) == 0:
            training.training = False  

        # do the training notify that it is over 
        if int(choise.get()) == 1 :
            thetas1,thetas2 = training.train(target)
            notification = "Training finished !\nThe last weights have been stored in last_w_2ddl.json\nYou \
                 can now animate the results." 
            notif(notification)
        else :
            thetas1,thetas2,thetas3 = training.train(target)
            notification = "Training finished !\nThe last weights have been stored in last_w_3ddl.json\nYou \
                can now animate the results." 
            notif(notification)
        
        # retrieval and recording of inputs values
        state_train = True
        json_obj = {"input_weights": network.wi, "output_weights": network.wo}
        with open(file, 'w') as fp:
            json.dump(json_obj, fp)
        print("The last weights have been stored in last_w.json")
        print("arrivé")
    


# Start animation 
def animate() : 
    global thetas1,thetas2,thetas3,target_x,target_y,robot,notification
    if int(save_animation.get()) == 1 : 
        robot.save = True
        robot.name_file = str(animation_name.get())
        notification="The animation was saved at the root of the project."
        notif(notification)
    tar = [float(target_x.get()),float(target_y.get())]
    Fig, ax = robot.draw_env(tar)
    if int(choise.get()) == 1 :
        rectangle = robot.draw_obstacle(Fig,ax)                
        line1, line2, pt1 = robot.draw_robot(Fig,ax)
        robot.train(thetas1,thetas2,line1,line2,pt1,Fig) 
    else :
        line1,line2,line3, pt1 = robot.draw_robot(Fig,ax)
        robot.train(thetas1,thetas2,thetas3,line1,line2,line3,pt1,Fig) 

# quit the similator       
def quit():
    root.destroy()

# Reset all the settings made before.
def Reset():
    global target_x,target_y,learning,load_model,notification
    target_x.set(0.00)
    target_y.set(0.00)
    learning.set(0)
    load_model.set(0)
    choise.set(1)
    hidden.set(20)
    notification="All settings have been reset.\n Start a new simulation !"
    notif(notification)

# draw the trajectory of the robot 
def plot_Trajectory():
    global robot,thetas1,thetas2,thetas3,target_x,target_y,state_train,notification
    tar = [float(target_x.get()),float(target_y.get())]
    if state_train : 
        if int(choise.get()) == 1 :
            robot.draw_Trajectory(thetas1,thetas2,tar)
        else :
            robot.draw_Trajectory(thetas1,thetas2,thetas3,tar)
    else : 
        notification = " You need to train the neural network first.\n Train the network, Start the animation,\n then you can view the graphs !"
        notif(notification)



# draw the gradient 
def plot_Gradient():
    global robot,training,notification
    if state_train : 
        if int(choise.get()) == 1 :
            robot.draw_grad_graph(training.grad0,training.grad1,training.t1)
        else :
            robot.draw_grad_graph(training.grad0,training.grad1,training.grad2,training.t)
    else : 
        notification = " You need to train the neural network first.\n Train the network, Start the animation,\n \
            then you can view the graphs ! "
        notif(notification)

# draw the velocity 
def plot_Velocity():
    global robot,training,notification
    if state_train : 
        robot.draw_velocity_graph(training.velocity0,training.velocity1,training.t2)
    else : 
        notification = " You need to train the neural network first.\n Train the network, Start the animation,\n \
            then you can view the graphs !  "
        notif(notification)

# draw the error
def plot_error():
    global robot,training,notification
    if state_train : 
        robot.draw_error_graph(training.error_x,training.error_y,training.t2)
    else : 
        notification = " You need to train the neural network first.\n Train the network, Start the animation,\n \
            then you can view the graphs !  "
        notif(notification)

def Tuto(): 

    webbrowser.open('https://sites.google.com/view/arm-robot-simulator/accueil')


# Buttons 
button1= ttk.Button(root, text='Train', width=18, command=train_network)
button1.place(x=23, y=470)

button2 = ttk.Button(root, text='Animate',width=18,command=animate)
button2.place(x=215, y=470)

button3 = ttk.Button(root, text='Quit',width=18,command = quit)
button3.place(x=600, y=470)

button4 = ttk.Button(root, text='Reset',width = 18, command=Reset)
button4.place(x=408, y=470)

button5 = ttk.Button(root,text='Tutoriel',width = 18, command=Tuto)
button5.place(x=792,y=470)

show_button1 = ttk.Button(Graph_frame, text=' Show ',width = 7, command=plot_Trajectory)
show_button1.place(x=270, y=2)

show_button2 = ttk.Button(Graph_frame, text=' Show ',width = 7, command=plot_Gradient)
show_button2.place(x=15, y=70)

show_button3 = ttk.Button(Graph_frame, text=' Show ',width = 7, command=plot_Velocity)
show_button3.place(x=140, y=70)

show_button4 = ttk.Button(Graph_frame, text=' Show ',width = 7, command=plot_error)
show_button4.place(x=265, y=70)

root.mainloop()