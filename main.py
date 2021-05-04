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
hl_size = tk.IntVar()
load_model.set(1)
learning.set(1)



# Network Settings zone
frame2 = ttk.LabelFrame(root, text=' Network Settings ', width=370, height=110)
frame2.place(x=23, y=45)
"""
# target_y text
Label1 = ttk.Label(root, text = 'Hidden Layer size')
Label1.place (x=50,y=80)


# Hidden Layer size value
boite1 = ttk.Spinbox(root,from_=5,to=100,increment=1,textvariable=hl_size,width=5)
boite1.place(x=300,y=78)
"""
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

#put an image 
photo = tk.PhotoImage(file='logominesnancy.png')
image_label = ttk.Label(
    root,
    image=photo,
    padding=5
)
image_label.place(x=415,y=52)

robot = Robot_manipulator()
HL_size= 25 #int(hl_size.get())
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
    thetas1,thetas2 = training.train(target)
    


def animate() : 
    global thetas1,thetas2,target_x,target_y
    tar = [float(target_x.get()),float(target_y.get())]
    Fig, ax = robot.draw_env(tar)
    line1, line2, pt1 = robot.draw_robot(Fig,ax)
    robot.train(thetas1,thetas2,line1,line2,pt1,Fig)    

def quit():
    root.destroy()

def Reset():
    global target_x,target_y,learning,load_model
    target_x.set(0.00)
    target_y.set(0.00)
    learning.set(0)
    load_model.set(0)

button1= ttk.Button(root, text='Train', width=18, command=train_network)
button1.place(x=23, y=470)

button2 = ttk.Button(root, text='Animate',width=18,command=animate)
button2.place(x=215, y=470)

button3 = ttk.Button(root, text='Quit',width=18,command = quit)
button3.place(x=600, y=470)

button4 = ttk.Button(root, text='Reset',width = 18, command=Reset)
button4.place(x=408, y=470)

"""json_obj = {"input_weights": network.wi, "output_weights": network.wo}
with open(file, 'w') as fp:
    json.dump(json_obj, fp)
print("The last weights have been stored in last_w.json")"""

root.mainloop()