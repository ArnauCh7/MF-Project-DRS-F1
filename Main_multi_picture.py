'''
================================================================================================================
IMPORTANT!!! This main file is only ment to be executed when using two bodies/images,
if you want to do a single body plot execute "Main_single_picture.py" file.
It is also important to install PIL library, and numba library.
INSTRUCTIONS:
    - On windows search bar, write "cmd"
    - Once you're in the cmd whrite "pip install pillow"
    - When it's done write "pip install numba"
    - If both libraries are installed correctly you can execute the program
================================================================================================================
'''

#Import all the libraries needed for the code
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import math
from Functions import *
import numba
import tkinter as tk
from tkinter import simpledialog

'''
-----------------------------------------------------------------------------------------------------------------
Set the parameters and starting grids for the simulation
-----------------------------------------------------------------------------------------------------------------
'''
#Numerical Parameters
filename = "aleron.png"
image = load_image(filename)
filename2 = "drs.png"
image2 = load_image(filename2)
Lx  = image.size[0]-1 #We set the size at 1 less than the image becouse we start at 0
Ly = image.size[1]-1
m = 299
n = 299
dx = Lx/m
dy = Ly/n
Vo = 106 #m/s
density = 1.225 #air density at sea level
Po = 101325 #Pressure at sea level (101325 Pa) and at 2000m (79495.22 Pa)
iterations = 25000 #number of iterations when aproximating

#Grids
grids = create_grids(image)
V = grids[0]#V module grid
Vx = grids[1]#Velocity on X axis grid
Vy = grids[2]#Velocity on Y axis grid
P = grids[3]#Pressure grid
sf = grids[4]#Stream Function grid

'''
-----------------------------------------------------------------------------------------------------------------
Set boundary conditions for our grid and the surface of the object
-----------------------------------------------------------------------------------------------------------------
'''
bc = set_boundary_conditions(Vo, dy, dx, density, Po, Lx, Ly, sf, Vx, V, P) #boundary conditions vector of grids
sf = bc[0]
V = bc[1]
Vx = bc[2]
P = bc[3]

'''
-----------------------------------------------------------------------------------------------------------------
Fill stream function grid with a first gess of values that are within the range of the boundary values
-----------------------------------------------------------------------------------------------------------------
'''
sf = first_guess(Lx, Ly, sf)# We only apply first guess on the ´stream function becouse every other grid depends on this one

'''
-----------------------------------------------------------------------------------------------------------------
Fill the grid with Stream Function values except the ones that are inside the body of the picture
-----------------------------------------------------------------------------------------------------------------
'''
print("Computing Stream function, Vx and Vy ...")
pic1 = image_to_matrix(image)
pic2 = image_to_matrix(image2)
pic = merge_image_grids(pic1, pic2, Lx, Ly)
sfvxvygrids = fill_grid_with_body(pic, Lx, Ly, sf, Vx, Vy, dx, dy, iterations) #Sf, Vx and Vy final vector of grids
sf = sfvxvygrids[0]
Vx = sfvxvygrids[1]
Vy = sfvxvygrids[2]

'''
-----------------------------------------------------------------------------------------------------------------
Fill the Pressure grid with discretized values
-----------------------------------------------------------------------------------------------------------------
'''
print("Computing V module and Pressure ...")
vpgrids = fill_pressure_grid(pic, Lx, Ly, Vx, Vy, V, P, Po, Vo, density, iterations) #V module and P vector of grids
V = vpgrids[0]
P = vpgrids[1]
print("Ploting ...")
'''
-----------------------------------------------------------------------------------------------------------------
Plot the Stream Function, streamlines, pressure and velocity fields
-----------------------------------------------------------------------------------------------------------------
'''
plots(sf, Lx, Ly, Vx, Vy, P, V)