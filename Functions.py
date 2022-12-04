'''
================================================================================================================
IMPORTANT!!! This file is not ment to be executed, to execute the code use the files named
"Main_single_picture.py" for a single body plot and "Main_multi_picture.py" for a two body plot.
================================================================================================================
'''

#Import all the libraries needed for the code
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from PIL import Image
import math
import numba

'''
-----------------------------------------------------------------------------------------------------------------
Import the image and converting it into a numpy array using the PIL library 
-----------------------------------------------------------------------------------------------------------------
'''
def load_image(pic):
    img = Image.open(pic) #Open the image file
    return img

def image_to_matrix(img):
    img_matrix = np.array(img.convert('P', palette = Image.ADAPTIVE, colors = 2)) #Transform the image into a numpy array to work with a matrix
    return img_matrix

'''
-----------------------------------------------------------------------------------------------------------------
Create the grids for every function we will work with, these grids are the same size as the image (300 x 300)
and all the values are 0
-----------------------------------------------------------------------------------------------------------------
'''
def create_grids(image):
    
    Velocity_X = np.zeros([image.size[1], image.size[0]])
    Velocity_Y = np.zeros([image.size[1], image.size[0]])
    Velocity_Module = np.zeros([image.size[1], image.size[0]])
    Pressure = np.zeros([image.size[1], image.size[0]])
    StreamFunction = np.zeros([image.size[1], image.size[0]])
    
    
    return Velocity_Module, Velocity_X, Velocity_Y, Pressure, StreamFunction

'''
-----------------------------------------------------------------------------------------------------------------
Merge the first and second image grids to discretize at the same time
-----------------------------------------------------------------------------------------------------------------
'''
def merge_image_grids(pic1, pic2, Lx, Ly):
    i=1
    while i < Lx:
        j = 1
        while j < Ly:
            if pic2[j][i] == 1:
                pic1[j][i] = 2 #Set another value for the second body points so that we can adjust the boundary conditions
            j+=1
        i+=1
    
    return pic1

'''
-----------------------------------------------------------------------------------------------------------------
Set boundary conditions for our grids
-----------------------------------------------------------------------------------------------------------------
'''
@numba.jit(nopython = True)#Decorator to optimise the code calculations
def set_boundary_conditions(Vo, dy, dx, density, Po, Lx, Ly, Sf_grid, Vx_grid, Vmodule_grid, P_grid):
    #boundary condition on the x = 0, y = j * dy axis and  x = Lx, y = j * dy axis
    j = 0
    while j <= Ly:
        Sf_grid[j][0] = Vo * j * dy
        Sf_grid[j][Lx] = Vo * j * dy
        Vx_grid[j][0] = Vo
        Vx_grid[j][Lx] = Vo
        Vmodule_grid[j][0] = Vo
        Vmodule_grid[j][Lx] = Vo
        P_grid[j][0] = Po
        P_grid[j][Lx] = Po
        j +=1
    
    #boundary conditios on the x = i * dx, y = 0 axis and x = i * dx, y = Ly
    i = 0
    while i <= Lx:
        Sf_grid[Ly][i] = Vo * Ly
        Vx_grid[Ly][i] = Vo
        P_grid[0][i] = Po
        P_grid[Ly][i] = Po
        i +=1
    
    return Sf_grid, Vmodule_grid, Vx_grid, P_grid
'''
-----------------------------------------------------------------------------------------------------------------
First Guess: we do a first guess for the values so that we can start iterating and aproximating them to the
real ones with the discretized functions
-----------------------------------------------------------------------------------------------------------------
'''
@numba.jit(nopython = True)
def first_guess(Lx, Ly, sf):
    i=1
    while i < Lx:
        j = 1
        while j < Ly:
            sf[j][i] = (sf[0][i]+sf[Ly][i])/2 #Set the values for all the points except the ones on the boundary as de mean value between the top and bottom boundary value
            j +=1
        i+=1
    return sf

'''
-----------------------------------------------------------------------------------------------------------------
Function to calculate the next grid value with the discretized Stream Function
-----------------------------------------------------------------------------------------------------------------
'''
@numba.jit(nopython = True)
def streamfunction_discretization(funder, fupper, fleft, fright, dx, dy):
    fij = ((fright + fleft)*(dy**2) + (fupper + funder)*(dx**2))/(2*((dx**2)+(dy**2)))
    return fij    
'''
-----------------------------------------------------------------------------------------------------------------
Function to calculate the next grid value with the discretized Vx
-----------------------------------------------------------------------------------------------------------------
'''
@numba.jit(nopython = True)
def Vx_discretization(funder, fupper, dy):
    Vx = (fupper - funder)/(2*dy)
    return Vx
'''
-----------------------------------------------------------------------------------------------------------------
Function to calculate the next grid value with the discretized Vy
-----------------------------------------------------------------------------------------------------------------
'''
@numba.jit(nopython = True)
def Vy_discretization(fright, fleft, dx):
    Vy = (fleft - fright)/(2*dx)
    return Vy

'''
-----------------------------------------------------------------------------------------------------------------
Fill the grid with Stream Function values except the ones that are inside the body of the picture
-----------------------------------------------------------------------------------------------------------------
'''
@numba.jit(nopython = True)
def fill_grid_with_body(pic, Lx, Ly, sf, Vx, Vy, dx, dy, iterations):
    counter = 0 #Number of iterations
    boundary_points_counter_1 = 0 #Counts the number of points on the boundary of the first object
    boundary_points_counter_2 = 0 #Counts the number of points on the boundary of the second object
    object_boundary_value_1 = 0 #Set value of the boundary condition as the value of the stream function on the first pint in the boundary of the first object
    object_boundary_value_2 = 0 #Set value of the boundary condition as the value of the stream function on the first pint in the boundary of the second object
    while counter < iterations: #the more iterations we do, the closest it will be to the real value
        i = 1 #set i and j to one becouse we don't want to change the boundary conditions
        j = 1
        while i < Lx:
            while j < Ly:
                if pic[j][i] == 1:#Inside the body, set the values as nan
                    sf[j][i] = np.nan
                    Vx[j][i] = np.nan
                    Vy[j][i] = np.nan
                elif pic[j][i] == 2:
                    sf[j][i] = np.nan
                    Vx[j][i] = np.nan
                    Vy[j][i] = np.nan
                elif pic[j+1][i] == 1 or pic[j-1][i] == 1 or pic[j][i+1] == 1 or pic[j][i-1] == 1:#Contour of the body, set a constant value
                    boundary_points_counter_1 +=1
                    if boundary_points_counter_1 == 1:
                        object_boundary_value_1 = (sf[Ly][0] * (j/Ly))
                        sf[j][i] = object_boundary_value_1
                    else:
                        sf[j][i] = object_boundary_value_1
                elif pic[j+1][i] == 2 or pic[j-1][i] == 2 or pic[j][i+1] == 2 or pic[j][i-1] == 2:
                    boundary_points_counter_2 +=1
                    if boundary_points_counter_2 == 1:
                        object_boundary_value_2 = (sf[Ly][0] * (j/Ly))
                        sf[j][i] = object_boundary_value_2
                    else:
                        sf[j][i] = object_boundary_value_2
                else:#Outside the body
                    sf[j][i] = streamfunction_discretization(sf[j-1][i], sf[j+1][i], sf[j][i-1], sf[j][i+1], dx, dy)
                    Vx[j][i] = Vx_discretization(sf[j-1][i], sf[j+1][i], dy)
                    Vy[j][i] = Vy_discretization(sf[j][i+1], sf[j][i-1], dy)
                j +=1
            i +=1
            j = 1
        counter +=1
        progress = (counter/iterations)*100
        if progress == 10 or progress == 20 or progress == 30 or progress == 40 or progress == 50 or progress == 60 or progress == 70 or progress == 80 or progress == 90 or progress == 100: #Show the progress in percentage
            print(int(progress), "%")
    return sf, Vx, Vy

'''
-----------------------------------------------------------------------------------------------------------------
calculate the velocity module
-----------------------------------------------------------------------------------------------------------------
'''
@numba.jit(nopython = True)
def velocity_module(Vx, Vy):
    V = math.sqrt((Vx**2) + (Vy**2))#Calculation of the velocity module with the Vx and Vy components
    return V

'''
-----------------------------------------------------------------------------------------------------------------
calculate pressure
-----------------------------------------------------------------------------------------------------------------
'''
@numba.jit(nopython = True)
def pressure(V, Vo, density, Po):
    P = Po + (1/2) * density * ((Vo**2) - (V**2))
    return P

'''
-----------------------------------------------------------------------------------------------------------------
Fill the pressire grid with discretized values
-----------------------------------------------------------------------------------------------------------------
'''
@numba.jit(nopython = True)
def fill_pressure_grid(pic, Lx, Ly, Vx, Vy, V, P, Po, Vo, density, iterations):
    counter = 0 #Number of iterations
    while counter < iterations:#the more iterations we do, the closest it will be to the real value
        i = 1
        j = 1
        while i < Lx:
            while j < Ly:
                if pic[j][i] == 1:#Inside the body, set the values as nan
                    V[j][i] = np.nan
                    P[j][i] = np.nan
                elif pic[j][i] == 2:
                    V[j][i] = np.nan
                    P[j][i] = np.nan
                elif pic[j+1][i] == 1 or pic[j-1][i] == 1 or pic[j][i+1] == 1 or pic[j][i-1] == 1: #Contour of the body, set a constant value
                    V[j][i] = Vo
                    P[j][i] = Po
                elif pic[j+1][i] == 2 or pic[j-1][i] == 2 or pic[j][i+1] == 2 or pic[j][i-1] == 2:
                    V[j][i] = Vo
                    P[j][i] = Po
                else:
                    V[j][i] = velocity_module(Vx[j][i], Vy[j][i])
                    P[j][i] = pressure(V[j][i], Vo, density, Po)
                j +=1
            i +=1
            j = 1
        counter +=1
        progress = (counter/iterations)*100
        if progress == 10 or progress == 20 or progress == 30 or progress == 40 or progress == 50 or progress == 60 or progress == 70 or progress == 80 or progress == 90 or progress == 100:
            print(int(progress), "%")
    return V, P

'''
-----------------------------------------------------------------------------------------------------------------
Ploting the streamfunction, streamlines, the velocity and pressure fields
-----------------------------------------------------------------------------------------------------------------
'''
def plots(sf, Lx, Ly, Vx, Vy, P, V):
    x_coord = np.linspace(0, 300, 300)#Create a 2d space to plot
    y_coord = np.linspace(0, 300, 300)
    
    plt.figure(1)
    plt.xlabel("X (cm)")
    plt.ylabel("Y (cm)")
    plt.title("Stream Function")
    plt.contourf(x_coord, y_coord, sf, 35 )
    plt.colorbar()
    plt.contour(x_coord, y_coord, sf, 35, colors = 'black', linewidths=(0.4), linestyles = 'solid' )
    
    plt.figure(2)
    plt.xlabel("X (cm)")
    plt.ylabel("Y (cm)")
    plt.title("Streamlines")
    plt.streamplot(x_coord, y_coord, Vx, Vy, density=3, color = 'black', linewidth = 0.25 )
    
    plt.figure(3)
    plt.xlabel("X (cm)")
    plt.ylabel("Y (cm)")
    plt.title("Pressure (Pa)")
    plt.contourf(x_coord, y_coord, P, 35, cmap = 'jet')
    plt.colorbar()
    
    plt.figure(4)
    plt.xlabel("X (cm)")
    plt.ylabel("Y (cm)")
    plt.title("Velocity (m/s)")
    plt.contourf(x_coord, y_coord, V, 35)
    plt.colorbar()
    
    plt.show()