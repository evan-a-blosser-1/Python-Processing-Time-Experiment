import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from scipy.optimize import fsolve
from icecream import ic
from scipy.linalg import inv 
import multiprocessing as mp
import time

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
########################################
################################### PATH
# MASCON I
Aster_M1CM_PATH  = 'Asteroid_Database/Asteroid_CM/MASCON1/'
Aster_VolM1_PATH = 'Asteroid_Database/Asteroid_CM/MASCON1/Tetra_Vol/'
# OBJ & Constant
Aster_OBJ_PATH   = 'Asteroid_Database/OBJ_Files/'
Aster_Const_PATH = 'Asteroid_Database/Asteroid_Constants/'
########################################
################################
# Big G in km^3/kg.s^2
G = 6.67430e-20
################################
##################
omega = 1

Contour_Size = 100
#################################################
####### Enter Asteroid Name & Update Here #######
Asteroid_Name = '1950DA_Prograde'
##################################################
########### Load File ############################
Aster_File_CM   = Aster_M1CM_PATH + Asteroid_Name + "_CM.dat"
Aster_File_OBJ  = Aster_OBJ_PATH  + Asteroid_Name + ".obj"

Aster_CM  = np.loadtxt(Aster_File_CM, delimiter=' ')


Aster_File_Const = Aster_Const_PATH + Asteroid_Name + '_const.in'
Asteroid_Const = pd.read_csv(Aster_File_Const, delimiter=' ')

Asteroid_Const.head()
#############################################################
Aster_Vol_File = Aster_VolM1_PATH + Asteroid_Name  + '_VolM1.csv'
Vol_Tetra = pd.read_csv(Aster_Vol_File, delimiter=' ')
scale = Asteroid_Const['Scaling'][0]
R_eff = Asteroid_Const['Mean Radius (km)'][0]
################################################# Non-dim
CM_MI = (Aster_CM*scale)/R_eff
Poly_CM_X = (Asteroid_Const['Poly CM X'][0]*scale)/R_eff
Poly_CM_Y = (Asteroid_Const['Poly CM Y'][0]*scale)/R_eff
Poly_CM_Z = (Asteroid_Const['Poly CM Z'][0]*scale)/R_eff
##################################################
##################################################
mu_I = []
for i in range(len(CM_MI)):
    mu = 1/len(CM_MI)
    mu_I.append(mu)
#################################################################
####################################### Equation Definition
def Poten_XY(x,y,CM_MI,mu_I,omega):
    U = np.zeros_like(x)
    for i in range(len(CM_MI)):
        R_x = x - CM_MI[i,0]
        R_y = y - CM_MI[i,1]
        R_z = 0 - CM_MI[i,2]
        R = np.sqrt(R_x**2 + R_y**2 + R_z**2)
        ##############################
        U  += - (mu_I[i])/R
    #######################
    return - (1/2)*(omega**2)*(x**2 + y**2) + U


def Poten_Check(x,CM_MI,mu_I,omega):
    U = 0
    for i in range(len(CM_MI)):
        R_x = x - CM_MI[i,0]
        R_y = 0
        R_z = 0 
        R = (R_x**2 + R_y**2 + R_z**2)**0.5
        U += - (mu_I[i])/R 
    return  U

#%% Processes
###############################################################
####################################################### Regular
def Regular_Process(Size):
    ################################################
    LowBound = -1.5
    UpBound  =  1.5
    x = np.linspace(LowBound,UpBound,Size)
    y = np.linspace(LowBound,UpBound,Size)
    X,Y = np.meshgrid(x,y)
    ############################## Begin
    t_0_non = time.time()
    #######
    PE_XY = Poten_XY(X,Y,CM_MI,mu_I,omega)
    ############################## End
    t_f_non = time.time()
    t_E_non = t_f_non - t_0_non
    #####  Save Results, Print execution time
    OutMessage1 = f"""
{'-'*42}
| Time Elapsed for Non-parallelized
| Python is:
|    
|       {t_E_non} seconds
|
{'-'*42}
    """
    print(OutMessage1)
    ##################################
    Contour_Levels = 1000
    ############################################################################
    figure = plt.figure(figsize=(10,10))
    Contu = plt.contour(X ,Y , PE_XY, Contour_Levels, cmap='viridis')
    cbar1 = figure.colorbar(Contu)
    cbar1.set_label('Gradient')
    cbar1.set_label(r'$DU^2/TU^2$', fontsize=14)
    plt.title("X-Y: Non-Parallelized Python")
    plt.xlabel(r'$X (DU)$', fontsize=14)
    plt.ylabel(r'$Y (DU)$', fontsize=14)
    # plt.show()
    return t_E_non


##########################################################################
############################################################## tensorflow
def Tensor_Process(Size):
    ################################################
    LowBound = -1.5
    UpBound  =  1.5
    x = tf.linspace(LowBound,UpBound,Size)
    y = tf.linspace(LowBound,UpBound,Size)
    X,Y = tf.meshgrid(x,y)
    ############################# Begin
    print("GPU Initialized: ", len(tf.config.list_physical_devices('GPU')))
    t_0_TF = time.time()
    #######
    with tf.device('/GPU:0'):
        PE_XY = Poten_XY(X,Y,CM_MI,mu_I,omega)
    ############################## End
    t_f_TF = time.time()
    t_E_TF = t_f_TF - t_0_TF
    #####  Save Results, Print execution time
    OutMessage1 = f"""
{'-'*42}
| Time Elapsed for tensorflow
| Python is:
|    
|       {t_E_TF} seconds
|
{'-'*42}
    """
    print(OutMessage1)
    ###################################
    PE_XY = PE_XY.numpy()
    ##################################
    Contour_Levels = 1000
    ############################################################################
    figure = plt.figure(figsize=(10,10))
    Contu = plt.contour(X ,Y , PE_XY, Contour_Levels, cmap='viridis')
    cbar1 = figure.colorbar(Contu)
    cbar1.set_label('Gradient')
    cbar1.set_label(r'$DU^2/TU^2$', fontsize=14)
    plt.title("X-Y: Tensorflow Python")
    plt.xlabel(r'$X (DU)$', fontsize=14)
    plt.ylabel(r'$Y (DU)$', fontsize=14)
    # plt.show()
    return t_E_TF

def Error_TF():
    ################################################
    LowBound = 0.0
    UpBound  =  1.5
    Size = 1000
    x_TF = tf.linspace(LowBound,UpBound,Size)
    ############################# Begin
    print("GPU Initialized: ", len(tf.config.list_physical_devices('GPU')))
    #######
    with tf.device('/GPU:0'):
        PE_TF = Poten_Check(x_TF,CM_MI,mu_I,omega)
    ############################## End
    #####  Save Results, Print execution time
    PE_TF  = PE_TF.numpy()
    x = np.linspace(LowBound,UpBound,Size)
    PE_Reg = Poten_Check(x,CM_MI,mu_I,omega)
    Error = ((PE_TF - PE_Reg)/PE_Reg)*100
    fig = plt.figure(figsize=(15,10))
    plt.plot(x,Error, label = 'TensorFlow Error',color = '#332288', linewidth = 2)
    plt.xlabel(r'$X (km)$', fontsize=25)
    plt.ylabel(r'$Error (\%)$', fontsize=25)
    plt.tick_params(axis='y', labelsize=20) 
    plt.tick_params(axis='x', labelsize=20)
    plt.legend(fontsize=20)
    Data_PATH = 'Databank/Conference/Fixed_Images/'
    # Save Data @ Path
    isExist = os.path.exists(Data_PATH)
    if not isExist:
        os.mkdir(Data_PATH)
    File_Name = Data_PATH +   'PRocess_Error' +'.eps'

    plt.savefig(File_Name)
    plt.show()



from multiprocessing import Pool, cpu_count

def Poten_XY_chunk(args):
    X_chunk, Y_chunk, CM_MI, mu_I, omega = args
    # Assuming Poten_XY is defined elsewhere to work on chunks of X and Y
    return Poten_XY(X_chunk, Y_chunk, CM_MI, mu_I, omega)

def worker_parallel(Size):
    LowBound = -1.5
    UpBound = 1.5
    x = np.linspace(LowBound, UpBound, Size)
    y = np.linspace(LowBound, UpBound, Size)
    X, Y = np.meshgrid(x, y)

    # Split X and Y into chunks for multiprocessing
    num_cpus = cpu_count()
    chunk_size = Size // num_cpus
    X_chunks = np.array_split(X, num_cpus)
    Y_chunks = np.array_split(Y, num_cpus)
    ############################# Begin
    t_0 = time.time()

    # Setup multiprocessing pool
    with Pool(num_cpus) as pool:
        # Map work to workers
        results = pool.map(Poten_XY_chunk, [(X_chunks[i], Y_chunks[i], CM_MI, mu_I, omega) for i in range(num_cpus)])

    # Combine results if necessary, assuming results is a list of arrays
    PE_XY = np.concatenate(results)
    ############################## End
    t_f = time.time()
    t_E_M = t_f - t_0
    ##################################
    Contour_Levels = 1000
    ############################################################################
    figure = plt.figure(figsize=(10,10))
    Contu = plt.contour(X ,Y , PE_XY, Contour_Levels, cmap='viridis')
    cbar1 = figure.colorbar(Contu)
    cbar1.set_label('Gradient')
    cbar1.set_label(r'$DU^2/TU^2$', fontsize=14)
    plt.title("X-Y: Multi-Processing in Python")
    plt.xlabel(r'$X (DU)$', fontsize=14)
    plt.ylabel(r'$Y (DU)$', fontsize=14)
    # plt.show()
    OutMessage2 = f"""
{'-'*42}
| Time Elapsed for Multiprocessing
|  in Python is:
|    
|       {t_E_M} seconds
|
{'-'*42}
"""
    print(OutMessage2)
    return t_E_M 

###########################################
##################################### Main
if __name__ == '__main__':
    #################
    Error_TF()
    Calc_Size = 10000
    for i in range(50):
        print(f"Run = {i}")   
        #################
        Time_Reg = Regular_Process(Size = Calc_Size)
        #################
        Time_TF  = Tensor_Process(Size = Calc_Size)
        #################
        Time_MP  = worker_parallel(Size = Calc_Size)
        ############################################
        Data_Path = 'Databank/Processing_Time_Test/'
        isExist = os.path.exists(Data_Path)
        if not isExist:
            os.mkdir(Data_Path)
        File_Name = "Run_Times_Size"+ str(Calc_Size) +".csv"
        File = Data_Path + File_Name
        data = {'Regular': [Time_Reg],
                'Tensorflow': [Time_TF],
                'Multiprocessing': [Time_MP]
                }
        df = pd.DataFrame(data)
        if os.path.exists(File):
            df.to_csv(File, mode='a', header=False)
        else:
            df.to_csv(File, mode='w', header=True)
