import numpy as np
import pandas as pd
import os
from funcs import *

bins = input("Enter no. of bins = ")
bins = eval(bins)
log_bins = input("log_bins (Y/N): ")

df3D_path = '/data74/chartier/3D_cubesDF/LH_data/df3D_LH/'
# '/data74/chartier/3D_cubesDF/fiducial/3D_df128_z0p000/'
# '/data74/chartier/3D_cubesDF/s8_m/3D_df128_z0p000/'


PK_path = "/data74/anirban/Quijote/LH_data_128/PK/"
# "/data74/anirban/Quijote/Fid_data_128/PK/"



PK_path = PK_path+str(bins)+'/'
if not os.path.exists(PK_path):
    os.mkdir(PK_path)

if log_bins == "Y":
    PK_path = PK_path+'log_bins/'
    if not os.path.exists(PK_path):
        os.mkdir(PK_path)
    
    
else:
    PK_path = PK_path+'normal_bins/'
    if not os.path.exists(PK_path):
        os.mkdir(PK_path)
    

Density_Field = DataLoader.load_df(df3D_path)

for i, delta in enumerate(Density_Field):

    if not os.path.exists(PK_path+str(i)+'/'):
        os.mkdir(PK_path+str(i)+'/')
    
    if log_bins == "Y":
        power=Pk(np.load(delta), bins=128, log_bins=True)
        np.save(PK_path+str(i)+'/'+"PK_"+str(bins)+"_log_bins.npy", power)
    else:
        power=Pk(np.load(delta), bins=128, log_bins=False)
        np.save(PK_path+str(i)+'/'+"PK_"+str(bins)+".npy", power)
        
    

