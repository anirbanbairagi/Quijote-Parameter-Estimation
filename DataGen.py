import numpy as np
import pandas as pd
from funcs import *

np.random.seed(0)
# Generating & Loading Train, Validation and Test set 

generate_df = input("Generate training sets of Density Fields (Y/N): ")
generate_pk = input("Generate training sets of Power Spectrums (Y/N): ")
                   

param_path=os.getcwd()+'/latin_hypercube/latin_hypercube_params.txt'
params_set=["Omega_m", "Omega_b", "h", "n_s", "sigma_8"]
df_path='/data74/chartier/3D_cubesDF/LH_data/df3D_LH/'
# '/data74/chartier/3D_cubesDF/fiducial/3D_df128_z0p000/'

data=DataLoader.load_LH_datasets(df_path,param_path)
# DataLoader.load_fid_datasets(df_path) 

length=len(data)

datafile='LH_data_128'
# 'Fid_data_128'
# 'Om_p'
# LH_data_128

if generate_pk == "Y":
    bins = input("Enter no. of bins = ")
    log_bins = input("log bins (Y/N): ")

    if log_bins=="Y":
        PK_path = "/data74/anirban/Quijote/"+datafile+"/PK/"+bins+"/log_bins/"

    else:    
        PK_path = "/data74/anirban/Quijote/"+datafile+"/PK/"+bins+"/normal_bins/"

    #################### ->->->->->->->->->->->->  ####################
    
    pk=DataLoader.load_LH_datasets(PK_path,param_path)
#     DataLoader.load_fid_datasets(PK_path)
#     DataLoader.load_LH_datasets(PK_path,param_path)
#     DataLoader.load_displaced_sims(PK_path, datafile)          
    
    length=len(pk)
    
    


train_ratio=0.75
val_ratio=0.15
test_ratio=0.10



permutation=list(np.random.permutation(length))


if generate_df=="Y":
    
    train_set=data.iloc[permutation[0:int(train_ratio*length)]].reset_index(drop=True)
    val_set=data.iloc[permutation[int(train_ratio*length):int((train_ratio+val_ratio)*length)]].reset_index(drop=True)
    test_set=data.iloc[permutation[int((train_ratio+val_ratio)*length):]].reset_index(drop=True)

#     fid_set=data.reset_index(drop=True)

    savepath='Dataset/'+datafile+'/'
    if not os.path.exists(savepath):
        os.mkdir(savepath)
        
    savepath=savepath+'df3D/'
    if not os.path.exists(savepath):
        os.mkdir(savepath)
        
    train_set.to_csv(savepath+'train.csv')
    val_set.to_csv(savepath+'val.csv')
    test_set.to_csv(savepath+'test.csv')

#     fid_set.to_csv(savepath+'fid_df.csv')

if generate_pk=="Y":
    
    train_pk=pk.iloc[permutation[0:int(train_ratio*length)]].reset_index(drop=True)
    val_pk=pk.iloc[permutation[int(train_ratio*length):int((train_ratio+val_ratio)*length)]].reset_index(drop=True)
    test_pk=pk.iloc[permutation[int((train_ratio+val_ratio)*length):]].reset_index(drop=True)
    
#     fid_pk=pk.reset_index(drop=True)
    
    savepath='Dataset/'+datafile+'/'
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    
    ##########  ->->->->->->->->
    savepath=savepath+'pk3D/'               
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    
    savepath=savepath+bins+'/'
    if not os.path.exists(savepath):
        os.mkdir(savepath)

 #     'Dataset/LH_data/pk3D_LH/'+bins+'/'
    
    
    if log_bins=="Y":
        savepath=savepath+'/log_bins/'
        if not os.path.exists(savepath):
            os.mkdir(savepath) 
    else:
        savepath=savepath+'/normal_bins/'
        if not os.path.exists(savepath):
            os.mkdir(savepath) 

    train_pk.to_csv(savepath+'train.csv')
    val_pk.to_csv(savepath+'val.csv')
    test_pk.to_csv(savepath+'test.csv')

#     fid_pk.to_csv(savepath+'fid_pk.csv')
