import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, glob
import math
import torch
from torch import nn
from torchvision import models, transforms, datasets
# from train import *
# import Pk_library as PKL

import powerbox as pbox
from powerbox import get_power
# np.random.seed(0)

displaced_param_dict = {'Om_p':[0.3275,0.049,0.6711,0.9624,0.834],
                 'Om_m':[0.3075,0.049,0.6711,0.9624,0.834],
                 'Ob2_p':[0.3175,0.050,0.6711,0.9624,0.834],
                 'Ob2_m':[0.3175,0.048,0.6711,0.9624,0.834],
                 'h_p':[0.3175,0.049,0.6911,0.9624,0.834],
                 'h_m':[0.3175,0.049,0.6511,0.9624,0.834],
                 'ns_p':[0.3175,0.049,0.6711,0.9824,0.834],
                 'ns_m':[0.3175,0.049,0.6711,0.9424,0.834], 
                 's8_p':[0.3175,0.049,0.6711,0.9624,0.849],
                 's8_m':[0.3175,0.049,0.6711,0.9624,0.819]}
    

class DataLoader:
    def load_df(path):
        length=len(glob.glob(path+'*'))
        image_files=[]
        for i in np.arange(length):
            image_files.append([os.path.join(path,str(i),file)
                                for file in os.listdir(path+str(i)) if file.endswith('.npy')][0]) 
        return image_files

    def load_param(path):
        param=open(path,'r+')
        param_list=[]
        for i, line in enumerate(param.readlines()):
            if i!=0:
                Omega_m, Omega_b, h, n_s, sigma_8=map(eval,line.split(' '))  #change when required
                param_list.append([Omega_m, Omega_b, h, n_s, sigma_8])       # 

        return param_list

    def load_LH_datasets(path_of_df,path_of_param):
        dataframe=pd.DataFrame(columns=["Density_Field", "Omega_m", "Omega_b", "h", "n_s", "sigma_8"])
        dataframe['Density_Field']=DataLoader.load_df(path_of_df)
        dataframe[["Omega_m", "Omega_b", "h", "n_s", "sigma_8"]]=DataLoader.load_param(path_of_param)

        return dataframe

    def load_displaced_sims(path_of_df, datafile):
        dataframe=pd.DataFrame(columns=["Density_Field", "Omega_m", "Omega_b", "h", "n_s", "sigma_8"])
        dataframe['Density_Field']=DataLoader.load_df(path_of_df)
        
        dataframe[["Omega_m", "Omega_b", "h", "n_s", "sigma_8"]]=displaced_param_dict[datafile]
        
        return dataframe


    def load_fid_datasets(path_of_df):
        dataframe=pd.DataFrame(columns=["Density_Field", "Omega_m", "Omega_b", "h", "n_s", "sigma_8"])
        dataframe['Density_Field']=DataLoader.load_df(path_of_df)
        dataframe[["Omega_m", "Omega_b", "h", "n_s", "sigma_8"]]=[0.3175,0.049,0.6711,0.9624,0.834]

        return dataframe
    
#     def load_LH_pk(path_of_df,path_of_param):
#         dataframe=pd.DataFrame(columns=["Power_Spectra", "Omega_m", "Omega_b", "h", "n_s", "sigma_8"])
#         dataframe['Power_Spectra']=DataLoader.load_df(path_of_df)
#         dataframe[["Omega_m", "Omega_b", "h", "n_s", "sigma_8"]]=DataLoader.load_param(path_of_param)

#         return dataframe


def Pk(delta, bins=128, log_bins=False):
    p_k_field, bins_field = get_power(delta, boxlength=1000, bins=bins, bin_ave=False, log_bins=log_bins)
    
#     bins_field=bins_field[~np.isnan(p_k_field)]
#     p_k_field= p_k_field[~np.isnan(p_k_field)]
    
    p_k_field[np.isnan(p_k_field)]=0
    
    # compute power spectrum
#     Pk = PKL.Pk(delta=delta, BoxSize=1000, axis=0, MAS='CIC', threads=0, verbose=False)

    # Pk is a python class containing the 1D, 2D and 3D power spectra, that can be retrieved as

    # 1D P(k)
#     k1D      = Pk.k1D
#     Pk1D     = Pk.Pk1D
#     Nmodes1D = Pk.Nmodes1D

#     # 2D P(k)
#     kpar     = Pk.kpar
#     kper     = Pk.kper
#     Pk2D     = Pk.Pk2D
#     Nmodes2D = Pk.Nmodes2D

#     # 3D P(k)
#     k       = Pk.k3D
#     Pk0     = Pk.Pk[:,0] #monopole
#     Pk2     = Pk.Pk[:,1] #quadrupole
#     Pk4     = Pk.Pk[:,2] #hexadecapole
#     Pkphase = Pk.Pkphase #power spectrum of the phases
#     Nmodes  = Pk.Nmodes3D
    
    return p_k_field




    
def shuffled(X,Y):
    np.random.seed(1)
    m = X.shape[0]                  # number of training examples
    
    permutation = list(np.random.permutation(m))
    shuffled_X = X.iloc[permutation].reset_index(drop=True)
    shuffled_Y = Y.iloc[permutation].reset_index(drop=True)
    
    return shuffled_X, shuffled_Y

def random_mini_batches(X, Y, batch_size = 64):
    
    m = X.shape[0]                  # number of training examples
    mini_batches = []
        
    # Step 1: Shuffle (X, Y)
    shuffled_X, shuffled_Y = shuffled(X, Y)
    
    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):

        mini_batch_X = shuffled_X[k*batch_size:(k+1)*batch_size]
        mini_batch_Y = shuffled_Y[k*batch_size:(k+1)*batch_size]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % batch_size != 0:

        mini_batch_X = shuffled_X[num_complete_minibatches*batch_size:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches*batch_size:]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

    
    
    
class transform:
    
    def horizontal_flip(data):
        return  transforms.Compose([transforms.ToTensor(),
                                    transforms.RandomHorizontalFlip(1)])(data)
    
    def vertical_flip(data):
        return transforms.Compose([transforms.ToTensor(),
                                   transforms.RandomVerticalFlip(1)])(data)
    
    def double_flip(data):
        return transforms.Compose([transforms.ToTensor(),
                                   transforms.RandomHorizontalFlip(1),
                                   transforms.RandomVerticalFlip(1)])(data)
    
    def rotate_90(data):
        return transforms.Compose([transforms.ToTensor(),
                                   transforms.RandomRotation(90)])(data)
    
    def rotate_180(data):
        return transforms.Compose([transforms.ToTensor(),
                                   transforms.RandomRotation(180)])(data)
    
    def rotate_270(data):
        return transforms.Compose([transforms.ToTensor(),
                                   transforms.RandomRotation(270)])(data)
        
        
        

dim= (128,128,128)

class preprocess:
    
    def process(data, label):    
        img = [np.load(i).reshape(*dim,1) for i in data]             
        img=np.moveaxis(img,-1,1)
        img = torch.Tensor(img)
        img = transforms.Normalize(0, 1.0476638)(img)
        label = torch.tensor(label.values)

        return img, label

    def process2(data, label):
        img = [(np.load(i)+1).reshape(*dim,1) for i in data] 
        img=np.moveaxis(img,-1,1)
        img = torch.Tensor(img)
        img = transforms.Normalize(0, 1.0476638)(img)
        
        label = torch.tensor(label.values)
#         label = (label- torch.mean(label,axis=0))/torch.std(label,axis=0)
        
        return img, label 


    def process2_horizon(data, label):
        img = [transform.horizontal_flip(np.load(i)+1).numpy().reshape(*dim,1) for i in data] 
        img=np.moveaxis(img,-1,1)
        img = torch.Tensor(img)
        
        label = torch.tensor(label.values)
        
        return img, label 
    
    def process2_vert(data, label):
        img = [transform.vertical_flip(np.load(i)+1).numpy().reshape(*dim,1) for i in data] 
        img=np.moveaxis(img,-1,1)
        img = torch.Tensor(img)
        
        label = torch.tensor(label.values)
        
        return img, label 
    
    def process2_bothflip(data, label):
        img = [transform.double_flip(np.load(i)+1).numpy().reshape(*dim,1) for i in data] 
        img=np.moveaxis(img,-1,1)
        img = torch.Tensor(img)
        
        label = torch.tensor(label.values)
        
        return img, label 
    
    def process2_rot90(data, label):
        img = [transform.rotate_90(np.load(i)+1).numpy().reshape(*dim,1) for i in data] 
        img=np.moveaxis(img,-1,1)
        img = torch.Tensor(img)
        
        label = torch.tensor(label.values)
        
        return img, label 
    
    def process2_rot180(data, label):
        img = [transform.rotate_180(np.load(i)+1).numpy().reshape(*dim,1) for i in data] 
        img=np.moveaxis(img,-1,1)
        img = torch.Tensor(img)
        
        label = torch.tensor(label.values)
        
        return img, label 
    
    def process2_rot270(data, label):
        img = [transform.rotate_270(np.load(i)+1).numpy().reshape(*dim,1) for i in data] 
        img=np.moveaxis(img,-1,1)
        img = torch.Tensor(img)
        
        label = torch.tensor(label.values)
        
        return img, label 
    
#     def process_pk(data, label):
#         img = [Pk(np.load(i)) for i in data]             
#         img=np.array(img)
#         img = torch.Tensor(img)
# #         img = transforms.Normalize(0, 1.0476638)(img)
#         label = torch.tensor(label.values)

        return img, label
    
    def PK_normal(data, label):
        img = [np.load(i) for i in data]             
#         img=np.array(img)
        img = torch.Tensor(img)
        mean=torch.tensor([1.78166005e-09, 1.98067278e+04, 2.43946198e+04, 2.45141946e+04,
       2.25702557e+04, 1.98174466e+04, 1.71475038e+04, 1.49731852e+04,
       1.32142648e+04, 1.18677144e+04, 1.09062465e+04, 1.01853614e+04,
       9.46566208e+03, 8.70123310e+03, 7.93569254e+03, 7.14648631e+03,
       6.38195759e+03, 5.71144242e+03, 5.15618634e+03, 4.72861396e+03,
       4.40694458e+03, 4.15256253e+03, 3.92245999e+03, 3.70381122e+03,
       3.47662853e+03, 3.24110184e+03, 3.00505088e+03, 2.78420225e+03,
       2.57999964e+03, 2.40613448e+03, 2.25385900e+03, 2.11941869e+03,
       1.99924672e+03, 1.88948926e+03, 1.78735449e+03, 1.68998175e+03,
       1.59503049e+03, 1.50399090e+03, 1.41753248e+03, 1.33650417e+03,
       1.25935151e+03, 1.18837876e+03, 1.12207469e+03, 1.06203450e+03,
       1.00581508e+03, 9.52228073e+02, 9.02676013e+02, 8.55287922e+02,
       8.10272783e+02, 7.67072986e+02, 7.25696578e+02, 6.87332692e+02,
       6.51298329e+02, 6.17550974e+02, 5.85325652e+02, 5.54960248e+02,
       5.26622949e+02, 4.99753213e+02, 4.74284879e+02, 4.50508691e+02,
       4.27894483e+02, 4.07127431e+02, 3.87178441e+02, 3.68442563e+02,
       3.50776933e+02, 3.34326569e+02, 3.18459107e+02, 3.04458772e+02,
       2.91098882e+02, 2.78628165e+02, 2.67012157e+02, 2.56416245e+02,
       2.46098911e+02, 2.36970009e+02, 2.26343641e+02, 2.14057400e+02,
       2.02177411e+02, 1.91590594e+02, 1.81308821e+02, 1.71935005e+02,
       1.62212116e+02, 1.53793506e+02, 1.45286985e+02, 1.38047221e+02,
       1.30283556e+02, 1.23724478e+02, 1.17049158e+02, 1.11218669e+02,
       1.05508371e+02, 1.00363279e+02, 9.49871692e+01, 9.09026550e+01,
       8.59282954e+01, 8.22208036e+01, 7.82402741e+01, 7.48634845e+01,
       7.11111523e+01, 6.79509793e+01, 6.51372439e+01, 6.22643111e+01,
       5.89704143e+01, 5.63459793e+01, 5.37530624e+01, 5.09474453e+01,
       4.72546084e+01, 4.46717444e+01, 4.12512798e+01, 3.99969096e+01,
       3.70918221e+01, 3.55412157e+01, 3.34809871e+01, 3.21579643e+01,
       3.03492220e+01, 2.94195115e+01, 2.76915414e+01, 2.70424736e+01,
       2.54898377e+01, 2.51177443e+01, 2.38274882e+01, 2.34707109e+01,
       2.26017945e+01, 2.21905270e+01, 2.16882822e+01, 2.12947824e+01,
       2.09737473e+01, 2.08330737e+01, 2.06852950e+01, 2.06819410e+01])
        
        std=torch.tensor([1.55014689e-09, 6.65141135e+03, 4.33856025e+03, 3.61622275e+03,
       2.42443914e+03, 1.82515467e+03, 1.33412983e+03, 1.01646632e+03,
       7.81913136e+02, 5.83859947e+02, 5.07198743e+02, 4.51664274e+02,
       3.66634068e+02, 3.23067456e+02, 2.84698564e+02, 2.27999400e+02,
       1.92471804e+02, 1.63159791e+02, 1.37274045e+02, 1.21638071e+02,
       1.09754977e+02, 9.89667267e+01, 8.71121507e+01, 8.06438435e+01,
       7.13472881e+01, 6.41404953e+01, 5.80312773e+01, 5.29840962e+01,
       4.70672927e+01, 4.37470242e+01, 3.86220992e+01, 3.44932595e+01,
       3.27045666e+01, 2.98989916e+01, 2.73642737e+01, 2.54694249e+01,
       2.32565063e+01, 2.13778835e+01, 2.01695192e+01, 1.83692862e+01,
       1.69258198e+01, 1.56348799e+01, 1.43801262e+01, 1.38505936e+01,
       1.27982569e+01, 1.20263173e+01, 1.13259283e+01, 1.04692565e+01,
       9.84426296e+00, 9.19826945e+00, 8.42925829e+00, 8.11258745e+00,
       7.48622487e+00, 7.06021325e+00, 6.64488093e+00, 6.26506380e+00,
       5.94530125e+00, 5.58407602e+00, 5.17219011e+00, 4.87312940e+00,
       4.58320441e+00, 4.43931509e+00, 4.18246452e+00, 3.87929761e+00,
       3.82879648e+00, 3.57861865e+00, 3.41225868e+00, 3.30439522e+00,
       3.15384919e+00, 2.95823497e+00, 2.85453546e+00, 2.75504952e+00,
       2.62362691e+00, 2.54765597e+00, 2.43674715e+00, 2.32001092e+00,
       2.17350467e+00, 2.14183404e+00, 2.00564460e+00, 1.95717155e+00,
       1.86370626e+00, 1.80314257e+00, 1.67596464e+00, 1.63471701e+00,
       1.55751387e+00, 1.50345191e+00, 1.44722428e+00, 1.43186260e+00,
       1.34050101e+00, 1.31190968e+00, 1.26620603e+00, 1.24869644e+00,
       1.18329398e+00, 1.14578299e+00, 1.12161138e+00, 1.08890700e+00,
       1.08789049e+00, 1.02649470e+00, 1.01230601e+00, 9.94902989e-01,
       9.76628631e-01, 9.62307873e-01, 9.42980282e-01, 9.43022027e-01,
       9.01593355e-01, 8.87895442e-01, 8.72958463e-01, 8.82595062e-01,
       8.57128597e-01, 8.70023691e-01, 8.36421131e-01, 8.83699330e-01,
       8.57818312e-01, 9.03495114e-01, 8.50514293e-01, 9.61938173e-01,
       9.28204933e-01, 1.04994559e+00, 9.93910873e-01, 1.24765081e+00,
       1.10523422e+00, 1.58700706e+00, 1.39272186e+00, 2.18212168e+00,
       1.96500515e+00, 3.65911895e+00, 3.95513428e+00, 1.18531404e+01]
                        )

        img= (img-mean)/std
        label = torch.tensor(label.values)

        return img, label

    
    def PK_log(data, label):
        img = [np.load(i).reshape(1,-1) for i in data]         #  .reshape(1,-1) for inception
        img=np.array(img)
        img = torch.Tensor(img)
        mean = torch.tensor([1.75549929e+04, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 2.09325953e+04, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 2.28837403e+04, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 2.40958906e+04, 0.00000000e+00,
       0.00000000e+00, 2.45922233e+04, 0.00000000e+00, 0.00000000e+00,
       2.47753251e+04, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       2.49868591e+04, 2.47059229e+04, 0.00000000e+00, 2.44013791e+04,
       2.41510177e+04, 2.38583959e+04, 2.36216703e+04, 2.31238651e+04,
       0.00000000e+00, 2.20709823e+04, 2.20774625e+04, 2.15851833e+04,
       2.10721295e+04, 2.05512355e+04, 2.00923913e+04, 1.92546807e+04,
       1.86949274e+04, 1.79092376e+04, 1.76000981e+04, 1.67563196e+04,
       1.61901500e+04, 1.57614633e+04, 1.50481375e+04, 1.45076750e+04,
       1.40131311e+04, 1.34780489e+04, 1.29890201e+04, 1.25621540e+04,
       1.21230640e+04, 1.18144250e+04, 1.14374238e+04, 1.11432366e+04,
       1.08493249e+04, 1.05655681e+04, 1.02168574e+04, 9.89255298e+03,
       9.56228849e+03, 9.23659663e+03, 8.86967685e+03, 8.47453727e+03,
       8.10907086e+03, 7.61571395e+03, 7.15442727e+03, 6.71122781e+03,
       6.28943042e+03, 5.84795202e+03, 5.48092368e+03, 5.14714478e+03,
       4.85152106e+03, 4.58069601e+03, 4.34430685e+03, 4.15242589e+03,
       3.96849970e+03, 3.77871442e+03, 3.58923360e+03, 3.38637783e+03,
       3.16538999e+03, 2.93804966e+03, 2.72102088e+03, 2.51977039e+03,
       2.33694173e+03, 2.16987813e+03, 2.02625284e+03, 1.89507126e+03,
       1.76874086e+03, 1.64483883e+03, 1.51990314e+03, 1.39829185e+03,
       1.28213290e+03, 1.17499101e+03, 1.07634260e+03, 9.84997763e+02,
       8.99143410e+02, 8.18369704e+02, 7.40801124e+02, 6.69289727e+02,
       6.02928804e+02, 5.41465074e+02, 4.85041931e+02, 4.33951236e+02,
       3.87445869e+02, 3.45375386e+02, 3.08025120e+02, 2.75263613e+02,
       2.47221558e+02, 2.18936395e+02, 1.87542493e+02, 1.59579043e+02,
       1.35287479e+02, 1.14367639e+02, 9.67500486e+01, 8.18438496e+01,
       6.93894012e+01, 5.87296235e+01, 4.77155879e+01, 3.77603086e+01,
       3.08086919e+01, 2.57318379e+01, 2.25743112e+01, 2.10143945e+01])
        
        std = torch.tensor([1.03309628e+04, 1, 1, 1,
       1, 1, 1, 1,
       1, 8.52497639e+03, 1, 1,
       1, 1, 1.14068311e+04, 1,
       1, 1, 1.38743381e+04, 1,
       1, 7.05715914e+03, 1, 1,
       7.19298547e+03, 1, 1, 1,
       1.01920776e+04, 6.36771578e+03, 1, 6.98498242e+03,
       6.97161192e+03, 1.19196545e+04, 6.79664608e+03, 4.76078662e+03,
       1, 1.27808053e+04, 4.46789717e+03, 5.06618874e+03,
       4.29754430e+03, 4.21210229e+03, 5.78905581e+03, 3.72650434e+03,
       2.62384014e+03, 2.99322469e+03, 3.64310626e+03, 2.28210467e+03,
       2.58648379e+03, 2.28152759e+03, 1.63585028e+03, 1.88840500e+03,
       1.88738733e+03, 1.26263776e+03, 1.33370956e+03, 1.36210726e+03,
       1.00253507e+03, 1.12940666e+03, 9.67721703e+02, 8.75512104e+02,
       8.58053673e+02, 7.11081214e+02, 6.49494199e+02, 5.84839572e+02,
       6.01325138e+02, 5.43031675e+02, 4.40303968e+02, 4.67041314e+02,
       3.89688598e+02, 3.35877715e+02, 2.97577614e+02, 2.84711431e+02,
       2.34025606e+02, 2.09020363e+02, 1.96723763e+02, 1.68698013e+02,
       1.53063283e+02, 1.26828047e+02, 1.21012251e+02, 1.14706700e+02,
       9.53103862e+01, 9.02735073e+01, 8.30233239e+01, 7.18870699e+01,
       6.28223764e+01, 5.62964752e+01, 5.03022725e+01, 4.50876524e+01,
       3.88866780e+01, 3.31829425e+01, 3.14244506e+01, 2.75192427e+01,
       2.46789475e+01, 2.16695007e+01, 1.91359281e+01, 1.70333514e+01,
       1.48458078e+01, 1.30560010e+01, 1.18007835e+01, 1.04555428e+01,
       9.36914689e+00, 8.12875177e+00, 7.25128209e+00, 6.40181092e+00,
       5.61519965e+00, 5.00810385e+00, 4.37662251e+00, 3.89553510e+00,
       3.42418969e+00, 3.14473654e+00, 2.80998508e+00, 2.51926642e+00,
       2.27849455e+00, 2.04722503e+00, 1.77961778e+00, 1.58349671e+00,
       1.37505405e+00, 1.23995015e+00, 1.09944723e+00, 9.79095366e-01,
       8.83294489e-01, 8.03780977e-01, 7.16631145e-01, 6.55534024e-01,
       6.26792539e-01, 6.30187990e-01, 7.67587663e-01, 1.46375295e+00])
        
        img= (img-mean)/std
        label = torch.tensor(label.values)

        
        return img, label 
    
    
def comparison_plot(data,datafile, parameters, method=preprocess.PK_log, save_path="", plot=True,  save_fig=True):
    criterion = nn.MSELoss()

    # Load the graph with the trained states
    model_path=save_path+'model.pth'
    model=torch.load(model_path)['model']
    model.load_state_dict(torch.load(model_path)['state_dict'])

    with torch.no_grad():
        model.eval()
        
        inputs, labels = method(data['Density_Field'], data[parameters])
        
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            inputs, labels = inputs.cuda(),labels.cuda() 
            output=torch.Tensor().cuda()
            
        else:
            model=model.cpu()
            output=torch.Tensor()
    
        loss=0
        m=32
    
        if len(inputs)>m:
            
            num_complete_minibatches=math.floor(len(inputs)/m)
            for i in range(num_complete_minibatches):
                output=torch.cat((output,model(inputs[i*m:(i+1)*m])),0)
                loss_i = criterion(output[i*m:(i+1)*m],labels[i*m:(i+1)*m]).item()
                loss+=loss_i*m
                torch.cuda.empty_cache()

                

            if len(inputs)%m!=0:
                output=torch.cat((output,model(inputs[num_complete_minibatches*m:])),0)
                loss_i = criterion(output[num_complete_minibatches*m:],labels[num_complete_minibatches*m:]).item()
                loss+=loss_i*(len(inputs)%m)
                
                

            loss=loss/len(inputs)

            
            
            
            
    
        else:        
    
            output = model(inputs)
            loss = criterion(output,labels).item()
            

    output=output.cpu()
    labels=labels.cpu()
        
        
    if plot:
        
        figure, axis = plt.subplots(3,2,figsize=(20,20))
        print("This is for {} data" .format(datafile))
        print("MSE loss: {} " .format(loss))

        axis[0,0].scatter(labels[:,0],output[:,0], s=10)
        axis[0,0].plot(np.arange(0.1,0.5,0.001),np.arange(0.1,0.5,0.001), 'r')
        axis[0,0].set_xlabel('True')
        axis[0,0].set_ylabel('Inference')
        axis[0,0].set_xlim([min(labels[:,0].min(),output[:,0].min())-0.03,max(labels[:,0].max(),output[:,0].max())+0.03])
        axis[0,0].set_ylim([min(labels[:,0].min(),output[:,0].min())-0.03,max(labels[:,0].max(),output[:,0].max())+0.03])
        axis[0,0].set_title("$\Omega_m$ \n MSE Loss : {}" .format(criterion(output[:,0],labels[:,0]).item()))
        axis[0,0].set_rasterized(True)

        axis[0,1].scatter(labels[:,1],output[:,1], s=10)
        axis[0,1].plot(np.arange(0.03,0.07,0.001),np.arange(0.03,0.07,0.001), 'r')
        axis[0,1].set_xlabel('True')
        axis[0,1].set_ylabel('Inference')
        axis[0,1].set_xlim([min(labels[:,1].min(),output[:,1].min())-0.003,max(labels[:,1].max(),output[:,1].max())+0.003])
        axis[0,1].set_ylim([min(labels[:,1].min(),output[:,1].min())-0.003,max(labels[:,1].max(),output[:,1].max())+0.003])
        axis[0,1].set_title("$\Omega_b$ \n MSE Loss : {}" .format(criterion(output[:,1],labels[:,1]).item()))
        axis[0,1].set_rasterized(True)

        axis[1,0].scatter(labels[:,2],output[:,2], s=10)
        axis[1,0].plot(np.arange(0.5,0.9,0.001),np.arange(0.5,0.9,0.001), 'r')
        axis[1,0].set_xlabel('True')
        axis[1,0].set_ylabel('Inference')
        axis[1,0].set_xlim([min(labels[:,2].min(),output[:,2].min())-0.03,max(labels[:,2].max(),output[:,2].max())+0.03])
        axis[1,0].set_ylim([min(labels[:,2].min(),output[:,2].min())-0.03,max(labels[:,2].max(),output[:,2].max())+0.03])
        axis[1,0].set_title("$h$ \n MSE Loss : {}" .format(criterion(output[:,2],labels[:,2]).item()))
        axis[1,0].set_rasterized(True)


        axis[1,1].scatter(labels[:,3],output[:,3], s=10)
        axis[1,1].plot(np.arange(0.8,1.2,0.001),np.arange(0.8,1.2,0.001), 'r')
        axis[1,1].set_xlabel('True')
        axis[1,1].set_ylabel('Inference')
        axis[1,1].set_xlim([min(labels[:,3].min(),output[:,3].min())-0.03,max(labels[:,3].max(),output[:,3].max())+0.03])
        axis[1,1].set_ylim([min(labels[:,3].min(),output[:,3].min())-0.03,max(labels[:,3].max(),output[:,3].max())+0.03])
        axis[1,1].set_title("$n_s$ \n MSE Loss : {}" .format(criterion(output[:,3],labels[:,3]).item()))
        axis[1,1].set_rasterized(True)


        axis[2,0].scatter(labels[:,4],output[:,4], s=10)
        axis[2,0].plot(np.arange(0.6,1,0.001),np.arange(0.6,1,0.001), 'r')
        axis[2,0].set_xlabel('True')
        axis[2,0].set_ylabel('Inference')
        axis[2,0].set_xlim([min(labels[:,4].min(),output[:,4].min())-0.03,max(labels[:,4].max(),output[:,4].max())+0.03])
        axis[2,0].set_ylim([min(labels[:,4].min(),output[:,4].min())-0.03,max(labels[:,4].max(),output[:,4].max())+0.03])
        axis[2,0].set_title("$\sigma_8$ \n MSE Loss : {}" .format(criterion(output[:,4],labels[:,4]).item()))
        axis[2,0].set_rasterized(True)
        
        
        train_loss= pd.read_csv(save_path+'train_result.csv')['train_loss']
        val_loss= pd.read_csv(save_path+'val_result.csv')['val_loss']
        epochs=len(train_loss)
        axis[2,1].plot(np.arange(1,epochs+1), train_loss, 'g')
        axis[2,1].plot(np.arange(1,epochs+1), val_loss, 'm')
        axis[2,1].set_xlabel('Epoch')
        axis[2,1].set_ylabel('Loss')
        axis[2,1].legend(['Train Loss', 'Validation Loss'])
        axis[2,1].set_title("Loss vs Epoch")
        axis[2,1].set_rasterized(True)
        
        if save_fig:
            plt.savefig(save_path+"comparison_"+datafile+".pdf",dpi=600)

    return output, labels


def cov(output):        #covariance from predicted parameters 
    
    l = len(output)
    mean = torch.mean(output,axis=0)
    mat = []
    for line in range(l): 
        arr1d = output[line] - mean
        arr2d = torch.outer(arr1d,arr1d)        
        
        mat.append(np.array(arr2d)) 
        
    mat=np.array(mat)    

    return np.mean(mat,axis=0)

def del_mu(parameters, process_list=[preprocess.PK_log], save_path=""):  #this will give transpose
    
#     df
#     param_pm_dict = {'Omega_m':[['/data74/chartier/3D_cubesDF/Om_p/3D_df128_z0p000/',[0.3275,0.049,0.6711,0.9624,0.834]],['/data74/chartier/3D_cubesDF/Om_m/3D_df128_z0p000/',[0.3075,0.049,0.6711,0.9624,0.834]]],
#                 'Omega_b':[['/data74/chartier/3D_cubesDF/Ob2_p/3D_df128_z0p000/',[0.3175,0.050,0.6711,0.9624,0.834]],['/data74/chartier/3D_cubesDF/Ob2_m/3D_df128_z0p000/',[0.3175,0.048,0.6711,0.9624,0.834]]],
#                  'h':[['/data74/chartier/3D_cubesDF/h_p/3D_df128_z0p000/',[0.3175,0.049,0.6911,0.9624,0.834]],['/data74/chartier/3D_cubesDF/h_m/3D_df128_z0p000/',[0.3175,0.049,0.6511,0.9624,0.834]]],
#                  'n_s':[['/data74/chartier/3D_cubesDF/ns_p/3D_df128_z0p000/',[0.3175,0.049,0.6711,0.9824,0.834]],['/data74/chartier/3D_cubesDF/ns_m/3D_df128_z0p000/',[0.3175,0.049,0.6711,0.9424,0.834]]], 
#                  'sigma_8':[['/data74/chartier/3D_cubesDF/s8_p/3D_df128_z0p000/',[0.3175,0.049,0.6711,0.9624,0.849]],['/data74/chartier/3D_cubesDF/s8_m/3D_df128_z0p000/',[0.3175,0.049,0.6711,0.9624,0.819]]]}
  
#     PK
    param_pm_dict = {'Omega_m':[['/data74/anirban/Quijote/Om_p/PK/128/log_bins/',[0.3275,0.049,0.6711,0.9624,0.834]],['/data74/anirban/Quijote/Om_m/PK/128/log_bins/',[0.3075,0.049,0.6711,0.9624,0.834]]],
                'Omega_b':[['/data74/anirban/Quijote/Ob2_p/PK/128/log_bins/',[0.3175,0.050,0.6711,0.9624,0.834]],['/data74/anirban/Quijote/Ob2_m/PK/128/log_bins/',[0.3175,0.048,0.6711,0.9624,0.834]]],
                 'h':[['/data74/anirban/Quijote/h_p/PK/128/log_bins/',[0.3175,0.049,0.6911,0.9624,0.834]],['/data74/anirban/Quijote/h_m/PK/128/log_bins/',[0.3175,0.049,0.6511,0.9624,0.834]]],
                 'n_s':[['/data74/anirban/Quijote/ns_p/PK/128/log_bins/',[0.3175,0.049,0.6711,0.9824,0.834]],['/data74/anirban/Quijote/ns_m/PK/128/log_bins/',[0.3175,0.049,0.6711,0.9424,0.834]]], 
                 'sigma_8':[['/data74/anirban/Quijote/s8_p/PK/128/log_bins/',[0.3175,0.049,0.6711,0.9624,0.849]],['/data74/anirban/Quijote/s8_m/PK/128/log_bins/',[0.3175,0.049,0.6711,0.9624,0.819]]]}
    
    
    mat= []
    for num, element in enumerate(parameters):
        sum_p = 0
        sum_m = 0
        
        for process in process_list:
        
            path_p=param_pm_dict[element][0][0]
            value_p=param_pm_dict[element][0][1]
            
        
            data_p=pd.DataFrame(columns=['Density_Field']+parameters)
            data_p['Density_Field']=DataLoader.load_df(path_p)
            data_p[parameters]=value_p
    
            output_p, _ = comparison_plot(data_p,str(element)+'_+', parameters=parameters, method=process, save_path=save_path,  plot=False, save_fig=False)
        
        
            path_m=param_pm_dict[element][1][0]
            value_m=param_pm_dict[element][1][1]


            data_m=pd.DataFrame(columns=['Density_Field']+parameters)
            data_m['Density_Field']=DataLoader.load_df(path_m)
            data_m[parameters]=value_m

            output_m, _ = comparison_plot(data_m,str(element)+'_-', parameters=parameters, method=process, save_path=save_path, plot=False, save_fig=False)

        
        
            mean_p = torch.mean(output_p, axis=0)
            mean_m = torch.mean(output_m, axis=0)
            

            sum_p += mean_p
            sum_m += mean_m
    
    
        overall_mean_p = sum_p/len(process_list)
        overall_mean_m = sum_m/len(process_list)
        
        derivative=(overall_mean_p - overall_mean_m)/(value_p[num]-value_m[num])
#         print(derivative)
        
        mat.append(np.array(derivative))
        
    mat=np.array(mat)
        
        
    return mat

def FisherMatrix(data, datafile, parameters, process_list=[preprocess.PK_log], save_path=""):
    sum_out=0
    
    for process in process_list:
        output, _ = comparison_plot(data, datafile, parameters=parameters, method=process, save_path=save_path, plot=False, save_fig=False)
        sum_out += output
        
    output= sum_out/len(process_list)
    
    C = cov(output)
    deriv_T = del_mu(parameters, process_list=process_list, save_path=save_path)
    
    return np.matmul(np.matmul(deriv_T,np.linalg.inv(C)),np.transpose(deriv_T))

    