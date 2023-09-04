# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 16:54:31 2023

@author: gianlucarloni and evapa
"""

import os

import mat73
from matplotlib import pyplot as plt
import numpy as np
from scipy import io


# De-comment Part 1 or Part 2.

#%% PART 1: VISUALIZATION OF SOME EXAMPLES. MOVE TO PART 2 BELOW FOR COMPUTATION OF THE METRICS (SSIM,PSNR,MSE)

# data_mat_recon = r'Y:/raid/home/gianlucacarloni/CMRxRecon/output_3c/AccFactor04/P111/cine_sax.mat' #TODO: CUSTOMIZE HERE
# data_mat_original = r'Y:/raid/home/gianlucacarloni/CMRxRecon/SingleCoil/Cine/TestSet/AccFactor04/P111/cine_sax.mat' #CUSTOMIZE
# data_mat_full = r'Y:/raid/home/gianlucacarloni/CMRxRecon/SingleCoil/Cine/TestSet/FullSample/P111/cine_sax.mat' #CUSTOMIZE

# dict_recon = io.loadmat(data_mat_recon)
# img_recon = dict_recon["img4ranking"]

# dict_original = mat73.loadmat(data_mat_original)
# kspace = dict_original["kspace_single_sub04"]

# dict_full = mat73.loadmat(data_mat_full)
# kspace_full = dict_full["kspace_single_full"]

# #img_original = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(kspace))) # 3D COMPUTATION INDUCES SOME ISSUES, LET'S DO IT IN 2D
# img_original = np.zeros_like(kspace, dtype=np.float32)
# for space in range(kspace.shape[2]):
#     for time in range(kspace.shape[3]):
#         img_original[:,:,space,time]=np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace[:,:,space,time]))))
        
# #img_full = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(kspace_full)))
# img_full = np.zeros_like(kspace_full, dtype=np.float32)
# for space in range(kspace_full.shape[2]):
#     for time in range(kspace_full.shape[3]):
#         img_full[:,:,space,time]=np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace_full[:,:,space,time]))))
        

# fig = plt.figure()
# fig.tight_layout()

# for i in range(10):
#     plt.subplot(2,5,i+1)
#     plt.imshow(img_recon[:,:,i,0],cmap="gray")
#     plt.axis('off')
#     plt.title("REC-s"+str(i))

# fig = plt.figure()
# fig.tight_layout()
# for i in range(10):
#     plt.subplot(2,5,i+1)
#     plt.imshow(img_original[:,:,i,0],cmap="gray")
#     plt.axis('off')
#     plt.title("Acc04-s"+str(i))


# fig = plt.figure()
# fig.tight_layout()
# for i in range(10):
#     plt.subplot(2,5,i+1)
#     plt.imshow(img_full[:,:,i,0],cmap="gray")
#     plt.axis('off')
#     plt.title("FULL-s"+str(i))



#%% PART 2: EVALUATION METRICS
'''
This takes as input (line 114,115) the folder of reconstructed images (from inference.py) saved in npy, and computes the metrics CSV file for each
'''
import time
import pandas as pd
from tqdm import tqdm
# import openpyxl


#option with utility function:
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse

def compute_metrics_2D(true_volume, recon_volume):
    #volumes are of shape [H, W, spatial_slices, temporal_instances]
    S = true_volume.shape[2]
    T = true_volume.shape[3]
    
    psnr_val = 0.0
    ssim_val = 0.0
    mse_val = 0.0
    
    for s_space in range(S):
        for t_time in range(T):
            img_full = true_volume[:,:,s_space,t_time].astype(float)
            img_recon = recon_volume[:,:,s_space,t_time].astype(float)
            
            psnr_val += psnr(img_full, img_recon, data_range=(img_recon.max()-img_recon.min()))  
            ssim_val += ssim(img_full, img_recon, data_range=(img_recon.max()-img_recon.min()))
            mse_val += mse(img_full, img_recon)

    # print(f"Spatial={S}; Temporal={T}; Average on S*T={S*T}")
    S=float(S)
    T=float(T)
    return psnr_val/(S*T), ssim_val/(S*T), mse_val/(S*T)


##TODO: customise the following:
name_of_reconstrucedImages_folder = "output_6c" #E.g., "output_3c", "output_6c"  
name_of_partition = "TestSet" ##"TestSet", "ValidationSet", "TrainingSet"
##

df = pd.DataFrame(columns=['Acc Factor', 'Patient name', 'Mat file', 'PSNR', 'SSIM', 'MSE'])
root_path_reconstructed_images = os.path.join(os.getcwd(), name_of_reconstrucedImages_folder) #e.g., Y:/raid/home/gianlucacarloni/CMRxRecon/output/
root_path_dataset = os.path.join(os.getcwd(),"SingleCoil","Cine", name_of_partition) #e.g., Y:/raid/home/gianlucacarloni/CMRxRecon/SingleCoil/Cine/TestSet/

acc_factors = os.listdir(root_path_reconstructed_images)

k=0
for acc_factor in acc_factors:
    patients_names = os.listdir(os.path.join(root_path_reconstructed_images,acc_factor))
    for patient_name in tqdm(patients_names):
        
        mat_files = os.listdir(os.path.join(root_path_reconstructed_images,acc_factor, patient_name))
    
        for mat_file in mat_files:
                print(f"Acc Factor: {acc_factor}, mat_file {mat_file}")
                recon_file_path = os.path.join(root_path_reconstructed_images, acc_factor, patient_name, mat_file)
                
                ## Option 1: Reconstructed images are in MAT 5.0 format
                # dict_recon = io.loadmat(recon_file_path)
                # img_recon = dict_recon["img4ranking"] #the dictionary stores the images directly
                ## TODO: Option 2: Variant using NUMPY
                img_recon = np.load(recon_file_path)
            
                ## Retrieve the corresponding original fully sampled image
                fully_file_path = os.path.join(root_path_dataset, "FullSample", patient_name, mat_file)  
                if fully_file_path.endswith(".npy"):
                    fully_file_path = fully_file_path[:-4]
                dict_full = mat73.loadmat(fully_file_path)
                kspace_full = dict_full["kspace_single_full"] #here, instead, we obtain kspaces so we need to convert to image space

                
                # let's do it slice by slice (2D)
                img_full = np.zeros_like(kspace_full, dtype=np.float32)
                for s_space in range(kspace_full.shape[2]):
                    for t_time in range(kspace_full.shape[3]):
                        img_full[:,:,s_space,t_time]=np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace_full[:,:,s_space,t_time]))))
                        
                ## Optionally visualize some of the paired images:                            
                # fig = plt.figure()
                # fig.tight_layout()            
                # plt.subplot(121)
                # plt.imshow(img_recon[:,:,1,1],cmap="gray")
                # plt.axis('off')
                # plt.title("REC-1")
                # plt.subplot(122)
                # plt.imshow(img_full[:,:,1,1],cmap="gray")
                # plt.axis('off')
                # plt.title("FULL-1")
                # plt.suptitle(f"{name_of_reconstrucedImages_folder} on {name_of_partition}: {acc_factor} {patient_name} {mat_file}")
                # # plt.savefig("myplot.png")
                # # plt.close()
                            
                ## Compute metrics                      
                psnr_val, ssim_val, mse_val = compute_metrics_2D(img_full, img_recon)    
    
                
                df.loc[k] = [acc_factor, patient_name, mat_file, psnr_val, ssim_val, mse_val]
                k+=1
                
                
                
                #create or over-write (update) the CSV file:
                try:
                    df.to_csv(os.path.join(os.getcwd(),f"metrics_{name_of_reconstrucedImages_folder}_{name_of_partition}.csv"))
                except PermissionError: #in case you are taking a look at the current CSV while it tries to write it again, wait 6 seconds
                    time.sleep(6)
                
                # create the excel file with image.. works fine but it is too heavy and poorly readable:                
                # try:
                #     df.to_excel(os.path.join(os.getcwd(),f"metrics_{name_of_reconstrucedImages_folder}_{name_of_partition}.xlsx"))
                #     wb = openpyxl.load_workbook(f"metrics_{name_of_reconstrucedImages_folder}_{name_of_partition}.xlsx")
                #     ws = wb.active    
                #     ws[f"A{k}"] = 'Sample images'
                #     img = openpyxl.drawing.image.Image('myplot.png')
                #     # img.anchor(ws.cell(row=k, column=1) )           
                #     ws.add_image(img,f"A{k}")
                #     wb.save(f"metrics_{name_of_reconstrucedImages_folder}_{name_of_partition}.xlsx")
                # except PermissionError: #in case you are taking a look at the current CSV while it tries to write it again, wait 6 seconds
                #     time.sleep(6)                    