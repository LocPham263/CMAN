import os
import shutil
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import h5py

BASE_DIR = '/home/avitech-pc4/Loc/Medical_Image_Processing/108 Hospital/Lung_COPD/SPECT_CT_LUNG/'
data_train_folder = '/'
data_val_folder = '/'

# Code to generate h5 file for train dataset
def h5_generator_train():
    folder_ls = sorted(os.listdir(BASE_DIR + data_train_folder))

    file_clone = h5py.File('./datasets/train_data_0.h5', 'w')
    for __, patient_name in tqdm(enumerate (folder_ls)):
        folder = BASE_DIR + data_train_folder + patient_name
        
        ct_sitk = sitk.ReadImage(folder + '/ct.nii.gz')
        ct = sitk.GetArrayFromImage(ct_sitk)

        file_name = file_clone.create_group(patient_name)
        file_name.create_dataset('volume',data=ct)

    # # Code to generate the text in json file
    # file_addr = './json_clone.txt'
    # with open(file_addr, 'w') as f:
    #     for __, patient_name in tqdm(enumerate (folder_ls)):
    #     # for __, file in enumerate (keys):
    #         # f.write('"108CE/' + ct_folder_ls[i] + '"' + ',' + '\n')
    #         f.write('"H108_CTlsf/' + patient_name + '"' + ',' + '\n')

# Code to generate h5 file for eval dataset
def h5_generator_eval():
    folder_ls = sorted(os.listdir(BASE_DIR + data_val_folder))

    file_clone = h5py.File('./datasets/H108_test_3.h5', 'w')
    for __, patient_name in tqdm(enumerate (folder_ls)):
        folder = BASE_DIR + data_val_folder + patient_name

        ct_sitk = sitk.ReadImage(folder + '/ct.nii.gz')
        ct = sitk.GetArrayFromImage(ct_sitk)

        seg_sitk = sitk.ReadImage(folder + '/seg.nii.gz')
        seg = sitk.GetArrayFromImage(seg_sitk)
        seg = seg*255

        file_name = file_clone.create_group(patient_name + ' Diag')
        file_name.create_dataset('volume',data=ct)
        file_name.create_dataset('seg',data=seg)

    # # Code to generate the text in json file
    # file_addr = './json_clone.txt'
    # with open(file_addr, 'w') as f:
    #     for __, patient_name in tqdm(enumerate (folder_ls)):
    #     # for __, file in enumerate (keys):
    #         # f.write('"108CE/' + ct_folder_ls[i] + '"' + ',' + '\n')
    #         f.write('"H108_test_3/' + patient_name + ' Diag"' + ',\n')

# Code to check the structure of the generated h5 file
def h5_viewer(data_path):
    file = h5py.File(data_path, "r")
    keys = list(file.keys())
    d1 = file[keys[0]]
    keys_d1 = list(d1.keys())

    volume_0 = d1[keys_d1[0]]
    volume_1 = d1[keys_d1[1]]

    print(len(keys))
    print(keys)
    print(keys_d1)
    print(volume_0.shape, type(volume_0), type(volume_0[1,1,1]), np.max(volume_0))
    print(volume_1.shape, type(volume_1), type(volume_1[1,1,1]), np.max(volume_1))
   