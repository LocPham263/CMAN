import subprocess
import os
import time
import shutil
from tqdm import tqdm
import SimpleITK as sitk
import numpy as np
import metrics

# weights_ls = ['Sep05-0238', 'Sep18-0309', 'Sep07-1904', 'Sep23-1113', 'Sep18-1337', 
#                 'Sep14-1919', 'Sep13-1704', 'Sep21-2112', 'Sep15-1323',
#                 'Sep07-1226', 'Sep05-1835',
#                 'Sep23-0152', 'Sep24-0309']
# weights_ls = ['Sep07-1226']
# 'Sep14-1919', 'Sep13-1704', 'Sep21-2112', 'Sep15-1323',
# 'Sep07-1226', 'Sep05-1835', 'Sep23-0152', 'Sep240309']                
# weights_ls = ['Sep12-2155']
# weights_ls = ['Mar02-0335/', 'Mar06-0202/', 'Mar06-0850/', 'Mar08-1632/']
weights_ls = ['Sep15-1323/']
# subprocess.run('python eval.py -c weights/Sep12-2155 -v 108nCE', shell=True)

for i in tqdm(range(len(weights_ls))):
    print(weights_ls[i])
    # subprocess.run('python eval.py -c weights/' + weights_ls[i] + ' -v sliver --batch 1', shell=True)
    # subprocess.run('python eval.py -c weights/' + weights_ls[i] + ' -v 108H --batch 1' , shell=True)
    # subprocess.run('python eval.py -c weights/' + weights_ls[i] + ' -v 108CE --batch 1', shell=True)
    # subprocess.run('python eval.py -c weights/' + weights_ls[i] + ' -v 108nCE', shell=True)
    # subprocess.run('python eval.py -c weights/' + weights_ls[i] + ' -v 108_multiphase --paired', shell=True)

subprocess.run('python train.py -b CMAN_CA -n 6 --batch 1', shell=True)
# subprocess.run('python train.py -b CMAN_SA -n 1 --batch 1', shell=True)
# subprocess.run('python train.py -b VTN -n 4 --batch 1 -c weights/Sep22-1951', shell=True)

# ASSD_val = []
# MSD_val = []
# DSC_val = []
# BASE_DIR = '/home/avitech-pc4/Loc/Medical_Image_Processing/RCN/evaluate/main_dataset/CMAN_CA-5-liver/108CE/108CE_p0101_pos_pv_009.nii.gz_108CE_p0123_diag_pv_010.nii.gz/'

# label = sitk.GetArrayFromImage(sitk.ReadImage(BASE_DIR + 'seg_fixed.nii.gz'))
# pred = sitk.GetArrayFromImage(sitk.ReadImage(BASE_DIR + 'seg_warped.nii.gz'))

# Loss = metrics.Metric(label, pred, (1,1,1))
# ASSD = Loss.get_ASSD()
# MSD = Loss.get_MSD()
# DSC = Loss.get_dice_coefficient()

# ASSD_val.append(ASSD)
# MSD_val.append(MSD)
# DSC_val.append(DSC)

# print(np.average(ASSD_val), np.average(MSD_val), np.average(DSC_val)/255)

#  "108F/p0135_pos_ar_004.nii.gz",
# "108F/p0135_pos_nc_002.nii.gz",
# "108F/p0135_pos_pv_005.nii.gz",
# "108F/p0136_diag_nc_002.nii.gz",
# "108F/p0136_diag_pv_006.nii.gz",
# "108F/p0137_diag_nc_002.nii.gz",
# "108F/p0137_diag_pv_008.nii.gz",
# "108F/p0138_diag_pv_010.nii.gz",
# "108F/p0149_pos_ar_005.nii.gz",
# "108F/p0156_pos_ar_003.nii.gz",
# "108F/p0158_pos_ar_005.nii.gz",
# "108F/p0166_pos_ar_005.nii.gz",
# "108F/p0171_pos_ar_005.nii.gz",
# "108F/p0175_pos_ar_004.nii.gz",
# "108F/p0183_pos_ar_005.nii.gz"