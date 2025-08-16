"""
Created on May 2025 
@author: Odette Rios-Ibacache 

"""

import sys
sys.path.append('/rtdsm')

from time import process_time
import gc, os, csv, json
import pandas as pd

from longiDICOM import *
import sys
sys.path.append('longiDICOM/code')
from Registration.dicom_registration import get_file_lists

import mcode_utils
 
from mcode_utils import get_dicom_tag_value, get_img_study, get_pat_data_in_dcm, get_acq_tags, recon_parameters
import dicom_utils*


####################################

if __name__ == "__main__":
    
    #SPECIFYING PATH IN THE CODE
    path_patient = '/mnt/iDriveShare/Kayla/CBCT_images/kayla_extracted/19'
    path_save  = 'TRY2'
    #path_RS_file = '/mnt/iDriveShare/Kayla/CBCT_images/kayla_extracted/'
    if not os.path.isdir(path_save):
        os.makedirs(path_save)
    create_folders_main_folders_per_patient(path_patient,path_save)
    paths_images_all = get_all_folder_images(path_patient)
    new_path_save_images, folder_path_radiomics, folder_path_dosiomics, folder_path = create_main_paths_per_patient(path_patient,path_save)
    create_folders_main_folders_per_patient(folder_path,new_path_save_images,folder_path_radiomics,folder_path_dosiomics)
    
    save_dicom_attributes_and_volume(paths_images_all[0],new_path_save_images)
    path_RS_all = get_all_folder_RS(path_patient)
    #create_ROI_folders(RS_file_path,folder_path_radiomics)
   
     
    #get_set_RS(dir_image_path)
    
      #   for ROI in rois:
     #        create_folder(folder_image+'/radiomics/'+ROI+'_radiomics')
    #seg = sitk.ReadImage('seg_Parotid_R.nrrd',imageIO='NrrdImageIO')
             
    #path_radiomics = '/radiomics/'+path_image+'/'+ROI+'_radiomics' 
             