"""
Created on May 2025 
@author: Odette Rios-Ibacache 

"""

import sys
sys.path.append('/rtdsm')
from time import process_time
import gc, os, csv, json
import pandas as pd


import sys
sys.path.append('../longiDICOM/code')

from Registration.dicom_registration import get_file_lists
import pydicom as dcm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import json
import sys
import os
import time

sys.path.append('../utils')
import radiomics_utils
from radiomics_utils import *

import mcode_utils
from mcode_utils import *
from mcode_utils import get_dicom_tag_value, get_img_study, get_pat_data_in_dcm, get_acq_tags, recon_parameters
####################################
system = platform.system()

path_patients = '../examples/GBM_burdenko_example/'

def main(): 
    start = time.time()
    if len(sys.argv[1:]) == 0:
        print("WARNING")
        print("Specify patient directory(ies) or write 'all' to sort all patients.")
        exit()
    
    for patient in sys.argv[1:]:
        if patient.lower() == "all":
            patients_to_sort = sorted([f for f in os.listdir(path_patients)])
            for path in patients_to_sort:
                path_save  = '../examples/output_example'
                path_patient = path_patients+path
                print(path_patient)
                if not os.path.isdir(path_save):
                    if system == "Linux":
                        os.system("sudo mkdir " + path_save) #add sudo
                    elif system == "Windows":
                        os.makedirs(path_save)
                new_path_save_images, folder_path_radiomics, folder_path_dosiomics, folder_path = mcode_utils.create_main_paths_per_patient(path_patient,path_save)
    #GET THE PATH OF THE IMAGES 
                paths_images_all = get_all_folder_images(path_patient)

    #GET THE SET OF THE IMAGES OF THE PATIENTS
                for path in paths_images_all:
                    set_images = get_set_images(path)
                    if len(set_images)==0:
                        continue
                    else:
                        save_dicom_attributes_and_volume(path,new_path_save_images)
                end = time.time()
                print("%%%%%%%%%%%%%%%%%%% Total time per patient %%%%%%%%%%%%%%")
                print('                      '+str(end-start)+" seconds              ")
                print('\n')
            
            end2 = time.time()
            print("%%%%%%%%%%%%%%%%%%% Total time %%%%%%%%%%%%%%%%")
            print('                      '+str(end2-start)+" seconds              ")
            print('\n')

        else:
            path_patient = path_patients+patient
    
            path_save  = '../examples/output_example'
            if not os.path.isdir(path_save):
                if system == "Linux":
                    os.system("sudo mkdir " + path_save) #add sudo
                elif system == "Windows":
                    os.makedirs(path_save)

            new_path_save_images, folder_path_radiomics, folder_path_dosiomics, folder_path = mcode_utils.create_main_paths_per_patient(path_patient,path_save)
            paths_images_all = get_all_folder_images(path_patient)

            for path in paths_images_all:
                set_images = get_set_images(path)
                if len(set_images)==0:
                    continue
                else:
                    save_dicom_attributes_and_volume(path,new_path_save_images)
            
            end = time.time()
            print("%%%%%%%%%%%%%%%%%%% Total time per patient %%%%%%%%%%%%%%")
            print('                      '+str(end-start)+" seconds              ")


if __name__ == "__main__":
    main()