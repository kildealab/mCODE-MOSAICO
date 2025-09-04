import pydicom as dcm
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import json
import gc
import time
import sys
sys.path.append('../utils')
import radiomics_utils
from radiomics_utils import *

import dicom_utils
from dicom_utils import *

factors = ['D0', 'D100', 'D50','mean','V0.5cc']

path_patients = '../examples/GBM_burdenko_example/'
path_save   = '../examples/output_example'

def main():
    start = time.time()
    if len(sys.argv[1:]) <= 0:
        print("WARNING")
        print("Specify patient directory(ies) or write 'all' to sort all patients." +'\n'+
              "Please ALSO specify the ROI of interest")
        exit()
        
    elif len(sys.argv)>=3:
        patient = sys.argv[1]
        ROI_name = sys.argv[2]  

        if patient.lower() == "all":
            patients_to_sort = sorted([f for f in os.listdir(path_patients)])
        
            for path in patients_to_sort:
                print("          Processing patient: "+path[0]+"       ")
                
                path_patient = path_patients+path
                PATH_TO_SAVE = check_main_dosimetric_per_patient(path_patient,path_save)+'/'

                paths_images_all = get_all_folder_images(path_patient)

                paths_RT = get_dirs_RT(paths_images_all,False)
                paths_RD = get_dirs_RD(paths_RT,paths_images_all,False)
                gc.collect()
                for key in paths_RT.items():
                    RS_file_path = paths_RT[key[0]]
                    RS_file_path_2 = ("/").join(RS_file_path.split('/')[:-1])
                    keys_RD = [key_RD[0] for key_RD in paths_RD.items()]
                    if RS_file_path_2 in keys_RD:
                        RD_file_path = paths_RD[RS_file_path_2]
                    
                        all_dosimetric_features_dvh_json_input(RS_file_path,RD_file_path,PATH_TO_SAVE,factors,str(ROI_name))
            
        else:
            
            path_patient = path_patients+patient
            PATH_TO_SAVE = check_main_dosimetric_per_patient(path_patient,path_save)+'/'
            
            print("             Processing patient: "+patient+"           ")
            
            paths_images_all = get_all_folder_images(path_patient)
            paths_RT = get_dirs_RT(paths_images_all,False)
            paths_RD = get_dirs_RD(paths_RT,paths_images_all,False)
            
            gc.collect()
            for key in paths_RT.items():
                    RS_file_path = paths_RT[key[0]]
                    RS_file_path_2 = ("/").join(RS_file_path.split('/')[:-1])
                    keys_RD = [key_RD[0] for key_RD in paths_RD.items()]
                
                    if RS_file_path_2 in keys_RD:
                        RD_file_path = paths_RD[RS_file_path_2]
                        all_dosimetric_features_dvh_json_input(RS_file_path,RD_file_path,PATH_TO_SAVE,factors,str(ROI_name))
            

if __name__ == "__main__":
    main()
    
    
