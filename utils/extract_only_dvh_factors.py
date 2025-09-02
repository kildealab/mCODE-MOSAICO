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
PATH_TO_SAVE = '../examples/output_example'
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
                path_patient = path_patients+path
                paths_images_all = get_all_folder_images(path_patient)
                
                paths_RT = get_dirs_RT(paths_images_all,False)
                paths_RD = get_dirs_RD(paths_RT,paths_images_all,False)
                gc.collect()
                for key in paths_RT.items():
                    print(paths_RT[key[0]])
                    RS_file_path = paths_RT[key[0]]
                    RD_file_path = paths_RD[RS_file_path]
                    all_dosimetric_features_dvh_json_input(RS_file_path,RD_file_path,PATH_TO_SAVE,factors,str(ROI_name))
   
        else:
            path_patient = path_patients+patient
            paths_images_all = get_all_folder_images(path_patient)
            
            paths_RT = get_dirs_RT(paths_images_all,False)
            paths_RD = get_dirs_RD(paths_RT,paths_images_all,False)
            gc.collect()
            for key in paths_RT.items():
                print(paths_RT[key[0]])
                RS_file_path = paths_RT[key[0]]
                RD_file_path = paths_RD[RS_file_path]
            
                all_dosimetric_features_dvh_json_input(RS_file_path,RD_file_path,PATH_TO_SAVE,factors,str(ROI_name))


if __name__ == "__main__":
    main()
    
    
