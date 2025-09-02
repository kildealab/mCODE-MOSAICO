import pydicom as dcm
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import json
import gc

import sys
sys.path.append('../utils')
import radiomics_utils
from radiomics_utils import *

factors = ['D0', 'D100', 'D50','mean','V0.5cc']

path_patients = '../examples/GBM_burdenko_example/'

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
                path_save  = '../examples/output_example'
                path_patient = path_patients+path
                
                
                RS_file_path = search_RS_file(paths_images_all[0],set_images,set_images[0].StudyInstanceUID)
                create_ROI_folders_and_radiomics_specific(RS_file[-1],ROI_name,folder_path_radiomics,set_images,new_path_save_images)
                
                RD_file_path = 
                all_dosimetric_features_dvh_json(RS_file_path,RD_file_path,PATH_TO_SAVE,factors,str(ROI_name))
                
                    
        else:
            path_patient = path_patients+patient
            RS_file_path = 
            RD_file_path = 
            all_dosimetric_features_dvh_json(RS_file_path,RD_file_path,PATH_TO_SAVE,factors,str(ROI_name))
            
                

    


if __name__ == "__main__":
    main()
    