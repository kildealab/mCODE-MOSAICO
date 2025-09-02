import pydicom as dcm
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import json
import sys
import gc

sys.path.append('../utils')
import radiomics_utils
from radiomics_utils import *

path_patients = '../examples/GBM_burdenko_example/'

def main():
    start = time.time()
    if len(sys.argv[1:]) <= 0:
        print("WARNING")
        print("Specify patient directory(ies) or write 'all' to sort all patients." +'\n'+
              "Please ALSO specify the ROI of interest and image of interest. You can write 'all' for both.")
        exit()
        
    elif len(sys.argv)>=4:
        patient = sys.argv[1]
        ROI_name = sys.argv[2]
        images = sys.argv[3]
        
        if patient.lower() == "all":
            patients_to_sort = sorted([f for f in os.listdir(path_patients)])
            for path in patients_to_sort:
                path_save  = '../examples/output_example'
                path_patient = path_patients+path
                paths_images_all = get_all_folder_images(path_patient)
                
                for path in paths_images_all:
                    set_images = get_set_images(path)
                    if len(set_images)!=0:
                        for path_2 in paths_images_all:
                            RS_file = search_RS_file(path_2,set_images[0].SeriesDescription)
                            if len(RS_file)!=0:
                                RS_file.sort(key = lambda x: (x.StructureSetDate))
                                ROis = get_ROI_keys_2(RS_file[0])
                                create_ROI_folders_and_radiomics_specific(RS_file[0],'Brain',folder_path_radiomics,set_images,new_path_save_images)
                    
        else:
            path_patient = path_patients+patient
            paths_images_all = get_all_folder_images(path_patient)
            
            RS_file_path = 
                            
            
                

    


if __name__ == "__main__":
    main()
    