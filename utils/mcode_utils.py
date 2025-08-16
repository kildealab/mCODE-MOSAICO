import pydicom as dcm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import json
import radiomics 
import shutil
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon
import random

import SimpleITK as sitk
import six
import nrrd

from radiomics import featureextractor, getFeatureClasses
from skimage.draw import polygon


from scipy.ndimage import shift
from matplotlib import ticker

###########################################################
# UTILS TO GET DICOM TAG INFORMATION 
############################################################

#GETS THE TAG VALUE FROM THE DICOM 
def get_dicom_tag_value(dicom, keyword, default=None):
    '''this function will get the dicom tag from the dicom filde for the given tag/code'''
    if isinstance(keyword,str):
        tag_value = dicom.get(keyword,None)
        if tag_value is None:
            return default
    else: 
        tag_value = dicom.get(keyword,None).value
    return tag_value

#WITH THIS CODE WE ARE READING THE IMAGE STUDY DICOM METADATA AND STRUCTURING IT!
#THE INFORMATION IS PART OF THE MCODE MEDICAL IMAGES EXTENSION
def get_img_study(path,dicom):
    tag_label_list = ['Date of Imaging',"Image Modality", "Image Identifier","Body Site","Body Structure or Part","Series Date","Study Description","Reason"]
    img_study = {}
    img_study_keywords = ['SeriesDate','Modality','StudyInstanceUID','AnatomicRegionSequence',"BodyPartExamined", "SeriesDate", 'StudyDescription',
                    'ReasonForPerformedProcedureCodeSequenceAttribute']
    
    for label in range(0,len(img_study_keywords)):
        if isinstance(tag_label_list[label], list):
            img_study[tag_label_list[label]] = {}
            for word in range(1,len(img_study_keywords[label])):
                img_study[tag_label_list[label][0]][tag_label_list[label][word]]= str(get_dicom_tag_value(dicom,img_study_keywords[label][word-1]))
        else:
            img_study[tag_label_list[label]] = str(get_dicom_tag_value(dicom,img_study_keywords[label]))
    
    return img_study

#FUNCTION GETS THE PATIENT DATA FROM THE DCM FILES THAT ARE COMPLIANT WITH mCODE
def get_pat_data_in_dcm(path,dicom):
    pat_keywords=['PatientName','PatientBirthDate','PatientSex',
                'EthnicGroup','PatientAge','PatientWeight']
    
    pat_labels=["Patient Name","Patient's Birth Date","Patient's Sex",
                "Patient's Ethnic Group","Patient's Age","Patient's Weight"]
    patient_data = {}
    for keyword in range(0,len(pat_keywords)):
        patient_data[pat_labels[keyword]] = str(get_dicom_tag_value(dicom,pat_keywords[keyword]))
        
    return patient_data

#GETS THE ACQUISITION ATTRIBUTES OF MEDICAL IMAGES (PART OF THE MEDICAL IMAGES MCODE EXTENSION)
def get_acq_tags(path,dicom):
    acquisition_dict = {"Imaging Protocol": str(get_dicom_tag_value(dicom,'PerfomedCodeSequenceAttribute')) 
                        +'\n' + str(get_dicom_tag_value(dicom,'CodeMeaning'))}
            
    acq_labels = ["Scanner Vendor","Scanner Type","Scan Duration",
                ["Contrast Enhancement/Bolus","Agent","Ingredient"],
                ["Acquisition Field Of View","Reconstruction FOV Diameter",
                 "FOV Shape","FOV Dimensions", "FOV/Geometry"],
                ["Patient Instructions","Patient Orientation",
                 "Instruction Sequence"]]
    
    acq_keywords = ['Manufacturer',
             'ManufacturerModelName',
                'AcquisitionTime',
                ["ContrastBolusAgent",
                "ContrastBolusIngredient"],
                ["ReconstructionDiameter",'FieldOfViewShape','FieldOfViewDimensions',"PercentPhaseFieldOfView"],
                ["PatientOrientation","PatientPosition",
                 "InstructionSequence"]]

    for label in range(0,len(acq_labels)):
        if isinstance(acq_labels[label], list):
            acquisition_dict[acq_labels[label][0]] = {}
            #print(len(acq_labels[label]))
            for word in range(1,len(acq_labels[label])):
                #print(acq_labels[label][word])
                acquisition_dict[acq_labels[label][0]][acq_labels[label][word]]= str(get_dicom_tag_value(dicom,acq_keywords[label][word-1]))
        else:
             acquisition_dict[acq_labels[label]] = str(get_dicom_tag_value(dicom,acq_keywords[label]))
        
    return acquisition_dict
    
#GETS THE RECONSTRUCTION PARAMETERS OF THE MEDICAL IMAGE OF INTEREST
#THIS INFORMATION IS PART OF OUR MCODE EXTENSION
def recon_parameters(path,dicom):
    recon_dict = {}
    recon_labels = ["Image Type","Slice Thickness (mm)","Slice Spacing (mm)", 
                "Pixel Spacing (mm)", ["Reconstruction Technique","Method","Algoritihm"],
                "Convolution Kernel"]
    
    recon_keywords = ["Modality","SliceThickness","SpacingBetweenSlices","PixelSpacing",
                ["ReconstructionMethod","ReconstructionAlgorithm"],"ConvolutionKernel"]
    
    for label in range(0,len(recon_labels)):
        if isinstance(recon_labels[label], list):
            recon_dict[recon_labels[label][0]] = {}
            #print(len(recon_labels[label]))
            for word in range(1,len(recon_labels[label])):
                #print(recon_labels[label][word])
                recon_dict[recon_labels[label][0]][recon_labels[label][word]]= str(get_dicom_tag_value(dicom,recon_keywords[label][word-1]))
        else:
             recon_dict[recon_labels[label]] = str(get_dicom_tag_value(dicom,recon_keywords[label]))
                
    return recon_dict
    
#GETS THE TAGS OF INTEREST FOR THE CT IMAGE (PART OF THE mCODE EXTENSION)
def get_ct_tags(path,dicom):
    ct_dict = {}
   
    ct_tags = ["ImageType","ScanOptions","KVP",
                "XRayTubeCurrent", "ExposureTime","SpiralPitchFactor",
               "ImageFilter"]
                #"Acquisition Field Of View": {"Scan Field of View": dicom.DataCollectionDiameter,"Reconstruction FOV Diameter": dicom.ReconstructionDiameter,
                #"FOV Origin": dicom.FieldOfViewOrigin},
                #"Image Orientation Patient": list(dicom.ImageOrientationPatient),
               # "Series Instance UID": str(dicom.SeriesInstanceUID),
            
    ct_labels = ["Image Type", "Scan Mode","KVP","XRayTubeCurrent","Exposure (msec)",
               "Spiral Pitch","Image Filter"]
                #"Acquisition Field Of View": {"Scan Field of View": dicom.DataCollectionDiameter,"Reconstruction FOV Diameter": dicom.ReconstructionDiameter,
                #"FOV Origin": dicom.FieldOfViewOrigin},
                #"Image Orientation Patient": list(dicom.ImageOrientationPatient),
               # "Series Instance UID": str(dicom.SeriesInstanceUID),
            
    for keyword in range(0,len(ct_tags)):
        ct_dict[ct_labels[keyword]] = str(get_dicom_tag_value(dicom,ct_tags[keyword]))
        
    return ct_dict
    
    
#MRI INFORMATION OF INTEREST FROM THE DICOM FILE
#REQUIRED INFORMATION FOR THE mCODE EXTENSION
def get_mri_tags(path,dicom):
    mri_dict = {}
   
    mri_tags = ["ImageType","SequenceName","MagneticFieldStrength","RepetitionTime","EchoTime",
                "EchoTrainLength","InversionTime","FlipAngle","NumberOfAverages","GeometryOfKSpaceTraversal"]
    
    mri_labels = ["Image Type","Scanning Sequence Acquired","Magnetic Field Strength",
                "Repetition Time","Echo Time","Echo Time Length","Inversion Time",
                "Flip Angle","Number of Excitations","k-Space Trajectory"]
    
    for keyword in range(0,len(mri_tags)):
        mri_dict[mri_labels[keyword]] = str(get_dicom_tag_value(dicom,mri_tags[keyword]))
    
    return mri_dict
    
#PET INFORMATION TO EXTRACT THE mCODE INFORMATION EXTRACTION
def get_pet_tags(dicom):
    pet_dict = {}
   
    pet_tags = ["ImageType",["Radiopharmaceutical",0x00091036],"RadiopharmaceuticalAdministrationEventUID",
                 [["RadionuclideTotalDose",0x00091038],"RadiopharmaceuticalSpecificActivity"],"TimeOfFlightinformationUsed",
               ["CorrectedImage","ScatterCorrectionMethod","ScatterFractionFactor","RandomsCorrectionMethod","AttenuationCorrectionMethod","DecayFactor","DeadTimeFactor"],"TypeOfDetectorMotion"]
    
    pet_labels = ["Image Type","Radioactive Tracer","Radioactive Tracer Admin. Method",["Injected Activity","Total Dose (Bq)","Specific Activity (Bq/micromole)"], "Time-of-flight",
                  ["Image Correction Method","Methods Applied","Scatter Correction","Scatter Correction Factor","Randoms Correction","Attenuation Correction","Decay Correction Factor","Dead Time Factor"],"Type of Detector Bed Motion"]
    
    for label in range(0,len(pet_tags)):
        #print(pet_tags[label])
        if isinstance(pet_labels[label], list):
            pet_dict[pet_labels[label][0]] = {}
            for word in range(1,len(pet_labels[label])):
                if isinstance(pet_tags[label][word-1], list):
                    count = 0
                    for word2 in range(0,len(pet_tags[label][word-1])):
                        
                        if str(get_dicom_tag_value(dicom,pet_tags[label][word-1][word2]))== None:
                            count = count + 1
                            continue
                        elif count==len(pet_tags[label][word]):
                            pet_dict[pet_labels[label][0]][pet_labels[label][word]] = None 
                        else:
                            pet_dict[pet_labels[label][0]][pet_labels[label][word]] = str(get_dicom_tag_value(dicom,pet_tags[label][word-1][word2]))
                else:
                    pet_dict[pet_labels[label][0]][pet_labels[label][word]]= str(get_dicom_tag_value(dicom,pet_tags[label][word-1]))
                
        elif isinstance(pet_tags[label], list)==True and isinstance(pet_labels[label], list)==False:
            for word in range(0,len(pet_tags[label])):
                count0 = 0        
                if str(get_dicom_tag_value(dicom,pet_tags[label][word]))==None:
                    count0 = count0+1
                    continue
                elif count0==len(pet_tags[label]):
                    pet_dict[pet_labels[label]] = None 
                else:
                    pet_dict[pet_labels[label]] = str(get_dicom_tag_value(dicom,pet_tags[label][word])) 
        else:
            pet_dict[pet_labels[label]] = str(get_dicom_tag_value(dicom,pet_tags[label]))
    
    return pet_dict


    
#####################################################
#  SAVE JSON FILES
#####################################################
    
#SAVE THE JSON FILES FOR A GIVEN PATH TO SAVE (PATH_SAVE), FOR A GIVEN DICOM IMAGE
def save_JSON_attributes(path_save,path,dicom):
    img_study_dict = get_img_study(path,dicom)
    with open(path_save+"/image_study_attributes.json", "w") as outfile: 
        json.dump(img_study_dict, outfile,indent=4)
    
    print('-------------------------------------------')
    pat_data_dict = get_pat_data_in_dcm(path,dicom)
    print('-------------------------------------------')
    print(pat_data_dict)
    with open(path_save+"/cancer_patient_attributes.json", "w") as outfile: 
        json.dump(pat_data_dict, outfile,indent=4)
   
    acq_data_dict =get_acq_tags(path,dicom)
    
    with open(path_save+"/image_acquisition_attributes.json", "w") as outfile: 
        json.dump(acq_data_dict, outfile,indent=4)
        
    recon_data_dict =recon_parameters(path,dicom)
    
    with open(path_save+"/reconstruction_attributes.json", "w") as outfile: 
        json.dump(recon_data_dict, outfile,indent=4)
        
    if str(img_study_dict['Image Modality']).lower()=='ct' or str(img_study_dict['Image Modality']).lower()=='cbct':    
        ct_dict_data = get_ct_tags(path,dicom)
        with open(path_save+"/CT_acquisition_properties.json", "w") as outfile: 
            json.dump(ct_dict_data, outfile,indent=4)
    elif str(img_study_dict['Image Modality']).lower()=='mri':
        mri_dict_data = get_mri_tags(path,dicom)
        with open(path_save+"/MRI_acquisition_properties.json", "w") as outfile:
            json.dump(mri_dict_data, outfile,indent=4)
    elif str(img_study_dict['Image Modality']).lower()=='pt':
        pet_dict_data = get_pet_tags(path,dicom)
        with open(path_save+"/PET_acquisition_properties.json", "w") as outfile:
            json.dump(pet_dict_data, outfile,indent=4)
            
    return 
            


#################################################
  #  UTILS TO CREATE AND SAVE THE PATHS FOR EACH PATIENT
#################################################

#CREATE THE FOLDER FOR THE GIVEN PATIENT USING THE MRN
def create_patient_folder(folder_path):
    #SET THE PATH FOR THE PATIENT #CHANGE THE PATH IF NECESSARY
    name_patient_mrn = folder_path.split('_')[-1]
    if os.path.isdir(folder_path):  
        print(f"The folder '{folder_path}' exists.") 
    else:
        
        print(f"The folder '{folder_path}' does not exist") #CREATION OF THE FOLDER
        print(f"Creating folder for Patient '{name_patient_mrn}'...")
        os.mkdir(folder_path)
        print(f"Folder created sucessfully")
        #return folder_path

#IT COPIES THE ORIGINAL DIRECTORY NAMES WITHOUT THE FILES IN IT, ONLY THE LINK PATHS OF THE FOLDERS. 
def copy_paths(source_folder):
    paths = []
   
    for root, _, files in os.walk(source_folder):
        for file in files:
            file_path = os.path.join(root, file)
            if '.dcm' in file_path or '.nrrd' in file_path or '.nii.gz' in file_path or '.nii' in file_path:
                continue
            else:
               
                paths.append(file_path)

        for subdir in os.listdir(root):
            subdir_path = os.path.join(root, subdir)
            if '.dcm' in subdir_path or '.nrrd' in subdir_path or '.nii.gz' in subdir_path or '.nii' in subdir_path:
                continue
            else:
                if os.path.isdir(subdir_path):
                  
                    paths.append(subdir_path)
    return paths

#GET THE PATHS OF THE DIRECTORY OF THE DIRECTORIES AVAILABLE
def create_folder(folder_path):
    if os.path.isdir(folder_path):  
        print(f"The folder '{folder_path}' exists.") 
    else:
        print(f"The folder '{folder_path}' does not exist") #CREATION OF THE FOLDER   
        os.mkdir(folder_path)
        print(f"Folder created sucessfully")
        print('\n')
    
def get_folders_in_directory(directory_path):
    """
    Returns a list of folder names present in the given directory.
    """
    folder_list = [entry.name for entry in os.scandir(directory_path) if entry.is_dir()]
    return folder_list
    
def get_dict_paths(directory_path,paths_dict,folders):
    for folder in folders:
        directory_path2= directory_path+'/'+folder
        last_format = directory_path2.split('.')[-1]
        folders2 = sorted(get_folders_in_directory(directory_path2))
        if len(folders2)!=0:
            paths_dict[folder] = folders2    
        else:
            continue
        
        get_dict_paths(directory_path2,paths_dict,folders2) 

#def save_dose_map()

    #resample_dose_dist(dsDose,number_slices,dsCTs,ctArray)
    #save_RT_dose_as_nrrd(rt_dose,set_images,dcm_save_path)
    

#CREATE THE FOLDER PER EACH LINK PATH IN THE NEW DIRECTORY (SAVE_PATH) 
#BASICALLY FOR NOW, IT CLONES THE FOLDERS IN EACH PATIENT FOLDER WITHOUT THE FILES, ONLY THE DIRECTORIES
def create_folder_per_path(save_path,paths_dict,folders):
    for folder in folders:
        directory_path2= save_path+'/'+folder+'/'
        create_folder(directory_path2)
        if folder in paths_dict.keys():
            folders2 = paths_dict[folder]
            for folder2 in folders2:
                new_path = directory_path2+folder2
                create_folder(new_path)
        else:
            continue
        
        create_folder_per_path(new_path,paths_dict,folders2)

#FUNCTION TO PERFORM THE EXTRACTION OF DATA OF ALL THE PATIENTS
def clone_folders_per_patient(path_patient,path_save):
    directory_path = path_patient
    folders = sorted(get_folders_in_directory(directory_path))
      
    paths_dict = {'Patient':directory_path.split('/')[-1]}
    paths_dict['Folders'] = folders
    
    #return paths_dict
    get_dict_paths(directory_path,paths_dict,folders)

    create_patient_folder(path_save,paths_dict['Patient'])
    folder_path = path_save+"/patient_"+paths_dict['Patient']
    path_save3 = folder_path+'/medical_images'
        
    create_folder(path_save3)
    
    create_folder_per_path(path_save3,paths_dict,paths_dict['Folders'])
    create_folder(folder_path+'/radiomics')
    create_folder(folder_path+'/dosiomics')
    return path_save3

def create_main_paths_per_patient(path_patient,path_save):
    directory_path = path_patient
    folders = sorted(get_folders_in_directory(directory_path))
      
    paths_dict = {'Patient':directory_path.split('/')[-1]}
    paths_dict['Folders'] = folders
     
    folder_path = path_save+"/patient_"+paths_dict['Patient']
    path_save_images = folder_path+'/medical_images'
    folder_path_radiomics = folder_path+'/radiomics'
    folder_path_dosiomics = folder_path+'/dosiomics'

    return path_save_images, folder_path_radiomics, folder_path_dosiomics, folder_path
    
def create_folders_main_folders_per_patient(folder_path,new_path_save_images,folder_path_radiomics,folder_path_dosiomics):
    create_patient_folder(folder_path)
    create_folder(new_path_save_images)
    create_folder(folder_path_radiomics)
    create_folder(folder_path_dosiomics)

def get_all_folder_images(path_patient):
    folder_path_images = sorted(copy_paths(path_patient))
    return folder_path_images

def save_dicom_attributes_and_volume(folder_image,new_path_save_images):
    #CREATING FOLDER FOR THE SPECIFIC PATIENT
    #GETTING SET OF IMAGES FROM THE PATH OF THE DATA

    name_folder = folder_image.split('/')[-1]
    set_images = get_set_images(folder_image)
    
    if len(set_images)!=0:
        if isCBCT(set_images)==True:
            try:
                image_name = set_images[0].ContentDate+'_CBCT' #DEFINE THE FOLDER NAME FOR THE MODALITY
            except:
                image_name = set_images[0].SeriesDate+'_CBCT'
        else: 
            try:
                image_name = set_images[0].ContentDate+'_'+set_images[0].Modality #DEFINE THE FOLDER NAME FOR THE MODALITY
            except:
                image_name = set_images[0].SeriesDate+'_'+set_images[0].Modality
             
    new_folder_name = new_path_save_images+'/'+image_name
    create_folder(new_folder_name)
    save_JSON_attributes(new_folder_name,folder_image,set_images[0])
    save_img_dcm_as_nrrd(set_images, new_folder_name,image_name)

def isCBCT(set_images):
    try: 
        is_CBCT= set_images[0][0x0008,0x114a].value[0][0x0008,0x1150].value
        return True
    except:
        return False
 

 
