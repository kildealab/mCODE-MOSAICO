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
#THE INFROMATION IS PART OF THE MCODE MEDICAL IMAGES EXTENSION
def get_img_study(path,dicom):
    
    tag_label_list = ['Date of Imaging',"Image Modality", "Image Identifier","Body Site","Body Structure or Part","Series Date","Study Description","Reason"]
    img_study = {}
    img_study_keywords = ['ContentDate','Modality','StudyInstanceUID','AnatomicRegionSequence',"BodyPartExamined", "SeriesDate", 'StudyDescription',
                    'ReasonForPerformedProcedureCodeSequenceAttribute']
    
    for label in range(0,len(img_study_keywords)):
        if isinstance(tag_label_list[label], list):
            img_study[tag_label_list[label]] = {}
            for word in range(1,len(img_study_keywords[label])):
               
                img_study[tag_label_list[label][0]][tag_label_list[label][word]]= str(get_dicom_tag_value(dicom,img_study_keywords[label][word-1]))
        else:
            #print(get_dicom_tag_value(dicom,tag_label_list[label]))
            img_study[tag_label_list[label]] = str(get_dicom_tag_value(dicom,img_study_keywords[label]))
    
    return img_study

#FUNCTION GETS THE PATIENT DATA FROM THE DCM FILES THAT ARE COMPLIANT WITH MCODE
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
#GETS THE TAGS OF INTEREST FOR THE CT IMAGE (PART OF THE MCODE EXTENSION)
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
#REQUIRED INFORMATION FOR THE MCODE EXTENSION

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
#PET INFORMATION TO EXTRACT THE MCODE INFORMATION EXTRACTION
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
                    #print(pet_tags[label][word-1])
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

############################################################################
# UTILS FOR THE RADIOMICS AND DOSIOMICS EXTRACTION
# NRRD CONVERSION, DOSE MAP EXTRACTION AND RT STRUCTURES

################################3

#GETS THE FORMAT OF THE MEDICAL IMAGES EITHER DCM, NII OR NRRD

def get_set_images(dir_image_path):
    format_image = os.listdir(dir_image_path)[0].split('.')[-1]
    if format_image=='dcm':
        img_files = sorted([os.path.join(dir_image_path, x) for x in os.listdir(dir_image_path) if '.dcm' in x])
        slices = [dcm.dcmread(j, force=True) for j in img_files]
        try:
            slice_filtered = [i for i in slices if i.Modality in ['CT','CBCT','CBCT_SCAN','MR','PT']]
            return slice_filtered
        except:
#CHECKING THAT IF THE UID IS NOT AVAILABLE THE TAG MODALITY IS ANOTHER WAY TO GET THE TYPE OF DCM IMAGE         
            CT = '1.2.840.10008.5.1.4.1.1.2'
            MR = '1.2.840.10008.5.1.4.1.1.4'
            PT = '1.2.840.10008.5.1.4.1.1.128'
            slice_filtered = [i for i in slices if i[0x0008, 0x0016].value in [CT, MR,PT]]
            return slice_filtered   
    elif format_image=='nii':
        img_files = sorted([os.path.join(dir_image_path, x) for x in os.listdir(dir_image_path) if '.nii' in x])
        return img_files
    elif format_image=='nrrd':
        img_files = sorted([os.path.join(dir_image_path, x) for x in os.listdir(dir_image_path) if '.nrrd' in x])
        return img_files
    else:
        return []

            
   

#IT GETS THE MASKS WHEN THEY ARE SAVED IN NII.GZ
def get_set_masks(dir_mask_path,ROI):
    
    mask_files = sorted([os.path.join(dir_mask_path, x) for x in os.listdir(dir_mask_path) if '.nii.gz' in x])

    return


#GET THE RD DCM FILE
def get_set_RD(dir_image_path):
    format_image = os.listdir(dir_image_path)[0].split('.')[-1]
    if format_image=='dcm':
        img_files = sorted([os.path.join(dir_image_path, x) for x in os.listdir(dir_image_path) if '.dcm' in x])
        slices = [dcm.dcmread(j, force=True) for j in img_files]
        try:
            slice_filtered = [i for i in slices if i.Modality=='RTDOSE']
            return slice_filtered
        except:
            RT = '1.2.840.10008.5.1.4.1.1.481.2'
            slice_filtered = [i for i in slices if i[0x0008, 0x0016].value==RT]
            return slice_filtered   
    elif format_image=='nii':
        img_files = sorted([os.path.join(dir_image_path, x) for x in os.listdir(dir_image_path) if '.nii' in x])
        return img_files
    elif format_image=='nrrd':
        img_files = sorted([os.path.join(dir_image_path, x) for x in os.listdir(dir_image_path) if '.nrrd' in x])
        return img_files


#THIS FUNCTION SAVES THE RT STRUCTURE FROM DCM TO NRRD 3D FILE
#FUNCTION TO GET THE CONTOUR FROM THE ROI_NAME 
#RECEIVES THE RS DICOM FILE (RT STRUCTURE) PATH
#IMAGE FILES (SET OF NP ARRAYS OF THE IMAGES)
def save_RT_tructure_dcm_as_nrrd(ROI_name, RS_file_path,image_files, RS_save_path):
    RS = dcm.read_file(RS_file_path)
    slices = image_files.copy()
    contour_coords = []

    roi_contour_seq = None
    for i, seq in enumerate(RS.StructureSetROISequence):
        
        if seq.ROIName == ROI_name:
            roi_seq = RS.ROIContourSequence[i]
            
            roi_contour_seq = roi_seq.ContourSequence
         
    slice_numbers = set()
 
    slice_to_contour_seq = dict() # maps slice # to structure contour sequence
    slices_id = []
    for contour_seq in roi_contour_seq:
        # uid of slice that this contour appears on
        slice_uid = contour_seq.ContourImageSequence[0].ReferencedSOPInstanceUID
        # find that corresponding slice in the files

        for i, ct_file in enumerate(image_files):
            if ct_file.SOPInstanceUID == slice_uid:
            
                slice_to_contour_seq[i] = contour_seq
                slices_id.append(i)    
   
    pixel_spacing = (float(ds0.PixelSpacing[0]),float(ds0.PixelSpacing[1]),float(ds0.SliceThickness)) #DESIRE PIXEL SPACING AND VOLUME
    pixels_dimensions = (int(ds0.Rows),int(ds0.Columns),len(dicom_files))  #DESIRE IMAGE SIZE
    mask_full = np.zeros(pixels_dimensions,dtype=ds0.pixel_array.dtype) #SET A BLANK MASK WITH THE SIZE OF THE IMAGE OF REFERENCE
    #USUALLY THE IMAGES ARE REGISTERED HAVING THE SAME SIZE #IF NOT, THE POSITION OF THE CONTOURS SHOULD BE ALIGNED

    for slicei in range(0,len(slices)): #READING THE SLICES 
       
        if slicei in slices_id:
            
            slice_new = np.array(slice_to_contour_seq[slicei].ContourData).reshape((-1,3))
            
            start_x2, start_y2, start_z2, pixel_spacing2 = get_start_position_dcm(image_files)
            coords_px = get_mask_nifti(slice_new,start_x2,start_y2,pixel_spacing2)
            
        else:
            coords_px = None
            
        mask = np.zeros(np.array(ds0.pixel_array).shape)
        if coords_px!=None:
            
            rows,cols = polygon(coords_px[1],coords_px[0])
            mask[rows,cols] = 1            
    
        mask_full[:,:,slicei] = mask        
    
    mask_full_v2= np.swapaxes(mask_full, 0, 1)
    try:
      nrrd.write(RS_save_path+'/seg_'+ROI_name+'.nrrd', mask_full_v2)
    
      print(f"Wrote nrrd file for RT structure {ROI_name} file sef_"+ROI_name+" to {RS_save_path}") 
    except:
      print(f"Failed to write nrrd for RT structure {ROI_name} file sef_"+ROI_name+" to {RS_save_path}")   

#THIS SAVES THE IMAGE NII AS NRRD
def save_img_nii_as_nrrd(set_image,save_path,image_name):
    
    images = sitk.GetArrayFromImage(set_image)
    pixels_dimensions = (int(images.shape[1]),int(images.shape[2]),int(images.shape[0]))
    array_dicom = np.zeros(pixels_dimensions)

    for image in range(0,len(images)):
      # IT CONCATENATES THE LAYER BBY LAYER (SLICE BY SLICE)
      array_dicom[:,:,image] = images[image]
    data_3d = np.swapaxes(array_dicom, 0, 1)
    
    try:
        nrrd.write(save_path+'/'+image_name+'.nrrd', data_3d)  
        print(f"Wrote nrrd file {image_name} to {save_path}")
    except:
        print(f"Failed to write nrrd file {image_name} to {save_path}")


#GET IMAGE FROM DCM TO NRRD
def save_img_dcm_as_nrrd(set_images, dcm_save_path,image_name):

    data_3d = None

    dicom_files = set_images
    ds0= set_images[0]

    pixel_spacing = (float(ds0.PixelSpacing[0]),float(ds0.PixelSpacing[1]),float(ds0.SliceThickness))
    pixels_dimensions = (int(ds0.Rows),int(ds0.Columns),len(dicom_files))
    array_dicom = np.zeros(pixels_dimensions,dtype=ds0.pixel_array.dtype)

    slices = [j for j in set_images]
    slices.sort(key = lambda x: (x.InstanceNumber))
              
    for dicom_file in slices:
        array_dicom[:,:,slices.index(dicom_file)] = dicom_file.pixel_array

    # transpose the data
    data_3d = np.swapaxes(array_dicom, 0, 1)
    
    try:
        nrrd.write(dcm_save_path+'/'+image_name+'.nrrd', data_3d)  
        print(f"Wrote nrrd file {image_name} to {dcm_save_path}")
    except:
        print(f"Failed to write nrrd file {image_name} to {dcm_save_path}")
         
    return

    
#SAVES RT DOSE AS NRRD FILE
#TO DO: ADD THE DATE!

def save_RT_dose_as_nrrd(rt_dose,set_images,dcm_save_path):
    data_3d = None

    dicom_files = set_images
    ds0= dcm.dcmread(set_images[0], force=True)

    pixel_spacing = (float(ds0.PixelSpacing[0]),float(ds0.PixelSpacing[1]),float(ds0.SliceThickness))
    pixels_dimensions = (int(ds0.Rows),int(ds0.Columns),len(dicom_files))
    array_dicom = np.zeros(pixels_dimensions,dtype=ds0.pixel_array.dtype)

    for data in range(0,len(rt_dose)):
        array_dicom[:,:,data] = rt_dose[data]

    # transpose the data
    data_3d = np.swapaxes(array_dicom, 0, 1)
    try:
        nrrd.write(dcm_save_path+'/doseMap.nrrd', data_3d)

        print(f"Wrote Dose Map nrrd file {image_name} to {dcm_save_path}")
    except:
        print(f"Failed to write Dose Map nrrd file {image_name} to {dcm_save_path}")
    
    return
   
#FUNCTION TO GET THE KEYS FOR EACH NAME ROI
def get_ROI_keys(RS_file_path):  #ROI Name different from the PTV
    RS_file = dcm.read_file(RS_file_path)
    contour_keys = RS_file.StructureSetROISequence
    return [str(x.ROIName) for x in contour_keys]

#FUNCTION TO GET THE KEY NAME FOR THE PTV RELATED ELEMENTS
def get_PTV_keys(RS_file_path): 
    ROI_keys = get_ROI_keys(RS_file_path)
    return [x for x in ROI_keys if 'ptv' in x.lower()]

#FUNCTION TO GET THE KEY NAME FOR THE GTV RELATED ELEMENTS

def get_GTV_keys(RS_file_path): 
    ROI_keys = get_ROI_keys(RS_file_path)
    return [x for x in ROI_keys if 'gtv' in x.lower()]

#FUNCTION TO GET THE KEY NAME FOR THE CTV  
#IN CASE THAT THERE ARE MORE THAN ONE CTV KEY (E.G. CTV_5MM) THE RADIOMICS/DOSIOMICS CAN BE EXTRACTED FOR ALL OF THEM
def get_CTV_keys(RS_file_path): 
    ROI_keys = get_ROI_keys(RS_file_path)
    return [x for x in ROI_keys if 'ctv' in x.lower()]
    

def get_mask_nifti(roi_array,start_x,start_y,pixel_spacing):
    '''
    Get the pixel positions (rather than the x,y coords) of the contour array so it can be plotted.
    '''
    x = []
    y = []
    
    for i in range(0,len(roi_array)):
        x.append((roi_array[i][0]/pixel_spacing[0]) - start_x/pixel_spacing[0])
        y.append((roi_array[i][1]/pixel_spacing[1]) - start_y/pixel_spacing[1])
        
    return x, y
    
#THIS FUNCTION GETS THE IMAGE PATIENT POSITIONING FROM THE IMAGE 
def get_start_position_dcm(slices):
    positions = []
    for f in slices:             
        positions.append(f.ImagePositionPatient)
    positions = sorted(positions, key=lambda x: x[-1])
    start_z = positions[0][2]
    start_x = positions[0][0]
    start_y = positions[0][1]
    pixel_spacing = f.PixelSpacing
    
    return start_x, start_y, start_z, pixel_spacing 

#FUNCTION TO RESAMPLE AND RESIZING THE DOSE MAP DISTRIBUTION
#IT RECEIVED THE DOSE MAP (dsDose), number of the slices in the image
#THE SET OF CT ARRAYS (ctArray) in pixel array, and dsCTs (DICOM information image)  
def resample_dose_dist(dsDose,number_slices,dsCTs,ctArray):
    ctArray = np.array(ctArray)
    pixel_spacing = dsCTs[0].PixelSpacing #get pixel spacing [#horizontal pixels, #vertical pixels]
    doseArray   = dsDose.pixel_array * dsDose.DoseGridScaling #Scale the pixel values with the DoseScale values
    assert doseArray.shape[0] == float(dsDose.NumberOfFrames) #check dimension
    #Define new dose spacing 
    doseSpacing = [float(each) for each in dsDose.PixelSpacing] + [float(dsCTs[0].SliceThickness)]
    #Define the dose position of the patients int the dose map
    doseImagePositionPatient = [float(each) for each in dsDose.ImagePositionPatient]
    #Define spacing of the CT
    ctSpacing = [float(each) for each in dsCTs[0].PixelSpacing] + [float(dsCTs[0].SliceThickness)]
    
    resample_set = []
    #resample each slice
    for slicei in doseArray:
        #WE CONVERT EACH SLICE TO AN IMAGE FROM THE ARRAY USING SITK TO RESAMPLE AND RESIZE
        doseImage = sitk.GetImageFromArray(slicei)
        #WE SET THE NEW SPACING
        doseImage.SetSpacing(doseSpacing) 
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(ctSpacing) #set new spacing
        #GET THE CT ARRAY SHAPE WITH NUMPY
        resampler.SetSize(ctArray.shape[::-1]) # SimpleITK convention: [HEIGHT,WIDTH,Slices], numpy convention: [Slices,HEIGHT,WIDTH]
        resampled_image = resampler.Execute(doseImage) #resample the dose slice
        #WE GET AN ARRAY FROM THE IMAGE USING SITK
        doseArrayResampled = sitk.GetArrayFromImage(resampled_image)
        
        ctImagePositionPatientMin = [float(each) for each in dsCTs[0].ImagePositionPatient]
        ctImagePositionPatientMax = [float(each) for each in dsCTs[-1].ImagePositionPatient]
        #re escale and shift the pixels position to have the equivalent dimension
        dx, dy, dz = ((np.array(doseImagePositionPatient) - np.array(ctImagePositionPatientMax)) / np.array(ctSpacing)).astype(int)
    
        doseArrayResampled = shift(doseArrayResampled, (dy, dx)) #shift using the sITK format
        resample_set.append(doseArrayResampled) 
        
    doseImage2 = sitk.GetImageFromArray(np.array(resample_set)) #transform to image again

    doseImage2.SetSpacing(ctSpacing) #set spacing in 3D with the same slice thickness
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(ctSpacing)
    
    resampler.SetSize(ctArray.shape[::-1])
    resampled_image2 = resampler.Execute(doseImage2)
    doseArrayResampled2 = sitk.GetArrayFromImage(resampled_image2)
    
    return doseArrayResampled2
      

#FUNCTION TO VISUALIZE DOSE MAP+CT MEDICAL IMAGE + SEGMENTATION
def data_visualize(dose_img,slices,seg):
    doseArrayResampled = sitk.GetArrayFromImage(dose_img)
    dose_min = np.min(doseArrayResampled)
    dose_max = np.max(doseArrayResampled)
    locator = ticker.MaxNLocator(11,min_n_ticks=10)
    dose_levels = locator.tick_values(dose_min,dose_max)
    plt.imshow(slices[75].pixel_array, cmap='gray')
    plt.imshow(doseArrayResampled[75],cmap=plt.cm.plasma,alpha=.5)
    plt.contour(sitk.GetArrayFromImage(seg)[75,:,:], colors='red', linewidths=0.4)
    return
    

def get_settings(param_files):
    return


    
#GET RADIOMICS FUNCTION, WITH METHOD AS INPUT TO GET THE FEATURES SEGMENTATION BASED OR VOXEL BASED
#FOR VOXEL BASED IT RETURNS THE FEATURE MAP.
def get_radiomics(method,imageName,maskName,ROIName,path_radiomics):
    
    software_version = radiomics.__version__
    parameters = os.path.abspath(os.path.join('Params.yaml'))
                          
    #params = os.path.join(dataDir, "examples", "exampleSettings", "Params.yaml")
    extractor = featureextractor.RadiomicsFeatureExtractor(parameters)
    featureClasses = getFeatureClasses()
    #featureVector = extractor.execute(image,image)
    
    #TO DO: reshape the feature map
    if method=='voxel':                
        featureVector = extractor.execute(imageName, maskName, voxelBased=True)
        for key, val in six.iteritems(featureVector):
            if isinstance(val, sitk.Image):  # Feature map
                sitk.WriteImage(val, path_radiomics+'/voxel_based/'+key + '.nrrd', True) #SAVE THE FEATURE MAP
                print("Stored feature %s in %s at 'radiomics/voxel_based/'" % (key, key + ".nrrd"))
                    
    else: #Get RADIOMICS SEGMENTATION BASED
        featureVector = extractor.execute(imageName, maskName)
        feature_dict = {'ROI Name': ROIName}
        parameters_dict= {}
    
        features_names = []
        keys_features = list(extractor.enabledFeatures.keys())
        for i,featureName in enumerate(featureVector.keys()):
            if featureName.split('_')[1] in keys_features:
                feature_dict[featureName] = float(featureVector[featureName])
        with open(path_radiomics+'seg_'+ROIName+"_radiomics.json", "w") as outfile: 
        #FEATURE JSON FILE SAVE IN THE RADIOMICS FOLDER PER PATIENT AND IMAGE STUDIED
                json.dump(feature_dict, outfile,indent=4)
    return  

#FUNCTION TO EXTRACT THE DOSIOMICS INFORMATION FROM A GIVEN ROI NAME
def get_dosiomics(method,imageName,maskName,ROIName,path_dosiomics):
    
    software_version = radiomics.__version__
    parameters = os.path.abspath(os.path.join('Params.yaml'))
                          
    #params = os.path.join(dataDir, "examples", "exampleSettings", "Params.yaml")
    extractor = featureextractor.RadiomicsFeatureExtractor(parameters)
    featureClasses = getFeatureClasses()
    #featureVector = extractor.execute(image,image)
    
    #TO DO: reshape the feature map
    if method=='voxel':                
        featureVector = extractor.execute(imageName, maskName, voxelBased=True)
        for key, val in six.iteritems(featureVector):
            if isinstance(val, sitk.Image):  # Feature map
                sitk.WriteImage(val, path_radiomics+'/voxel_based/'+key + '.nrrd', True) #SAVE THE FEATURE MAP
                print("Stored feature %s in %s at 'dosiomics/voxel_based/'" % (key, key + ".nrrd"))
                    
    else: #Get DOSIOMICS SEGMENTATION BASED
        featureVector = extractor.execute(imageName, maskName)
        feature_dict = {'ROI Name': ROIName}
        parameters_dict= {}
    
        features_names = []
        keys_features = list(extractor.enabledFeatures.keys())
        for i,featureName in enumerate(featureVector.keys()):
            if featureName.split('_')[1] in keys_features:
                feature_dict[featureName] = float(featureVector[featureName])
        with open(path_dosiomics+'Seg_'+ROIName+"_dosiomics.json", "w") as outfile: 
        #FEATURE JSON FILE SAVE IN THE RADIOMICS FOLDER PER PATIENT AND IMAGE STUDIED
                json.dump(feature_dict, outfile,indent=4)
    return 
    
def get_dosimetricfeatures():
  return
    
def get_data_from_csv(path):
    return

def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError
    
def get_data_from_documents(path,type):
    if type=='csv':
        get_data_from_csv(path)
        
    return
    
###############
#  SAVE JSON FILES
###############
    
#SAVE THE JSON FILES FOR A GIVEN PATH TO SAVE (PATH_SAVE), FOR A GIVEN DICOM IMAGE
def save_JSON_attributes(path_save,path,dicom):

    img_study_dict = get_img_study(path,dicom)
    with open(path_save+"/image_study_attributes.json", "w") as outfile: 
        json.dump(img_study_dict, outfile,indent=4)
    
    print('---')
    pat_data_dict = get_pat_data_in_dcm(path,dicom)
    print('---')
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
            


###################
  #  UTILS TO CREATE AND SAVE THE PATHS FOR EACH PATIENT
############

#CREATE THE FOLDER FOR THE GIVEN PATIENT USING THE MRN
def create_patient_folder(path_save,name_patient_mrn):
    
    folder_path = path_save+"/patient_"+name_patient_mrn  #SET THE PATH FOR THE PATIENT #CHANGE THE PATH IF NECESSARY
    if os.path.isdir(folder_path):  
        print(f"The folder '{folder_path}' exists.") 
    else:
        
        print(f"The folder '{folder_path}' does not exist") #CREATION OF THE FOLDER
        print(f"Creating folder for Patient '{name_patient_mrn}'...")
        os.mkdir(folder_path)
        print(f"Folder created sucessfully")
        #return folder_path
    

#IT COPIES THE ORIGINAL DIRECTORY NAMES WITHOUR THE FILES IN IT, ONLY THE LINK PATHS OF THE FOLDERS. 
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
    #while len(folders)!=0:
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

#FUNCTION TO PERFOM THE EXTRACTION DATA OF ALL THE PATIENTS
def clone_folders_per_patient_v1(path_patient,path_save):
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
    #folder_path_images = copy_paths(path_patient)
    
    #set_images = get_set_images(dir_imag_path)
    #save_JSON_attributes(path_save,path,dicom)

def get_radiomics_dosiomics(imageName,maskName,ROIname,path_radiomics,path_dosiomics):
    get_radiomics('segmentation',imageName,maskName,ROIname,path_radiomics)
    get_dosiomics('segmentation',imageName,maskName,ROIname,path_dosiomics) 
    return
#########################

if __name__ == "__main__":
    
    #SPECIFYING PATH IN THE CODE
    path_patient = '/mnt/iDriveShare/Kayla/CBCT_images/kayla_extracted/632'
    path_save  = 'TRY'
    path_RS_file = '/mnt/iDriveShare/Kayla/CBCT_images/kayla_extracted/'
    
    #CREATING FOLDER FOR THE SPECIFIC PATIENT
    new_path_save = clone_folders_per_patient_v1(path_patient,path_save)
 
    #GETTING SET OF IMAGES FROM THE PATH OF THE DATA
    #dicom = dcm.dcmread(set_images[0], force=True)
       
    folder_path_images = sorted(copy_paths(path_patient))
    new_folder_path_images = sorted(copy_paths(new_path_save))
   
    for folder_image in folder_path_images:
         #create_folder(folder_path)
         print(folder_image)
         name_folder = folder_image.split('/')[-1]
         for new_folder_name in new_folder_path_images:
             new_name_folder = new_folder_name.split('/')[-1]
             if name_folder==new_name_folder:
                 set_images = get_set_images(folder_image)
                 save_JSON_attributes(new_folder_name,folder_image,set_images[0])
                 try:
                     image_name = set_images[0].ContentDate+'_'+set_images[0].Modality
                 except:
                     image_name = set_images[0].AcquisitionDate+'_'+set_images[0].Modality
                 save_img_dcm_as_nrrd(set_images, new_folder_name,image_name)
      #   for ROI in rois:
     #        create_folder(folder_image+'/radiomics/'+ROI+'_radiomics')
    #seg = sitk.ReadImage('seg_Parotid_R.nrrd',imageIO='NrrdImageIO')
             
    #path_radiomics = '/radiomics/'+path_image+'/'+ROI+'_radiomics' 
             

 