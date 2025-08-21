
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
from dicompylercore import dicomparser, dvh, dvhcalc

import SimpleITK as sitk
import six
import nrrd

from radiomics import featureextractor, getFeatureClasses
from skimage.draw import polygon

from scipy.ndimage import shift
from matplotlib import ticker
import sys
<<<<<<< HEAD
sys.path.append('../longiDICOM/code')
import RD_tools
from RD_tools import find_dose_file, get_dose_in_gy, get_dose_xyz, get_dose_spacing, resample_dose_map_3D, resize_dose_map_3D, get_struct_dose_values, create_binary_mask, extract_dose_values
from rs_tools import find_RS_file, find_ROI_names
=======
sys.path.append('./longiDICOM/code')
>>>>>>> origin/master
from mcode_utils import save_JSON_attributes
from dicompylercore import dicomparser, dvh, dvhcalc


############################################################################
# UTILS FOR THE RADIOMICS AND DOSIOMICS EXTRACTION
# NRRD CONVERSION, DOSE MAP EXTRACTION AND RT STRUCTURES
############################################################################

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
        print('************* Error in path: No medical images found *************')
        return []

#IT GETS THE MASKS WHEN THEY ARE SAVED IN NII.GZ
def get_set_masks_nii(dir_mask_path,ROI):
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
        
def get_set_RS(dir_image_path):
    format_image = os.listdir(dir_image_path)[0].split('.')[-1]
    if format_image=='dcm':
        img_files = sorted([os.path.join(dir_image_path, x) for x in os.listdir(dir_image_path) if '.dcm' in x])
        slices = [dcm.dcmread(j, force=True) for j in img_files]
        try:
            slice_filtered = [i for i in slices if i.Modality=='RTSTRUCT']
            return slice_filtered
        except:
            RT = '1.2.840.10008.5.1.4.1.1.481.3'
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
   
    ds0 =slices[0]
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
      print("--------------------------------------------------------")
      print(f"-   Wrote nrrd file for RT structure {ROI_name} file sef_"+ROI_name+" to " + f"{RS_save_path}"+"   -")
      print("--------------------------------------------------------") 
    except:
      print("--------------------------------------------------------")
      print(f"-   Failed to write nrrd for RT structure {ROI_name} file sef_"+ROI_name+" to " + f"{RS_save_path}"+"   -")
      print("--------------------------------------------------------")   

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
        print("--------------------------------------------------------")
        print(f"-   Wrote nrrd file {image_name} to {save_path}   -")
        print("--------------------------------------------------------")
    except:
        print("--------------------------------------------------------")
        print(f"-   Failed to write nrrd file {image_name} to {save_path}   -")
        print("--------------------------------------------------------")


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
        print("--------------------------------------------------------")
        print(f"-   Wrote nrrd file {image_name} to {dcm_save_path}   -")
        print("--------------------------------------------------------")
    except:
        print("--------------------------------------------------------")
        print(f"-   Failed to write nrrd file {image_name} to {dcm_save_path}   -")
        print("--------------------------------------------------------")
         
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

def get_mask_nifti_dose(roi_array,start_x,start_y,start_z,pixel_spacing):
    '''
    Get the pixel positions (rather than the x,y coords) of the contour array so it can be plotted.
    '''
    x = []
    y = []
    z = ((roi_array[0][2]/pixel_spacing[2]) - (start_z/pixel_spacing[2]))
    for i in range(0,len(roi_array)):
        x.append((roi_array[i][0]/pixel_spacing[0]) - start_x/pixel_spacing[0])
        y.append((roi_array[i][1]/pixel_spacing[1]) - start_y/pixel_spacing[1])
        
    return x, y, z
    
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

def get_dosimetric_from_RD_data(ROI_name,RT_file,RS_file):
    #dose_file = 
    # Script that tabulates basic DVH statistics for each structure
    roi_contour_seq = None
    for i, seq in enumerate(RT_file.StructureSetROISequence):
        if seq.ROIName == ROI_name:
            dvh_seq = dose_file.DVHSequence[i]
            min_dose = dvh_seq.DVHMinimumDose
            max_dose = dvh_seq.DVHMaximumDose
            mean_dose = dvh_seq.DVHMeanDose
    return min_dose, mean_dose, max_dose

def get_dosimetric_factors_dvh(index_roi,RS_file_path,RD_file_path,factors = ['D0', 'D100', 'D50','mean']):
    rtdose = dicomparser.DicomParser(RD_file_path)
    calc_dvh = dvh.DVH.from_dicom_dvh(rtdose.ds, index_roi)
    result_factors = []
    for factor in factors:
        if factor=='mean':
            result_factors.append(calc_dvh.mean)
        else:
            dose_f = calc_dvh.statistic(factor).value
            result_factors.append(dose_f)      
    return result_factors, factors


def search_check_keys(roi_names,ROI_name):
    stri = ''
    for roi in roi_names:
        if ROI_name.lower() in roi.lower():
            stri=roi
        else:
            continue
    return stri
    
def all_dosimetric_features_dvh_json(RS_file_path, RD_file_path,path_dosiomics, factors):
    dp = dicomparser.DicomParser(RS_file_path)
    # i.e. Gets a dict of structure information
    structures = dp.GetStructures()
    elements_structures = [[key] + list(inner.values())[0:2] for key, inner in structures.items()]
    roi_names = np.array(elements_structures)[:,2]

    for ROI_names in sys.argv[1:]:
        if ROI_names.lower() == "all":
            for ROI_name in roi_names:
                index_roi = list(roi_names).index(ROI_name) + 1
                try:
                    results_dvh,factors =  get_dosimetric_factors_dvh(index_roi,RS_file_path,RD_file_path,factors)
                    feature_dict = {'ROI Name': ROI_name, 'units' : 'GY'}
        
                    for i in range(0,len(results_dvh)):
                        feature_dict[factors[i]] = float(results_dvh[i])
                        
                    print(feature_dict)
                    print('\n')
                    try:
                        with open(path_dosiomics+'_'+ROI_name+'_dosimetric.json', 'w') as file:
                            json.dump(feature_dict, file, indent=4)
                        print('------------'+ ROI_name+' JSON file with DOSIMETRIC factors were saved correctly ------------')
                        print('\n')
                    except:
                        print('---------------- ERROR ERROR ERROR check files and path -------------------')
                except:
                    print('---------' + ROI_name+ ' does not have DVH information ----------------')
                    continue
               
        else:
            search_key = search_check_keys(roi_names,ROI_names)
            if search_key=='':
                print('---------------- ERROR in ROI name. Check the labels in the RS file and input -------')
                sys.exit()
            
            index_roi = list(roi_names).index(ROI_names) + 1
            results_dvh,factors =  get_dosimetric_factors_dvh(index_roi,RS_file_path,RD_file_path,factors)
            feature_dict = {'ROI Name': ROI_names, 'units' : 'GY'} #check the units from the RD file!
            for i in range(0,len(results_dvh)):
                feature_dict[factors[i]] = float(results_dvh[i])
            
            print(feature_dict)
            print('\n')
            try:
                with open(path_dosiomics+'_'+ROI_names+'_dosimetric.json', 'w') as file:
                    json.dump(feature_dict, file, indent=4)
                print('----------'+ ROI_names+' JSON file with DOSIMETRIC factors were saved correctly ------------')
                print('\n')
            except:
                print('--------------------- ERROR ERROR ERROR check files and path -------------------')
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
        try:
            with open(path_dosiomics+'Seg_'+ROIName+"_dosiomics.json", "w") as outfile: #FEATURE JSON FILE SAVE IN THE RADIOMICS FOLDER PER PATIENT AND IMAGE STUDIED
                json.dump(feature_dict, outfile,indent=4)
            print('----------- JSON file with DOSIOMICS features were saved correctly ------------')
        except:
            print('----------- ERROR ERROR ERROR check files and path -------------------')
    return 

def get_dosimetric_features(seg_roi,doseArray):
    seg_2 = np.array(seg_roi,dtype=bool)
    dose = doseArray[seg_2]
    mean_dose, median_dose, max_dose, min_dose = np.mean(dose),np.median(dose),np.max(dose),np.min(dose) 
    return  mean_dose, median_dose, max_dose, min_dose

def get_radiomics_dosiomics(imageName,maskName,ROIname,path_radiomics,path_dosiomics):
    get_radiomics(method,imageName,maskName,ROIname,path_radiomics)
    get_dosiomics(method,imageName,maskName,ROIname,path_dosiomics) 
    return
    
def get_only_radiomics(imageName,maskName,ROIname,path_radiomics,method):
    get_radiomics(method,imageName,maskName,ROIname,path_radiomics)
    return

def get_only_dosiomics(imageName,maskName,ROIname,path_dosiomics,method):
    get_dosiomics(method,imageName,maskName,ROIname,path_dosiomics)
    return

def create_ROI_folders(RS_file_path,folder_path_radiomics):
    ROI_names = get_ROI_keys(RS_file_path)
    for ROI in ROI_names:
        new_folder = folder_path_radiomics+'/'+ROI+'_radiomics'
        create_folder(new_folder)
        


      
