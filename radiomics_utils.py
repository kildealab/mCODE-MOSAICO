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
        with open(path_dosiomics+'Seg_'+ROIName+"_dosiomics.json", "w") as outfile: 
        #FEATURE JSON FILE SAVE IN THE RADIOMICS FOLDER PER PATIENT AND IMAGE STUDIED
                json.dump(feature_dict, outfile,indent=4)
    return 

def get_radiomics_dosiomics(imageName,maskName,ROIname,path_radiomics,path_dosiomics):
    get_radiomics('segmentation',imageName,maskName,ROIname,path_radiomics)
    get_dosiomics('segmentation',imageName,maskName,ROIname,path_dosiomics) 
    return

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

def create_ROI_folders(RS_file_path,folder_path_radiomics):
    ROI_names = get_ROI_keys(RS_file_path)
    for ROI in ROI_names:
        new_folder = folder_path_radiomics+'/'+ROI+'_radiomics'
        create_folder(new_folder)
      
