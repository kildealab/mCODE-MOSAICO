
from pydicom.uid import generate_uid
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ImplicitVRLittleEndian
from rt_utils.image_helper import get_contours_coords
from rt_utils.utils import ROIData, SOPClassUID

import sys
from typing import List
from pydicom.dataset import Dataset
from pydicom.filereader import dcmread
import warnings
from rt_utils import RTStructBuilder
import datetime
import numpy as np
import json
import pydicom
import pydicom as dcm

import matplotlib.pyplot as plt
from skimage.draw import polygon
import random
import SimpleITK as sit
import os


import radiomics_utils
from radiomics_utils import *


'''snippet code from rt-utils. I modified it just a little bit'''
'''CREDITS TO RT-UTILDS. FOR MORE DETAILS PLEASE SEE THEIR GITHUB REPOSITORY'''

def create_contour_sequence(contours_coords, series_data) -> Sequence:
    """
    Iterate through each slice of the mask
    For each connected segment within a slice, create a contour
    """
    contour_sequence = Sequence()

    #contours_coords = get_contours_coords(roi_data, series_data)

    for series_slice, slice_contours in zip(series_data, contours_coords):
        for contour_data in slice_contours:
            contour = create_contour(series_slice, contour_data)
            contour_sequence.append(contour)

    return contour_sequence

def create_structure_set_roi(roi_data):
    # Structure Set ROI Sequence: Structure Set ROI 1
    structure_set_roi = Dataset()
    structure_set_roi.ROINumber = roi_data.number
    structure_set_roi.ReferencedFrameOfReferenceUID = frame_of_reference_uid
    structure_set_roi.ROIName = roi_data.name
    structure_set_roi.ROIDescription = roi_data.description
    #structure_set_roi.ROIGenerationAlgorithm = roi_data.roi_generation_algorithm
    return structure_set_roi


def create_contour_image_sequence(series_data):
    contour_image_sequence = Sequence()
    for series in series_data:
        #print(series)
        contour_image = Dataset()
        contour_image.ReferencedSOPClassUID = series.SOPClassUID
        contour_image.ReferencedSOPInstanceUID = series.SOPInstanceUID
        contour_image_sequence.append(contour_image)

    return contour_image_sequence

def create_rtstruct_dataset(series_data) -> FileDataset:
    ds = generate_base_dataset()
    add_study_and_series_information(ds, series_data)
    add_patient_information(ds, series_data)
    add_refd_frame_of_ref_sequence(ds, series_data)
    return ds

def create_frame_of_ref_study_sequence(series_data) -> Sequence:
    reference_ds = series_data[0]  # All elements in series should have the same data
    rt_refd_series = Dataset()
    rt_refd_series.SeriesInstanceUID = reference_ds.SeriesInstanceUID
    rt_refd_series.ContourImageSequence = create_contour_image_sequence(series_data)

    rt_refd_series_sequence = Sequence()
    rt_refd_series_sequence.append(rt_refd_series)

    rt_refd_study = Dataset()
    rt_refd_study.ReferencedSOPClassUID = SOPClassUID.DETACHED_STUDY_MANAGEMENT
    rt_refd_study.ReferencedSOPInstanceUID = reference_ds.StudyInstanceUID
    rt_refd_study.RTReferencedSeriesSequence = rt_refd_series_sequence

    rt_refd_study_sequence = Sequence()
    rt_refd_study_sequence.append(rt_refd_study)
    return rt_refd_study_sequence

def generate_base_dataset() -> FileDataset:
    file_name = "rt-utils-struct"
    file_meta = get_file_meta()
    ds = FileDataset(file_name, {}, file_meta=file_meta, preamble=b"\0" * 128)
    add_required_elements_to_ds(ds)
    add_sequence_lists_to_ds(ds)
    return ds


def get_file_meta() -> FileMetaDataset:
    file_meta = FileMetaDataset()
    file_meta.FileMetaInformationGroupLength = 202
    file_meta.FileMetaInformationVersion = b"\x00\x01"
    file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = SOPClassUID.RTSTRUCT
    file_meta.MediaStorageSOPInstanceUID = (
        generate_uid()
    )  
    file_meta.ImplementationClassUID = SOPClassUID.RTSTRUCT_IMPLEMENTATION_CLASS
    return file_meta


def add_required_elements_to_ds(ds: FileDataset):
    dt = datetime.datetime.now()
    # Append data elements required by the DICOM standarad
    ds.SpecificCharacterSet = "ISO_IR 100"
    ds.InstanceCreationDate = dt.strftime("%Y%m%d")
    ds.InstanceCreationTime = dt.strftime("%H%M%S.%f")
    ds.StructureSetLabel = "RTstruct"
    ds.StructureSetDate = dt.strftime("%Y%m%d")
    ds.StructureSetTime = dt.strftime("%H%M%S.%f")
    ds.Modality = "RTSTRUCT"
    ds.Manufacturer = ""
    ds.ManufacturerModelName = "rt-utils"
    ds.InstitutionName = ""
    # Set the transfer syntax
    ds.is_little_endian = True
    ds.is_implicit_VR = True
    # Set values already defined in the file meta
    ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID

    ds.ApprovalStatus = "UNAPPROVED"


def add_sequence_lists_to_ds(ds: FileDataset):
    ds.StructureSetROISequence = Sequence()
    ds.ROIContourSequence = Sequence()
    ds.RTROIObservationsSequence = Sequence()


def add_study_and_series_information(ds: FileDataset, series_data):
    reference_ds = series_data[0]  # All elements in series should have the same data
    ds.StudyDate = reference_ds.StudyDate
    ds.SeriesDate = getattr(reference_ds, "SeriesDate", "")
    ds.StudyTime = reference_ds.StudyTime
    ds.SeriesTime = getattr(reference_ds, "SeriesTime", "")
    ds.StudyInstanceUID = reference_ds.StudyInstanceUID
    ds.SeriesInstanceUID = generate_uid()  # TODO: find out if random generation is ok
    ds.StudyID = reference_ds.StudyID
    ds.SeriesNumber = "1"  # TODO: find out if we can just use 1 (Should be fine since its a new series)


def add_patient_information(ds: FileDataset, series_data):
    reference_ds = series_data[0]  # All elements in series should have the same data
    ds.PatientName = getattr(reference_ds, "PatientName", "")
    ds.PatientID = getattr(reference_ds, "PatientID", "")


def add_refd_frame_of_ref_sequence(ds: FileDataset, series_data):
    refd_frame_of_ref = Dataset()
    refd_frame_of_ref.FrameOfReferenceUID =  getattr(series_data[0], 'FrameOfReferenceUID', generate_uid())
    refd_frame_of_ref.RTReferencedStudySequence = create_frame_of_ref_study_sequence(series_data)
    # Add to sequence
    ds.ReferencedFrameOfReferenceSequence = Sequence()
    ds.ReferencedFrameOfReferenceSequence.append(refd_frame_of_ref)
    
def create_frame_of_ref_study_sequence(series_data):
    reference_ds = series_data[0]  # All elements in series should have the same data
    rt_refd_series = Dataset()
    rt_refd_series.SeriesInstanceUID = reference_ds.SeriesInstanceUID
    rt_refd_series.ContourImageSequence = create_contour_image_sequence(series_data)

    rt_refd_series_sequence = Sequence()
    rt_refd_series_sequence.append(rt_refd_series)

    rt_refd_study = Dataset()
    rt_refd_study.ReferencedSOPClassUID = SOPClassUID.DETACHED_STUDY_MANAGEMENT
    rt_refd_study.ReferencedSOPInstanceUID = reference_ds.StudyInstanceUID
    rt_refd_study.RTReferencedSeriesSequence = rt_refd_series_sequence

    rt_refd_study_sequence = Sequence()
    rt_refd_study_sequence.append(rt_refd_study)
    return rt_refd_study_sequence

def get_z_positions(CT_path):
    positions = []
    for f in [file for file in os.listdir(CT_path) if 'CT' in file]:
        d = dcm.dcmread(CT_path+'/'+f)
        positions.append(d.ImagePositionPatient) 
 
    positions = sorted(positions, key=lambda x: x[-1])
    return np.array(positions)[:,2]

def series_conturss(zs,contour,data):
    series_contours = []
    for i in range(0,len(zs)):
        xys= []
        for p in data[contour]:
            if p[2]==zs[i]:
                xys.append(p) 
        xys2 = [[x for lst in xys for x in lst]]
        series_contours.append(xys2)
        
    return series_contours

    
def create_contour(series_slice: Dataset, contour_data: np.ndarray) -> Dataset:
    #print(contour_data)
    contour_image = Dataset()
    contour_image.ReferencedSOPClassUID = series_slice.SOPClassUID
    contour_image.ReferencedSOPInstanceUID = series_slice.SOPInstanceUID

    # Contour Image Sequence
    contour_image_sequence = Sequence()
    contour_image_sequence.append(contour_image)

    contour = Dataset()
    contour.ContourImageSequence = contour_image_sequence
    contour.ContourGeometricType = ("CLOSED_PLANAR")
    contour.NumberOfContourPoints = (len(contour_data) / 3)  # Each point has an x, y, and z value

    # Rounds ContourData to 10 decimal places to ensure it is <16 bytes length, as per NEMA DICOM standard guidelines.
    contour.ContourData = contour_data

    return contour
    

def plot_dcm_contour_with_images(dc,roi_number,ct_files):
# Get the list of structures from the RTSS file
    roi_sequences = dc.StructureSetROISequence
# Print the list of ROI names and corresponding ROI numbers
# Prompt the user to select a ROI

# Get the Contour Sequence for the selected ROI
# Contour Sequence = Sequence of Contours (per slice)
    roi_contour_seq = None
    for roi_seq in dc.ROIContourSequence:
        if roi_seq.ReferencedROINumber == roi_number:
            roi_contour_seq = roi_seq.ContourSequence

    contour_keys = dc.StructureSetROISequence
    slice_numbers_list = set()
    slice_numbers = set()
    slice_to_contour_seq = dict() # maps CT slice # to structure contour sequence
    
    for contour_seq in roi_contour_seq:
        # uid of slice that this contour appears on
        slice_uid = contour_seq.ContourImageSequence[0].ReferencedSOPInstanceUID
        # find that corresponding slice in our CT files
    #print(slice_uid)
        for i, ct_file in enumerate(ct_files):
            if ct_file.SOPInstanceUID == slice_uid:
                slice_numbers.add(i)
                slice_to_contour_seq[i] = contour_seq
            
    # sort and choose a random CT slice (only between our contour limits)
    slice_numbers_list = list(slice_numbers)
    slice_numbers_list.sort()

    for number_file in range(0,len(ct_files)):

        selected_contour_seq = slice_to_contour_seq[number_file]

    # ContourData = Sequence of (x,y,z) triplets defining a contour
    # "-1" in reshape # of rows not specified and determined 
    # based on size of ContourData and number of columns (= 3)
    # Ultimately, # of rows = slice count for contour and 3 columns = (x,y,z)
    
        if selected_contour_seq.ContourData==None:
            continue
        else:
            
            ct_file = ct_files[number_file]
            pixel_spacing = ct_file.PixelSpacing
            ct_pixel_array = ct_file.pixel_array
    
            contour_data = np.array(selected_contour_seq.ContourData).reshape((-1, 3))
            non_nan_mask = ~np.isnan(contour_data)
            contour_data_2 = contour_data[non_nan_mask]
            if len(contour_data_2)==0:
                continue
            else:
                image_pos = ct_file.ImagePositionPatient
                selected_slice = np.zeros(np.array(ct_pixel_array).shape) # array of 0 size of CT

                contour_data[:,1] /= pixel_spacing[1] # y scaling
                contour_data[:,1] -= (image_pos[1] / pixel_spacing[1]) # y shift
                contour_data[:,0] /= pixel_spacing[0] # x scaling
                contour_data[:,0] -= (image_pos[0] / pixel_spacing[0]) # x shift
                rows, cols = polygon(contour_data[:,1], contour_data[:,0]) 
        
                selected_slice[rows, cols] = 1 # set only contour area to 1
            
                fig, ax = plt.subplots()
                ax.imshow(ct_pixel_array, cmap=plt.cm.gray)
                ax.plot(contour_data[:,0],contour_data[:,1],'r')
                plt.savefig('image_slice_'+str(number_file)+'.png')
                plt.show()
                

def get_info(path_dicom_patient,modality):
    #INSERT MODALITY NAME 'MR', 'PET','CT', or if the files are saved with a number '1'. PLEASE TYPE '1' if so
    path_files = os.path.join(path_patient, path_dicom_patient)
    files = [x for x in os.listdir(path_files) if modality in x]
    
    return path_files,files


def save_RS_dicom(contour_name,patient_id,output_name,path_dcm,path_contour,path_to_save):    
    f = open(path_contour+contour_name+".json")
    data = json.load(f) #LOADING THE CONTOUR PREVIOUSLY SAVED IN JSON FORMAT
    f.close()

    path_files, files = get_info(path_dicom_patient,modality)

    series_data = []
    for file in files:
        file_dcm = dcm.read_file(path_dcm+"/"+file)
        series_data.append(file_dcm)
    
    series_data.sort(key=lambda x: x.ImagePositionPatient[2])
    contour_image_sequence = create_contour_image_sequence(filess)
    ds = create_rtstruct_dataset(series_data)
    frame_of_reference_uid = ds.ReferencedFrameOfReferenceSequence[-1].FrameOfReferenceUID 
 
    rtstruct = RTStructBuilder.create_new(dicom_series_path=path_dcm)
    zs = get_z_positions(path_dcm)
    series_contours = series_conturss(zs,contour_name,data)
    contour_dcm = create_contour_sequence(series_contours, series_data ) 
    roi_contour = Dataset()
    roi_contour.ROIDisplayColor = [255,0,255]
    roi_contour.ContourSequence = contour_dcm
    roi_contour.ReferencedROINumber = str(1)
    file_path = output_name+'.dcm'
    file_path = file_path if file_path.endswith(".dcm") else file_path + ".dcm"
    
    file = open(file_path, "w")
    print('===============================================')
    print("         Writing file into ", file_path)
    print('===============================================')

    structure_set_roi = Dataset()
    structure_set_roi.ROINumber = 1
    structure_set_roi.ReferencedFrameOfReferenceUID = frame_of_reference_uid
    structure_set_roi.ROIName = contour_name
    structure_set_roi.ROIDescription = ""
    structure_set_roi.ROIGenerationAlgorithm = ""
    ds.StructureSetROISequence.append(structure_set_roi)
    ds.ROIContourSequence.append(roi_contour)
    ds.save_as(file_path)


def save_dcm(data_directory,path_to_save,image_3D,modality):    
    
    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(data_directory)
    
    if not series_IDs:
        print('ERROR: given directory "'+ data_directory+ '" does not contain a DICOM series.')
        sys.exit(1)
        
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(data_directory, series_IDs[0])

    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)
    series_reader.MetaDataDictionaryArrayUpdateOn()
    series_reader.LoadPrivateTagsOn()
    image3D = series_reader.Execute()

    filtered_image = image_3D

    writer = sitk.ImageFileWriter()
# Use the study/series/frame of reference information given in the meta-data
# dictionary and not the automatically generated information from the file IO
    writer.KeepOriginalImageUIDOn()

# Copy relevant tags from the original meta-data dictionary (private tags are
# also accessible).
    tags_to_copy = [
        "0010|0010",  # Patient Name
        "0010|0020",  # Patient ID
        "0010|0030",  # Patient Birth Date
        "0020|000D",  # Study Instance UID, for machine consumption
        "0020|0010",  # Study ID, for human consumption
        "0008|0020",  # Study Date
        "0008|0030",  # Study Time
        "0008|0050",  # Accession Number
        "0008|0060",  # Modality
    ]

    modification_time = time.strftime("%H%M%S")
    modification_date = time.strftime("%Y%m%d")


    direction = filtered_image.GetDirection()
    series_tag_values = [
        (k, series_reader.GetMetaData(0, k.lower()))
        for k in tags_to_copy
        if series_reader.HasMetaDataKey(0, k.lower())
    ] + [
        ("0008|0031", modification_time),  # Series Time
        ("0008|0021", modification_date),  # Series Date
        ("0008|0008", "DERIVED\\SECONDARY"),  # Image Type
        ("0020|000e","1.2.826.0.1.3680043.2.1125." + modification_date + ".1" + modification_time,),
    # Series Instance UID
    ("0020|0037","\\".join(map(str,(direction[0],direction[3],direction[6],direction[1],direction[4],direction[7],
),  # Image Orientation (Patient)
            )
        ),
    ),
    (
        "0008|103e",
        series_reader.GetMetaData(0, "0008|103e")
        if series_reader.HasMetaDataKey(0, "0008|103e")
        else "" + " Processed-SimpleITK",
    ),  # Series Description is an optional tag, so may not exist
]

    number_digits = len(str(filtered_image.GetDepth()))
    
    for i in range(filtered_image.GetDepth()):
        digits = len(str(i))
        number_zeros = number_digits - digits
    
        image_slice = filtered_image[:, :, i]
        # Tags shared by the series.
        for tag, value in series_tag_values:
            image_slice.SetMetaData(tag, value)
        # Slice specific tags.
        #   Instance Creation Date
        image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d"))
        #   Instance Creation Time
        image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S"))
        #   Image Position (Patient)
        image_slice.SetMetaData(
            "0020|0032",
            "\\".join(map(str, filtered_image.TransformIndexToPhysicalPoint((0, 0, i)))),
        )
        #   Instance Number
        image_slice.SetMetaData("0020|0013", str(i))
        image_slice.SetMetaData("0008|0018", new_UID)
        # Write to the output directory and add the extension dcm, to force writing
        # in DICOM format.
        new_UID = generate_uid()
        writer.SetFileName(os.path.join(path_to_save, modality+'.'+new_UID+".dcm"))
        writer.Execute(image_slice)
    print('===============================================')
    print("         DCM file saved into ",path_to_save)
    print('===============================================')



def generate_sitk_image(DCM_path,modality):
    """
    generate_sitk_image	Reads DICOM file at DCM_path as a SITK (SimpleITK) image.

    :param DCM_path: Path to dicom series directory. 
    :returns: The DICOM image in SITK format
    """
    series_id = ''
    for file in os.listdir(DCM_path):
        if modality in file:
            series_id = dcm.read_file(DCM_path+file).SeriesInstanceUID
            continue
    fixed_reader = sitk.ImageSeriesReader()
    dicom_names = fixed_reader.GetGDCMSeriesFileNames(DCM_path, seriesID=series_id)
    fixed_reader.SetFileNames(dicom_names)
    fixed_image = fixed_reader.Execute()

    return fixed_image

def save_dose_RD_mask(ROI_name, RS_file_path, image_files, dsDose, RS_save_path):
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
    contour = []
    for contour_seq in roi_contour_seq:
        # uid of slice that this contour appears on
        slice_uid = contour_seq.ContourImageSequence[0].ReferencedSOPInstanceUID
        # find that corresponding slice in the files
        for i, ct_file in enumerate(image_files):
            if ct_file.SOPInstanceUID == slice_uid:
                slice_new = np.array(contour_seq.ContourData).reshape((-1,3))
                contour.append(slice_new)
   

    doseSpacing = [float(each) for each in dsDose.PixelSpacing] + [float(slices[0].SliceThickness)]
    doseImagePositionPatient = [float(each) for each in dsDose.ImagePositionPatient]
    mask_full = np.zeros(dsDose.pixel_array.shape,dtype='uint32')#SET A BLANK MASK WITH THE SIZE OF THE IMAGE OF REFERENCE
    
    #USUALLY THE IMAGES ARE REGISTERED HAVING THE SAME SIZE #IF NOT, THE POSITION OF THE CONTOURS SHOULD BE ALIGNED
    
    for slicei in contour:
        coords_px = get_mask_nifti_dose(slicei,doseImagePositionPatient[0],doseImagePositionPatient[1],doseImagePositionPatient[2],doseSpacing)
        rows,cols = polygon(coords_px[1],coords_px[0])
        #if int(coords_px[-1])<dsDose.pixel_array.shape[0]:
        for row in range(0,len(rows)):
            if 0<=rows[row]<dsDose.pixel_array.shape[1]:
                if 0<=cols[row]<dsDose.pixel_array.shape[2]:
                    mask_full[int(coords_px[-1]),rows[row],cols[row]] = 1              
    
    #mask_full_v2= np.swapaxes(mask_full, 0, 1)
    #nrrd.write(RS_save_path+'/seg_dose'+ROI_name+'.nrrd', mask_full_v2)
    seg_roi = np.array(mask_full.copy())
    nrrd.write(RS_save_path+'/seg_'+ROI_name+'_RD.nrrd', seg_roi)
    
    return seg_roi


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
        
def search_RD_path(dir_RD_path,RT_ref=None, dicoPATH=False):    
    if dicoPATH==False:
        if RT_ref!=None:
            RD_files = sorted([os.path.join(dir_RD_path, x) for x in os.listdir(dir_RD_path) if '.dcm' in x])
            slices = [dcm.dcmread(j, force=True) for j in RD_files]
            if len(slices)!=0:
                try:
                    if slices[0].Modality=='RTDOSE' and slices[0].FrameOfReferenceUID==RT_ref:
                        return True
                except:
                    if slices[0][0x0008, 0x0016].value=='1.2.840.10008.5.1.4.1.1.481.2' and slices[0].FrameOfReferenceUID==RT_ref:
                        return True


def get_dirs_RD(paths_RT,paths_images_all,dicoPATH=False):
    paths = {}
    for key in paths_RT.items():
        RT_path = ("/").join(paths_RT[key[0]].split('/')[:-1])
        if dicoPATH==False:
            RT_files = sorted([os.path.join(RT_path, x) for x in os.listdir(RT_path) if '.dcm' in x])
            #TO DO: TO CHECK THE DATES OF THE RT FILES TO SORT THEM IN CASE THERE ARE TWO. IT HAS TO TAKE THE MOST RECENT ONE
            slices = [dcm.dcmread(j, force=True) for j in RT_files]
            for path_2 in paths_images_all:
                RD_file = search_RD_path(path_2,slices[0].ReferencedFrameOfReferenceSequence[0].FrameOfReferenceUID,dicoPATH)
                if RD_file==True:      
                    RD_files = sorted([os.path.join(path_2, x) for x in os.listdir(path_2) if '.dcm' in x])
                    paths[RT_path] = RD_files[0]
                else:
                    continue
        else:
            path_RD = get_path_RD_dicoPATH(RT_path)
            paths[RT_path] = path_RD
    return paths
    
def search_RS_path(dir_RS_path,image_UID=None):    
    if image_UID!=None:
        RS_files = sorted([os.path.join(dir_RS_path, x) for x in os.listdir(dir_RS_path) if '.dcm' in x])
        slices = [dcm.dcmread(j, force=True) for j in RS_files]
        if len(slices)!=0:
            if slices[0].Modality=='RTSTRUCT' and slices[0].StructureSetLabel==image_UID:
                return True
            elif slices[0][0x0008, 0x0016].value=='1.2.840.10008.5.1.4.1.1.481.3' and slices[0].StructureSetLabel==image_UID:
                return True
                #else:
                    #print('############  Check RT format ############')
       
                #uid_rt = slices[0].ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0]
                #if slices[0].Modality=='RTSTRUCT' and uid_rt==image_UID:
                ##
        
def get_dirs_RT(paths_images_all,dicoPATH=False):
    paths = {}
    for path in paths_images_all:
        set_images = get_set_images(path)
        if len(set_images)!=0:
            if dicoPATH==False:
                for path_2 in paths_images_all:
                    RS_file = search_RS_path(path_2,set_images[0].SeriesDescription)
                    if RS_file==True:
                        RS_files = sorted([os.path.join(path_2, x) for x in os.listdir(path_2) if '.dcm' in x])
                        paths[path] = RS_files[0]
                    else:
                        continue
            else:
                if path.split('/')[-1][9:11] == 'CT' and len(path.split('/')[-1])==23:
                    path_RS = get_path_RS_dicoPATH(path)
                    paths[path] = path_RS
                else:
                    continue
    return paths
    
def search_RS_file(path,image_UID=None):
    RS_files = get_set_RS_path(path,image_UID)
    return RS_files

def get_path_RS_dicoPATH(path_CT):   
    '''Gets the RS path (Rt structure file) for the CT folder'''  
    file_RS = [x for x in os.listdir(path_CT) if 'RS' in x][0]
    return os.path.join(path_CT, file_RS)

def get_path_RD_dicoPATH(path_RS): 
    file_RD = [x for x in os.listdir(path_RS) if 'RD' in x]
    files = [[dcm.dcmread(path_RS+'/'+x).InstanceCreationTime,x] for x in file_RD]
    sorted_data = sorted(files)# finds the RD file with the name RD.######.dcm
    #print(sorted_data)
    return os.path.join(path_RS, sorted_data[-1][-1])



