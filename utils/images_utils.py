from longiDICOM import *
import sys
sys.path.append('longiDICOM/code')
import sys
import matplotlib.pyplot as plt
import pydicom as dcm
import SimpleITK as sitk
import numpy as np
from __future__ import print_function
import time, os
from pydicom.uid import generate_uid
import shutil
from importlib import reload

from Registration.registration_core import find_registration_file_two_images, register_two_images, find_moving_reference
from sitk_img_tools import save_dicoms, generate_sitk_image
from Registration.dicom_registration import get_file_lists


def register_images():
  register_patient(patients_path[INDEXP], plot = True,opt=True)

  return 
  

def save_registration_dcm():
  return 
  
def save_registered_image_dcm():
  return 


