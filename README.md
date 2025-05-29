# mCODE-MOSAICO: Multi (Radi-Dosi)Omics features Acquisition and Images Characteristics for Oncology

## Table of Contents
  *  [Authors](#Authors)
  *  [Motivation](#Motivation)
  *  [Features](#Features)
  *  [Dependencies](#Dependencies)
  *  [Usage](#Usage)
  *  [Use Case](#UseCase)
  *  [Acknowledgements](#Acknowledgments)
## Authors
Odette Rios-Ibacache

Contact email: <a href="mailto:odette.riosibacache@mail.mcgill.ca">odette.riosibacache@mail.mcgill.ca</a>

Website:  [www.kildealab.com](https://www.kildealab.com) 

## Motivation
The scattered nature of health data, along with the lack of standardization and interoperability, limits the potential of Artificial Intelligence (AI) incorporating medical images and radiomics to automate outcomes assessment in radiotherapy (RT) treatments. Establishing a standardized lexicon and data structure could enhance multicenter clinical studies. Our goal is to structure patient data relevant to RT research and create a knowledge base (KB), a machine-readable repository, with an ontology as a domain, including radiomics and medical images. Methods: We aim to identify the essential data elements needed to encode radiomics and dosiomics information and develop the ontology. We are building our study on Minimal Common Oncology Data Elements (mCODE), an international initiative to improve interoperability by establishing a core set of structured data elements. We propose an extension to link patients' medical image data, radiomics, and dosiomics with their health records. A review of the existing literature on the standardization of radiomics and dosiomics methods was conducted to include the minimum parameters that would impact their acquisition. We included data elements recommended by the Image Biomarker Standardisation Initiative (IBSI) guidelines.

## Features
![Optional Text](diagram.png)

    
## Usage

<pre> /path/to/patient/directories/ 
├── 📁patient_id
│   ├── 📁medical_images
│       ├── 📁 date_modality
│           ├──📄image_study.json
│           ├──📄acquisition_properties.json
│           ├──📄modality_properties.json 
│           └──📄date_modality.nrrd 
│       ├── ... 
│   ├── 📁RT_plans
│       ├── 📁 date_RT
│   ├── 📁radiomics
│       ├── 📁 ROI_radiomics
│           ├── 📁 date_modality
│               ├──📄seg_ROI.nrrd
│               ├──📄seg_ROI_radiomics.json
│               └──📁 voxel_based
│                  ├──📄feature1.nrrd
│                  ├──📄feature2.nrrd
│                  └── ... 
│        ├── 📁 ROI2_radiomics
│            ├── 📁 date_modality
|                ├──📄seg_ROI2.nrrd
│                ├──📄seg_ROI2_radiomics.json
│                └──📁 voxel_based
│                   ├──📄feature1.nrrd
│                   ├──📄feature2.nrrd
│                   └── ... 
│   └── 📁dosiomics
│       ├── 📁 ROI_dosiomics
│           ├── 📁 date_RT
│               ├──📄seg_ROI.nrrd
│               ├──📄seg_ROI_dosiomics.json
│           └──📁 voxel_based
│              ├──📄feature1.nrrd
│              ├──📄feature2.nrrd
│              └── ... 
│ 
...
├── 📁patient_idN
|   └── ...
</pre>

#### Requirements
  *  [sys](https://docs.python.org/3/library/sys.html)
  *  [shutil](https://docs.python.org/3/library/shutil.html)
  *  [matplotlib](https://matplotlib.org/)
  *  [time](https://docs.python.org/3/library/time.html)
  *  [datetime](https://docs.python.org/3/library/datetime.html)
  *  [scipy](https://scipy.org/)
  *  [skimage](https://scikit-image.org/)
  *  [numpy](https://numpy.org/)
  *  [os](https://docs.python.org/3/library/os.html)
  *  [gc](https://docs.python.org/3/library/gc.html)
  *  [pandas](https://pandas.pydata.org/)
  *  [pydicom](https://pydicom.github.io/pydicom/stable/)
  *  [json](https://docs.python.org/3/library/json.html)
  *  [SimpleITK](https://docs.python.org/3/library/json.html)
  *  [pynrrd](https://pynrrd.readthedocs.io/en/stable/index.html#)
