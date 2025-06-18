from setuptools import setup, find_packages

setup(
    name='mCODE-MOSAICO',
    version='0.1.0',
    description='A package and pipeline to extract radiomics and dosiomics structured data',
    author='Odette Rios-Ibacache',
    author_email='odette.riosibacache@mail.mcgill.ca',
    packages=find_packages(),
    install_requires=[
        'pydicom', 
        'scipy',
        'numpy',
        'pyvista', 
        'scikit-learn',
        'shapely',
        'opencv-python',
    ]
