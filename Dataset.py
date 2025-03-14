!pip install kaggle
!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d rabieelkharoua/alzheimers-disease-dataset
import os
import zipfile

zip_file = "alzheimers-disease-dataset.zip"
extract_folder = "alzheimers-disease-dataset"

with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)


print("Dataset extracted successfully!")
os.listdir(extract_folder)
