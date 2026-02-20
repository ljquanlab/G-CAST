
### Preloading of the datasets and weight files
Before starting the tutorials in fold *Tutorial*, please access the link: https://pan.quark.cn/s/fa511e11c294, download datasets and the weight files, and place them in  *../Dataset* folder. Then you can try to start GCAST by following the related tutorials with details. 


###  The structure of datasets and the weighted models 
To illustrate the dataset organization, we use DLPFC-151674 as an example. Under the folder "../Dataset/DLPFC/", the sample 151674 is structured as follows:

../Dataset/DLPFC/151674/ <br>
│── spatial/  # spatial metadata and histology images <br>
│   ├── scalefactors_json.json <br>
│   ├── tissue_full_image.tif <br>
│   ├── tissue_hires_image.png <br>
│   ├── tissue_lowres_image.png <br> 
│   ├── tissue_positions_list.csv  <br>
│   ├── tissue_positions_list.txt  <br> 
│
│── filtered_feature_bc_matrix.h5   # filtered gene expression matrix   <br>
│── 151674.pth                      # PyTorch weighted feature file  <br>
│── 151674.npy                      # NumPy feature file   <br> 
│── truth.txt                       # ground truth labels  <br>


Moreover, we have placed all the involved weight files and the processed spatial transcriptomics datasets under the specified directory path for convenient access and reproducibility. 

../Dataset/DLPFC/151674/151674.pth                  # Tutorial1  <br>
../Dataset/DLPFC/subject/subject3.pth               # Tutorial2   <br>
../Dataset/v10x/cross-region/AP.pth                 # Tutorial3   <br>
../Dataset/Mouse/cross_platform/CrossPlatform.pth   # Tutorial4 <br>
../Dataset/Mouse/Mouse_EMbryo/MouseEmbryo.pth       # Tutorial5 <br>