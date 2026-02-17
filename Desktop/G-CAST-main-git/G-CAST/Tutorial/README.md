
### Preloading of the datasets and weight files
Before starting the tutorials in fold *Tutorial*, please access the link: https://pan.quark.cn/s/fa511e11c294, download datasets and the weight files, and place them in  *../Dataset* folder. Then you can try to start GCAST by following the related tutorials with details. 


###  The structure of datasets and the weighted models 
To illustrate the dataset organization, we use DLPFC-151674 as an example. Under the folder "../Dataset/DLPFC/", the sample 151674 is structured as follows:

../Dataset/DLPFC/151674/
│── spatial/  # spatial metadata and histology images
│   ├── scalefactors_json.json
│   ├── tissue_full_image.tif
│   ├── tissue_hires_image.png
│   ├── tissue_lowres_image.png
│   ├── tissue_positions_list.csv
│   ├── tissue_positions_list.txt
│
│── filtered_feature_bc_matrix.h5   # filtered gene expression matrix  
│── 151674.pth                      # PyTorch weighted feature file
│── 151674.npy                      # NumPy feature file 
│── truth.txt                       # ground truth labels

