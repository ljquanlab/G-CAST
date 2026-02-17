## Identifying multiscale domains from spatial transcriptomics via graph autoencoder with contrastive learning based on cross-modality and data augmentation
### Overview
We propose GCAST, a graph contrastive autoencoder framework for spatial transcriptomics that seamlessly integrates multimodal SRT data. The framework not only captures spatial gene expression patterns to identify tissue domains but also adapts to datasets with or without histological images and supports the integration of multiple datasets for joint analyses.
![image](GCAST.png)
### Datasets
| Datasets   | Sources       |
|------------|---------------|
| DLPFC    | [DLPFC](http://research.libd.org/spatialLIBD/)    |
| Mouse v10x    | [ v10x](https://www.10xgenomics.com/datasets)     |
| Mouse Olfatory bulb(Stereo-seq)    | [Olfatory](https://github.com/JinmiaoChenLab/SEDR_analyses/tree/master/data)    |
| Mouse Embyro Data   |  [Embyro](https://db.cngb.org/stomics/datasets/STDS0000058)    |

### Hardware Requirements
GPU / VRAM: An NVIDIA GPU with at least 16 GB of VRAM (e.g.,  V100 [16–32 GB], RTX 3090 [24 GB]), or A100 [40 GB] is strongly recommended to ensure stable and efficient training.

### Environmental installation
```bash
# Step1
git clone https://github.com/ljquanlab/G-CAST
cd ./G-CAST

# Step2 
# 方式一：GPU
conda create -n G-CAST python=3.10 -y
conda activate G-CAST
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt

# 方式二：使用 pip
pip3 install torch torchvision
pip install -r requirements.txt
```
Or you can try GCAST through docker.
```
# Dockerfile
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app


COPY . /app
RUN pip install --upgrade pip
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir -r requirements.txt

CMD ["--help"]
ENTRYPOINT ["python", "app.py"]
```
```
docker build -t gcast:version1 .
```

### Quick Start
```
# for CPU or GPU
python app.py --input_path ./Dataset/DLPFC --n_clusters 7 --sample_name 151674
# for Docker
docker run  gcast:version1  --input_path ./Dataset/DLPFC --n_clusters 7 --sample_name 151674
```


Or you can follow the Tutorial to conduct relevant experiments.




