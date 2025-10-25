import cv2
import os
import numpy as np
import json


def load_diameter_fullres(path, scale=False, scalerfactor_json="/spatial/scalefactors_json.json"):
    with open(path+ scalerfactor_json) as f:
        data = json.load(f)
    print("spot_diameter_fullres:", data["spot_diameter_fullres"])
    print("tissue_hires_scalef:", data["tissue_hires_scalef"])
    print("fiducial_diameter_fullres", data["fiducial_diameter_fullres"])
    print("tissue_lowres_scalef", data["tissue_lowres_scalef"])
    if not scale:
        return np.ceil(data["spot_diameter_fullres"])
    else:
        # return np.ceil(data["tissue_hires_scalef"]*data["spot_diameter_fullres"]), data["tissue_hires_scalef"]
        return  np.ceil(data["spot_diameter_fullres"]), data["tissue_hires_scalef"]

def get_image_size(file_path, file_image_name="spatial/tissue_full_image.tif"):
    file_image = os.path.join(file_path, file_image_name)
    # print(file_image)
    # img = Image.open(file_image)
    img = cv2.imread(file_image, cv2.IMREAD_UNCHANGED)
    if img.any():
        H, W = img.shape[:2]
        return H, W
    return


def generate_image_embedding_color(
        adata,
        file_path,
        slide,
        # clip_img_path,
        scale=1,
        file_image_name="spatial/tissue_full_image.tif",
        patch_size: int = 128,
        # clip_img_path=os.path.join(f"Dataset/v10x", 'clip_img'),
):
    clip_img_path=os.path.join(file_path, 'clip')
    file_image = os.path.join(file_path, file_image_name)

    img = cv2.imread(file_image, cv2.IMREAD_UNCHANGED)
    if img is None:
        print("Error: Unable to load image from path:", file_image)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    os.makedirs(clip_img_path, exist_ok=True)
    position = adata.obsm['spatial']
    for i in range(position.shape[0]):
        x, y = position[i, 0], position[i, 1]
        x, y = int(x*scale), int(y*scale)
        r = np.ceil(patch_size * scale/ 2).astype(int)

        x_start, x_end = max(0, x - r), min(img.shape[0], x + r)
        y_start, y_end = max(0, y - r), min(img.shape[1], y + r)
        # x_start,x_end, y_start, y_end = np.ceil(x_start*scale).astype(int), np.ceil(x_end*scale).astype(int), np.ceil(y_start*scale).astype(int), np.ceil(y_end*scale).astype(int)
        patch = img[x_start:x_end, y_start:y_end, :]

        cv2.imwrite(os.path.join(clip_img_path, f'{i}.png'),
                    cv2.cvtColor(patch.astype(np.uint8), cv2.COLOR_RGB2BGR))
    print(f"{slide} patch finished!")

























