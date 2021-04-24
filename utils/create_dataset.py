import os 
import cv2 
import glob2

import numpy as np
from tqdm import tqdm 

def create_mask(root_dir="dataset_btc", annot_dir="annots", image_dir="images", mask_dir="masks"):
    annots_path = os.path.join(root_dir, annot_dir)
    images_path = os.path.join(root_dir, image_dir)
    masks_path = os.path.join(root_dir, mask_dir)

    # create mask dir
    if not os.path.exists(masks_path):
        os.mkdir(masks_path)

    list_image_error = []

    for subdir in os.listdir(images_path):
        subdir_annot_path = os.path.join(annots_path, subdir)
        subdir_image_path = os.path.join(images_path, subdir)
        subdir_mask_path = os.path.join(masks_path, subdir)

        # create subdir mask
        if not os.path.exists(subdir_mask_path):
            os.mkdir(subdir_mask_path)

        list_images_path = glob2.glob(os.path.join(subdir_image_path, "*.jpg"))
        len_list_images_path = len(list_images_path)

        with tqdm(total=len_list_images_path) as pbar:
            for image_path in list_images_path:
                try:
                    image_name = (image_path.split("/")[-1]).split(".")[0]
                    annot_path = os.path.join(subdir_annot_path, image_name + ".txt")

                    # read list points
                    with open(annot_path) as f:
                        content = f.readlines()
                    
                    list_coord = [x.strip().split(" ") for x in content] 
                    arr_points = np.array(list_coord, dtype=int)

                    # read image
                    image = cv2.imread(image_path)
                    mask = np.zeros(image.shape[:2], np.uint8)

                    cv2.drawContours(mask, [arr_points], -1, (255, 255, 255), -1, cv2.LINE_AA)
                    
                    # mask image name
                    mask_name = "{}_mask.jpg".format(image_name)
                    mask_path = os.path.join(subdir_mask_path, mask_name)

                    cv2.imwrite(mask_path, mask)
                 
                except FileNotFoundError:
                    list_image_error.append(image_path)
                    pass
    
                pbar.update(1)
    
    file_error_name = os.path.join(root_dir, "error_name.txt")
    with open(file_error_name, "w") as f:
        for image_path in list_image_error:
            f.write("{}\n".format(image_path))

def create_dataset(root_dir="dataset_btc", dest_dir="data", image_dir="images", masks_dir="masks"):
    root_images_dir = os.path.join(root_dir, image_dir)
    root_masks_dir = os.path.join(root_dir, masks_dir)
    dest_images_dir = os.path.join(dest_dir, image_dir)
    dest_masks_dir = os.path.join(dest_dir, masks_dir)

    # create dest dir
    if not os.path.exists(dest_images_dir):
        os.mkdir(dest_images_dir)

    if not os.path.exists(dest_masks_dir):
        os.mkdir(dest_masks_dir)
    
    for subdir in os.listdir(root_masks_dir):
        subdir_image_path = os.path.join(root_images_dir, subdir)
        subdir_mask_path = os.path.join(root_masks_dir, subdir)

        list_masks_path = glob2.glob(os.path.join(subdir_mask_path, "*.jpg"))
        len_list_mask_path = len(list_masks_path)

        with tqdm(total=len_list_mask_path) as pbar:
            for mask_path in list_masks_path:
                mask_name = ((mask_path.split("/")[-1]).split(".")[0]).split("_")[0]
                image_path = os.path.join(subdir_image_path, mask_name+".jpg")

                mask = cv2.imread(mask_path)
                image = cv2.imread(image_path)
                
                mask_dest_path = os.path.join(dest_masks_dir, "{}_{}.jpg".format(subdir, mask_name))
                image_dest_path = os.path.join(dest_images_dir, "{}_{}.jpg".format(subdir, mask_name))

                # save image
                cv2.imwrite(mask_dest_path, mask)
                cv2.imwrite(image_dest_path, image)

                pbar.update(1)

if __name__ == "__main__":
    # create_mask()
    create_dataset()