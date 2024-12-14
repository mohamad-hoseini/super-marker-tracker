
import os
from glob import glob

import cv2
import numpy as np
import torch

import pose_helpers as pose_helpers
import image_utils



class FacemapDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        datadir=None,
        image_data=None,
        keypoints_data=None,
        bbox=None,
        train=True,
        img_size=(256, 256),
        scorer="All",
    ):
        self.datadir = datadir
        self.scorer = scorer
        self.img_size = img_size
        self.bodyparts = [
            "eye(back)",
            "eye(bottom)",
            "eye(front)",
            "eye(top)",
            "lowerlip",
            "mouth",
            "nose(bottom)",
            "nose(r)",
            "nose(tip)",
            "nose(top)",
            "nosebridge",
            "paw",
            "whisker(I)",  # "whisker(c1)",
            "whisker(III)",  # "whisker(c2)",  # "whisker(d2)",
            "whisker(II)",  # "whisker(d1)",
        ]
        # Set image and keypoints data
        if datadir is None:
            self.images = self.preprocess_images(image_data)
            if keypoints_data is None:
                self.keypoints = None
            else:
                self.keypoints = torch.from_numpy(keypoints_data)
        else:  # Load data from directory - not used for GUI
            self.images = self.load_images()
            if keypoints_data is None:
                self.keypoints = None
            else:
                self.keypoints = torch.from_numpy(keypoints_data)
        if self.keypoints is not None:
            self.num_keypoints = self.keypoints.shape[1]
        self.num_images = self.__len__()
        # Set bounding box
        if bbox is not None:
            # Create a list of bounding boxes for each image by repeating the bbox for each image
            self.bbox = torch.from_numpy(np.tile(bbox, (self.num_images, 1)))
        else:
            self.bbox = self.estimate_bbox_from_keypoints()

        if train:
            self.augmentation = True
        else:
            self.augmentation = False

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image, keypoints = self.preprocess_data(
            self.images[item].clone().detach(),
            self.keypoints[item].clone().detach(),
            self.bbox[item].clone().detach(),
        )

        if self.augmentation:
            image, keypoints = image_utils.augment_data(image, keypoints)

        # If not a tensor, convert to tensor
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image)
        if not isinstance(keypoints, torch.Tensor):
            keypoints = torch.from_numpy(keypoints)

        data = {
            "image": image,
            "keypoints": keypoints,
            "bbox": self.bbox[item],
            "item": item,
        }
        return data

    def preprocess_images(self, image_data):
        imgs = []
        for im in image_data:
            im = torch.from_numpy(im)
            # Normalize
            im = pose_helpers.normalize99(im)
            imgs.append(im)
        return imgs

    def preprocess_data(self, image, keypoints, bbox):
        # 1. Crop image
        # if self.augmentation: #randomize bbox/cropping during training
        bbox = image_utils.randomize_bbox_coordinates(bbox, image.shape[-2:])
        image = image_utils.crop_image(image, bbox)
        y1, _, x1, _ = bbox
        keypoints[:, 0] = keypoints[:, 0] - x1
        keypoints[:, 1] = keypoints[:, 1] - y1

        # 2. Pad image to square
        image, (pad_y_top, _, pad_x_left, _) = image_utils.pad_img_to_square(image)
        keypoints = image_utils.pad_keypoints(keypoints, pad_y_top, pad_x_left)

        # 3. Resize image to resize_shape for model input
        keypoints = image_utils.resize_keypoints(
            keypoints, desired_shape=self.img_size, original_shape=image.shape[-2:]
        )
        image = image_utils.resize_image(image, self.img_size)

        return image, keypoints

    def load_images(self):
        # Check if the directory contains .png files
        img_files = sorted(glob(os.path.join(self.datadir, "*.png")))
        if len(img_files) == 0:  # If not, check if it contains subdirectories
            img_files = sorted(glob(os.path.join(self.datadir, "*/*.png")))
        if len(img_files) == 0:
            raise ValueError("No .png files found in the directory")

        imgs = []
        for file in img_files:
            # Normalize images
            im = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            # Add channel dimension
            im = im[np.newaxis, ...]
            # Convert numpy array to tensor
            im = torch.from_numpy(im)

            # Convert to float32 in the range 0 to 1
            if im.dtype == float:
                pass
            elif im.dtype == torch.uint8:
                im = im.float() / 255.0
            elif im.dtype == torch.uint16:
                im = im.float() / 65535.0
            else:
                print("Cannot handle im type " + str(im.dtype))
                raise TypeError

            # Normalize
            im = pose_helpers.normalize99(im)
            imgs.append(im)

        return imgs



    def estimate_bbox_from_keypoints(self):
        bbox = []
        for i in range(self.keypoints.shape[0]):
            x_min = np.nanmin(self.keypoints[i, :, 0])
            x_max = np.nanmax(self.keypoints[i, :, 0])
            y_min = np.nanmin(self.keypoints[i, :, 1])
            y_max = np.nanmax(self.keypoints[i, :, 1])
            bbox.append([y_min, y_max, x_min, x_max])
        # Convert to tensor
        bbox = torch.from_numpy(np.array(bbox))
        return bbox
