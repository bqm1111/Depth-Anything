import cv2
import numpy as np

import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm
import torchvision.transforms as transforms

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
import open3d.camera as camera
import open3d as o3d
from metric_depth.zoedepth.models.builder import build_model
from metric_depth.zoedepth.utils.config import get_config
from PIL import Image
import os
import getHHA
from tqdm import tqdm


def fisheyeundistort(img, K, dist, new_K=None, R=np.eye(3)):
    h, w = img.shape[:2]
    DIM = (w, h)
    if new_K is None:
        # new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, dist, DIM, np.eye(3), balance=1)
        new_K = K
    # with new intrinsic matrix, undistort image
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, dist, R, new_K, DIM, cv2.CV_16SC2
    )
    undistorted_img = cv2.remap(
        img,
        map1,
        map2,
        interpolation=cv2.INTER_AREA,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return undistorted_img


FY = 5.1885790117450188e+02
FX = 5.1946961112127485e+02

K = np.array(
    [
        [591.9355482818029, 0.0, 3.2558244941119034e+02],
        [0.0, 591.5510948827254, 2.5373616633400465e+02],
        [0.0, 0.0, 1.0],
    ]
)
D = np.array(
    [
        -0.017516516278310947,
        -0.02772068623726405,
        0.04940346323827067,
        -0.043988876827107594,
    ]
)
depth_anything = (
    DepthAnything.from_pretrained(
        "LiheYoung/depth_anything_vitl14").to("cuda").eval()
)

transform = Compose(
    [
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method="lower_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ]
)

w = 640
h = 480
DATASET = "nyu"
config = get_config("zoedepth", "eval", DATASET)
config.pretrained_resource = "local::./ckpts/depth_anything_metric_depth_indoor.pt"
model = build_model(config).to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()
path = "/home/sherlock/Pictures/segmentation/sunrgbd/sunrgbd_trainval"

for image_path in tqdm(os.listdir(os.path.join(path, "image"))):
    image = cv2.imread(os.path.join(path, "image", image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = (
        transforms.ToTensor()(Image.fromarray(image))
        .unsqueeze(0)
        .to("cuda" if torch.cuda.is_available() else "cpu")
    )

    pred = model(image_tensor, DATASET)
    if isinstance(pred, dict):
        pred = pred.get("metric_depth", pred.get("out"))
    elif isinstance(pred, (list, tuple)):
        pred = pred[-1]
    pred = pred.squeeze().detach().cpu().numpy()
    resized_pred = Image.fromarray(pred).resize((w, h), Image.NEAREST)
    resized_pred = np.array(resized_pred)
    raw_save_file = image_path.split(".")[0] + ".npy"
    save_path = os.path.join(path, "rawDepthAnything")
    save_file = os.path.join(save_path, raw_save_file)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(save_file, resized_pred)
    # cv2.imshow("img", (resized_pred / np.max(resized_pred)* 255.0).astype(np.uint8))
    # cv2.waitKey()
    # camera_matrix = getHHA.getCameraParam('color')
    # hha_complete = getHHA.getHHA(camera_matrix, resized_pred, resized_pred)
    # #
    # resized_pred = resized_pred / np.max(resized_pred) * 255.0
    # # cv2.imwrite(os.path.join(path, "DepthAnything_metric_HHA", image_path), hha_complete.astype(np.uint8))
