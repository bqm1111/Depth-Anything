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

FY = 591.5510948827254
FX = 591.9355482818029

K = np.array(
    [
        [591.9355482818029, 0.0, 361.3170292279398],
        [0.0, 591.5510948827254, 285.250250752649],
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
    DepthAnything.from_pretrained("LiheYoung/depth_anything_vitl14").to("cuda").eval()
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

w = 720
h = 540
# image_path = "/home/sherlock/Pictures/Drone_data/Ballast_tank_autonomous_1/selected/frame006294.png"
# image_path = "/home/sherlock/Pictures/Drone_data/Ballast_tank_autonomous_1/selected/frame007797.png"
# image_path = "/home/sherlock/Pictures/Drone_data/Forest/selected/frame001166.png"
# image_path = "/home/sherlock/Pictures/Drone_data/something/selected/frame002773.png"
image_path = "/home/sherlock/Pictures/Drone_data/something/selected/frame003176.png"

image = cv2.imread(image_path)

undistorted_image = fisheyeundistort(image, K, D)
cv2.imshow("undistorted.png", undistorted_image)
cv2.waitKey(100)
combine = np.concatenate((image, undistorted_image), axis=1)
undistorted_image = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2RGB)
DATASET = "nyu"
config = get_config("zoedepth", "eval", DATASET)
config.pretrained_resource = "local::./ckpts/depth_anything_metric_depth_indoor.pt"
model = build_model(config).to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

image_tensor = (
    transforms.ToTensor()(Image.fromarray(undistorted_image))
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
focal_length_x, focal_length_y = (FX, FY) 
x, y = np.meshgrid(np.arange(w), np.arange(h))
x = (x - w / 2) / focal_length_x
y = (y - h / 2) / focal_length_y
z = np.array(resized_pred)
print(z.max())
points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
colors = np.array(undistorted_image).reshape(-1, 3) / 255.0

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([pcd])
# vis = o3d.visualization.Visualizer()
# vis.create_window()
# vis.add_geometry(pcd)
# vis.run()  # user changes the view and press "q" to terminate
# param = vis.get_view_control().convert_to_pinhole_camera_parameters()
# o3d.io.write_pinhole_camera_parameters("viewpoint.json", param)
# vis.destroy_window()

# vis = o3d.visualization.Visualizer()
# vis.create_window(width=1920, height=1080)
# ctr = vis.get_view_control()
# param = o3d.io.read_pinhole_camera_parameters("view.json")
# vis.add_geometry(pcd)
# ctr.convert_from_pinhole_camera_parameters(glb.trajectory.parameters[glb.index], True)
# vis.run()
# vis.destroy_window()
