import cv2
import numpy as np

import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
import open3d.camera as camera
import open3d as o3d


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
video_path = "/home/sherlock/Pictures/Drone_data/Forest.avi"
cap = cv2.VideoCapture(video_path)
cnt = 0
scene_name = "Forest"
w = 720
h = 540
video = cv2.VideoWriter(
    "video_out/" + scene_name + ".mp4", cv2.VideoWriter_fourcc(*"mp4v"), 20, (2 * w, 2 * h)
)
vis = o3d.visualization.Visualizer()
vis.create_window()

while True:
    cnt = cnt + 1
    ret, image = cap.read()
    if not ret:
        break

    undistorted_image = fisheyeundistort(image, K, D)
    combine = np.concatenate((image, undistorted_image), axis=1)
    undistorted_image = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2RGB)
    undistorted_image = undistorted_image / 255.0

    undistorted_image = transform({"image": undistorted_image})["image"]
    undistorted_image = torch.from_numpy(undistorted_image).unsqueeze(0).to("cuda")
    import time
    start = time.time()
    with torch.no_grad():
        depth = depth_anything(undistorted_image)
    print(f"time elapse = {time.time() - start}")
    depth = F.interpolate(depth[None], (h, w), mode="bilinear", align_corners=False)[
        0, 0
    ]

    depth = (depth - depth.min()) / (depth.max() - depth.min())

    depth = depth.cpu().numpy()

    depth_image = o3d.geometry.Image(depth)
    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        w, h, 591.9355482818029, 591.5510948827254, 361.3170292279398, 285.250250752649
    )

    point_cloud = o3d.geometry.PointCloud.create_from_depth_image(
        depth_image, camera_intrinsic, depth_scale=100
    )

    depth_color = cv2.applyColorMap(
        (depth * 255).astype(np.uint8), cv2.COLORMAP_INFERNO
    )
    vis.get_render_option().point_color_option = (
        o3d.visualization.PointColorOption.Color
    )
    vis.get_render_option().point_size = 3.0
    vis.add_geometry(point_cloud)
    vis.capture_screen_image("result/file.jpg", do_render=True)
    vis.remove_geometry(point_cloud)
    pc_image = cv2.imread("result/file.jpg")
    pc_image = cv2.resize(pc_image, (w, h))
    pc_image = cv2.flip(pc_image, 0)
    depth_combine = np.concatenate((depth_color, pc_image), axis=1)
    combine = np.concatenate((combine, depth_combine), axis=0)
    video.write(combine)
    # cv2.imshow("combine", combine)
    if cv2.waitKey(1) == ord('q'):
        break

video.release()
vis.destroy_window()
