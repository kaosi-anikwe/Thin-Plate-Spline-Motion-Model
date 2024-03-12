import matplotlib

matplotlib.use("Agg")
import sys
import yaml
from argparse import ArgumentParser
from tqdm import tqdm
from scipy.spatial import ConvexHull
import numpy as np
import imageio
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
from modules.inpainting_network import InpaintingNetwork
from modules.keypoint_detector import KPDetector
from modules.dense_motion import DenseMotionNetwork
from modules.avd_network import AVDNetwork

import os
import json
import uuid
import runpod
import shutil
import requests
import tempfile
import traceback
import firebase_admin
from firebase_admin import storage
from dotenv import load_dotenv


load_dotenv()
SERVICE_CERT = json.loads(os.getenv("SERVICE_CERT"), strict=False)
STORAGE_BUCKET = os.getenv("STORAGE_BUCKET")
cred_obj = firebase_admin.credentials.Certificate(SERVICE_CERT)
firebase_admin.initialize_app(cred_obj, {"storageBucket": STORAGE_BUCKET})

if sys.version_info[0] < 3:
    raise Exception(
        "You must use Python 3 or higher. Recommended version is Python 3.9"
    )


def relative_kp(kp_source, kp_driving, kp_driving_initial):

    source_area = ConvexHull(kp_source["fg_kp"][0].data.cpu().numpy()).volume
    driving_area = ConvexHull(kp_driving_initial["fg_kp"][0].data.cpu().numpy()).volume
    adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)

    kp_new = {k: v for k, v in kp_driving.items()}

    kp_value_diff = kp_driving["fg_kp"] - kp_driving_initial["fg_kp"]
    kp_value_diff *= adapt_movement_scale
    kp_new["fg_kp"] = kp_value_diff + kp_source["fg_kp"]

    return kp_new


def load_checkpoints(config_path, checkpoint_path, device):
    with open(config_path) as f:
        config = yaml.full_load(f)

    inpainting = InpaintingNetwork(
        **config["model_params"]["generator_params"],
        **config["model_params"]["common_params"],
    )
    kp_detector = KPDetector(**config["model_params"]["common_params"])
    dense_motion_network = DenseMotionNetwork(
        **config["model_params"]["common_params"],
        **config["model_params"]["dense_motion_params"],
    )
    avd_network = AVDNetwork(
        num_tps=config["model_params"]["common_params"]["num_tps"],
        **config["model_params"]["avd_network_params"],
    )
    kp_detector.to(device)
    dense_motion_network.to(device)
    inpainting.to(device)
    avd_network.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    inpainting.load_state_dict(checkpoint["inpainting_network"])
    kp_detector.load_state_dict(checkpoint["kp_detector"])
    dense_motion_network.load_state_dict(checkpoint["dense_motion_network"])
    if "avd_network" in checkpoint:
        avd_network.load_state_dict(checkpoint["avd_network"])

    inpainting.eval()
    kp_detector.eval()
    dense_motion_network.eval()
    avd_network.eval()

    return inpainting, kp_detector, dense_motion_network, avd_network


def make_animation(
    source_image,
    driving_video,
    inpainting_network,
    kp_detector,
    dense_motion_network,
    avd_network,
    device,
    mode="relative",
):
    assert mode in ["standard", "relative", "avd"]
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(
            0, 3, 1, 2
        )
        source = source.to(device)
        driving = (
            torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32))
            .permute(0, 4, 1, 2, 3)
            .to(device)
        )
        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(driving[:, :, 0])

        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]
            driving_frame = driving_frame.to(device)
            kp_driving = kp_detector(driving_frame)
            if mode == "standard":
                kp_norm = kp_driving
            elif mode == "relative":
                kp_norm = relative_kp(
                    kp_source=kp_source,
                    kp_driving=kp_driving,
                    kp_driving_initial=kp_driving_initial,
                )
            elif mode == "avd":
                kp_norm = avd_network(kp_source, kp_driving)
            dense_motion = dense_motion_network(
                source_image=source,
                kp_driving=kp_norm,
                kp_source=kp_source,
                bg_param=None,
                dropout_flag=False,
            )
            out = inpainting_network(source, dense_motion)

            predictions.append(
                np.transpose(out["prediction"].data.cpu().numpy(), [0, 2, 3, 1])[0]
            )
    return predictions


def find_best_frame(source, driving, cpu):
    import face_alignment

    def normalize_kp(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp

    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._2D,
        flip_input=True,
        device="cpu" if cpu else "cuda",
    )
    kp_source = fa.get_landmarks(255 * source)[0]
    kp_source = normalize_kp(kp_source)
    norm = float("inf")
    frame_num = 0
    for i, image in tqdm(enumerate(driving)):
        try:
            kp_driving = fa.get_landmarks(255 * image)[0]
            kp_driving = normalize_kp(kp_driving)
            new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
            if new_norm < norm:
                norm = new_norm
                frame_num = i
        except:
            pass
    return frame_num


def download_image(url):
    try:
        response = requests.get(url)
        print(f"IMAGE DOWNLOAD: {response.status_code}")
        if response.ok:
            # Create a temporary file to save the image
            _, temp_filename = tempfile.mkstemp(suffix=".jpg")

            with open(temp_filename, "wb") as temp_file:
                temp_file.write(response.content)

            return temp_filename
        return None
    except:
        return None


def download_video(url):
    try:
        response = requests.get(url, stream=True)
        print(f"VIDEO DOWNLOAD: {response.status_code}")
        if response.ok:
            # Create a temporary file to save the video
            _, temp_filename = tempfile.mkstemp(suffix=".mp4")

            with open(temp_filename, "wb") as temp_file:
                shutil.copyfileobj(response.raw, temp_file)

            return temp_filename
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def upload_to_firebase(filename: str):
    storage_client = storage.bucket()
    blob = storage_client.blob(os.path.join("spline_motion_outputs", filename))
    blob.upload_from_filename(filename)
    blob.make_public()
    blob_url = blob.public_url
    return blob_url


def handler(job):
    request = job["input"]
    config = "config/vox-256.yaml"
    checkpoint = "checkpoints/vox.pth.tar"
    mode = request.get("mode", "relative")
    user_id = request.get("user_id")

    source_image_url = request.get("source_image")
    source_image = download_image(source_image_url)
    if not source_image:
        return {"error": f"Failed to download image with URL: {source_image_url}"}
    source_image_path = source_image

    driving_video_url = request.get("driving_video")
    driving_video = download_video(driving_video_url)
    if not driving_video:
        return {"error": f"Failed to download video with URL: {driving_video_url}"}
    driving_video_path = driving_video

    result_video = f"{user_id}_{uuid.uuid4().hex}.mp4"
    img_shape = [256, 256]

    try:
        source_image = imageio.imread(source_image)
        reader = imageio.get_reader(driving_video)
        fps = reader.get_meta_data()["fps"]
        driving_video = []
        try:
            for im in reader:
                driving_video.append(im)
        except RuntimeError:
            pass
        reader.close()

        device = torch.device("cuda")

        source_image = resize(source_image, img_shape)[..., :3]
        driving_video = [resize(frame, img_shape)[..., :3] for frame in driving_video]
        inpainting, kp_detector, dense_motion_network, avd_network = load_checkpoints(
            config_path=config, checkpoint_path=checkpoint, device=device
        )

        predictions = make_animation(
            source_image,
            driving_video,
            inpainting,
            kp_detector,
            dense_motion_network,
            avd_network,
            device=device,
            mode=mode,
        )

        imageio.mimsave(
            result_video, [img_as_ubyte(frame) for frame in predictions], fps=fps
        )
        output_url = upload_to_firebase(result_video)
        if os.path.exists(result_video):
            os.remove(result_video)

        return {"success": True, "output_video_url": output_url}
    except Exception as e:
        print(traceback.format_exc())
        return {"error": str(e)}
    finally:
        if os.path.exists(source_image_path):
            os.remove(source_image_path)
        if os.path.exists(driving_video_path):
            os.remove(driving_video_path)


runpod.serverless.start({"handler": handler})
