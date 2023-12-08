import numpy as np
import argparse
import os
import glob
import shutil
import scipy


def main(args):

    # Get all data folders (views)
    data_folders = sorted(glob.glob(os.path.join(args.input_dir,'*.data')))
    n_views = len(data_folders)

    # Get camera poses
    CAMERA_CALIB_PATH = os.path.join(args.source_dir,'Calib_Results.mat')
    camera_dict = scipy.io.loadmat(CAMERA_CALIB_PATH)

    # Get projections matrices
    bottom = np.array([0, 0, 0, 1],dtype=float).reshape((1,4))
    K = np.concatenate([np.concatenate([camera_dict['KK'],np.zeros((3,1),dtype=float)],axis=1), bottom], axis=0)
    R_w2c_mats = [camera_dict[f"Rc_{idx+1}"].astype(np.float32) for idx in range(n_views)]
    T_w2c_mats = [camera_dict[f"Tc_{idx+1}"].astype(np.float32) for idx in range(n_views)]
    RT_w2c_mats = [np.concatenate([np.concatenate([R_w2c_mats[idx], T_w2c_mats[idx]], axis=1), bottom], axis=0) for idx in range(n_views)]
    proj_mats = [K @ RT_w2c_mats[idx] for idx in range(n_views)]

    # Create the output directory if it doesn't exist
    MASK_PATH = os.path.join(args.output_dir,'mask')
    NORMAL_PATH = os.path.join(args.output_dir,'normal')
    ALBEDO_PATH = os.path.join(args.output_dir,'albedo')
    os.makedirs(MASK_PATH, exist_ok=True)
    os.makedirs(NORMAL_PATH, exist_ok=True)
    os.makedirs(ALBEDO_PATH, exist_ok=True)

    # Copy data
    proj_dict = {}
    for i, data_folder in enumerate(data_folders):

        # Save projection matrix
        proj_dict[f"world_mat_{i}"] = proj_mats[i]

        # Copy masks, normal and albedo maps
        shutil.copy(os.path.join(args.source_dir,f'view_{i+1:02}','mask.png'), os.path.join(MASK_PATH,f"{i:03}.png"))
        shutil.copy(os.path.join(data_folder,'normal.png'), os.path.join(NORMAL_PATH,f"{i:03}.png"))
        shutil.copy(os.path.join(data_folder,'baseColor.png'), os.path.join(ALBEDO_PATH,f"{i:03}.png"))

    # Save .npz files
    np.savez(os.path.join(args.output_dir, "cameras.npz"), **proj_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy SDM-UniPS data to IDR format.")
    parser.add_argument("--input_dir", help="Path to the normal map (RGB) input image.")
    parser.add_argument("--source_dir", help="Path to the cameras.npz and masks folder.")
    parser.add_argument("--output_dir", default="", help="Directory to save images.")
    args = parser.parse_args()
    main(args)
