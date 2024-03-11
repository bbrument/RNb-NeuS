import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os
from glob import glob
from icecream import ic
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp


# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

def load_image(path):
    image = cv.imread(path, cv.IMREAD_UNCHANGED)
    if image.dtype == "uint8":
        bit_depth = 8
    elif image.dtype == "uint16":
        bit_depth = 16
    return cv.cvtColor(image, cv.COLOR_BGR2RGB)/np.float32(2**bit_depth - 1)

def load_normal(path):
    image = load_image(path)
    normal = image * 2.0 - 1.0  # Convert to range [-1, 1]
    normal[:,:,1] = -normal[:,:,1] # y axis is flipped
    normal[:,:,2] = -normal[:,:,2] # z axis is flipped
    return normal

def save_image(path, image, bit_depth=8):
    image_cp = np.copy(image)
    image_cp = (image_cp * np.float64(2**bit_depth - 1))
    image_cp = np.clip(image_cp, 0, 2**bit_depth - 1)
    if bit_depth == 8:
        image_cp = image_cp.astype(np.uint8)
    elif bit_depth == 16:
        image_cp = image_cp.astype(np.uint16)
    image_cp = cv.cvtColor(image_cp, cv.COLOR_RGB2BGR)
    cv.imwrite(path, image_cp, [cv.IMWRITE_PNG_COMPRESSION, 0])

def save_normal(path, normal, bit_depth=8):
    normal_flipped = np.copy(normal)
    normal_flipped[:,:,1] = -normal_flipped[:,:,1] # y axis is flipped
    normal_flipped[:,:,2] = -normal_flipped[:,:,2] # z axis is flipped
    image = (normal_flipped + 1) / 2
    save_image(path, image, bit_depth=bit_depth)


class Dataset:
    def __init__(self, conf, no_albedo=False):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        self.device = torch.device('cuda')
        self.conf = conf

        self.data_dir = conf.get_string('data_dir')
        self.normal_dir = conf.get_string('normal_dir', default='normal')
        self.albedo_dir = conf.get_string('albedo_dir', default='')
        self.no_albedo = no_albedo
        if self.albedo_dir == '':
            self.no_albedo = True
        self.mask_dir = conf.get_string('mask_dir', default='mask')

        self.render_cameras_name = conf.get_string('render_cameras_name')
        self.object_cameras_name = conf.get_string('object_cameras_name')
        
        self.camera_outside_sphere = conf.get_bool('camera_outside_sphere', default=True)
        self.scale_mat_scale = conf.get_float('scale_mat_scale', default=1.1)

        camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))
        self.camera_dict = camera_dict

        masks_lis = sorted(glob(os.path.join(self.data_dir, self.mask_dir, '*.png')))
        masks_np = np.stack([cv.imread(im_name, -1) for im_name in masks_lis]) / 255.0
        masks_np = np.where(masks_np > 0.5, 1.0, 0.0)
        self.n_images, self.H, self.W = masks_np.shape

        normals_lis = sorted(glob(os.path.join(self.data_dir, self.normal_dir, '*.png')))
        normals_np = np.stack([load_normal(im_name) for im_name in normals_lis]) # [n_images, H, W, 3]
        self.normals_lis = normals_lis

        if not self.no_albedo:
            albedos_lis = sorted(glob(os.path.join(self.data_dir, self.albedo_dir, '*.png')))
            albedos_np = np.stack([load_image(im_name) for im_name in albedos_lis])
            self.albedos_lis = albedos_lis

        light_directions_cam_warmup_np = self.gen_light_directions().transpose() # [3(n_lights),3]
        self.n_lights = light_directions_cam_warmup_np.shape[0]
        shaded_images_warmup_np = np.maximum(np.sum(normals_np[:, np.newaxis, :, :, :] * light_directions_cam_warmup_np[np.newaxis, :, np.newaxis, np.newaxis, :], axis=-1), 0)[:,:,:,:,np.newaxis]
        if not self.no_albedo:
            images_warmup_np = albedos_np[:,np.newaxis,:,:,:] * shaded_images_warmup_np
        else:
            images_warmup_np = np.tile(shaded_images_warmup_np, (1, 1, 1, 1, 3))

        light_directions_cam_np = self.gen_light_directions(normals_np) # [n_images, 3(n_lights), H, W, 3]
        # light_directions_cam_np = np.zeros((self.n_images, self.n_lights, self.H, self.W, 3))
        shaded_images_np = np.maximum(np.sum(normals_np[:, np.newaxis, :, :, :] * light_directions_cam_np, axis=-1), 0)[:,:,:,:,np.newaxis]

        if not self.no_albedo:
            images_np = albedos_np[:,np.newaxis,:,:,:] * shaded_images_np
        else:   
            images_np = np.tile(shaded_images_np, (1, 1, 1, 1, 3))

        # world_mat is a projection matrix from world to image
        self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
        self.scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        self.intrinsics_all = []
        self.pose_all = []
        light_directions_warmup_np = np.zeros((self.n_images, self.n_lights, 3))
        light_directions_np = np.zeros((self.n_images, self.n_lights, self.H, self.W, 3))
        for idx, scale_mat, world_mat in zip(range(self.n_images), self.scale_mats_np, self.world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

            # convert light_directions_np to world space
            for idx_light in range(self.n_lights):
                light_directions_warmup_np[idx, idx_light, :] = np.matmul(pose[:3, :3], light_directions_cam_warmup_np[idx_light, :].T).T

                light_dir_res = light_directions_cam_np[idx, idx_light, :, :, :].reshape(-1, 3)
                light_dir_world = np.matmul(pose[:3, :3], light_dir_res.T).T
                light_directions_np[idx, idx_light, :, :, :] = light_dir_world.reshape(self.H, self.W, 3)
        
        self.light_directions_warmup = torch.from_numpy(light_directions_warmup_np.astype(np.float32)).cpu()  # [n_images, 3(n_lights), 3]
        self.images_warmup = torch.from_numpy(images_warmup_np.astype(np.float32)).cpu()  # [n_images, 3, H, W, 3]
        self.light_directions = torch.from_numpy(light_directions_np.astype(np.float32)).cpu()  # [n_images, 3(n_lights), H, W, 3]
        self.images = torch.from_numpy(images_np.astype(np.float32)).cpu()  # [n_images, 3, H, W, 3]
        self.masks = torch.from_numpy(masks_np.astype(np.float32)).unsqueeze(3).cpu()
        del normals_np
        if not self.no_albedo:
            del albedos_np
        del light_directions_cam_warmup_np
        del light_directions_warmup_np
        del images_warmup_np
        del light_directions_cam_np
        del light_directions_np
        del images_np
        del masks_np
        self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)   # [n_images, 4, 4]
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = torch.stack(self.pose_all).to(self.device)# [n_images, 4, 4]

        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])
        # Object scale mat: region of interest to **extract mesh**
        object_scale_mat = np.load(os.path.join(self.data_dir, self.object_cameras_name))['scale_mat_0']
        object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
        object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
        self.object_bbox_min = object_bbox_min[:3, 0]
        self.object_bbox_max = object_bbox_max[:3, 0]

        print('Load data: End')
    
    def gen_light_directions(self, normal=None):
        tilt = np.radians([0, 120, 240])
        slant = np.radians([30, 30, 30]) if normal is None else np.radians([54.74, 54.74, 54.74])
        n_lights = tilt.shape[0]

        u = -np.array([
            np.sin(slant) * np.cos(tilt),
            np.sin(slant) * np.sin(tilt),
            np.cos(slant)
        ]) # [3, 3(n_lights)]

        if normal is not None:
            n_images, n_rows, n_cols, _ = normal.shape # [n_images, H, W, 3]
            # normal_flat = normal.reshape(-1, 3) # [n_images*H*W, 3]
            # outer_prod = np.einsum('ij,ik->ijk', normal_flat, normal_flat) # [n_images*H*W, 3, 3]
            outer_prod = np.einsum('...j,...k->...jk', normal, normal) # [n_images, H, W, 3, 3]
            U, _, _ = np.linalg.svd(outer_prod)

            det_U = np.linalg.det(U)
            det_U_sign = np.where(det_U < 0, -1, 1)[..., np.newaxis, np.newaxis]

            R = np.where(det_U_sign < 0, 
                        np.einsum('...ij,jk->...ik', U, np.array([[0, 0, 1], [-1, 0, 0], [0, 1, 0]])), 
                        np.einsum('...ij,jk->...ik', U, np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])))
            
            R_22 = (R[..., 2, 2] < 0)[..., np.newaxis, np.newaxis]
            R = np.where(R_22, np.einsum('...ij,jk->...ik', R, np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])), R)

            light_directions_all = np.einsum('...lm,mn->...ln', R, u) # [n_images, H, W, 3, 3(n_lights)]
            light_directions = light_directions_all.transpose(0, 4, 1, 2, 3)
        else:
            light_directions = u

        return light_directions

    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return (
            rays_o.transpose(0, 1),
            rays_v.transpose(0, 1),
            pixels_x.transpose(0, 1),
            pixels_y.transpose(0, 1)
        )

    def ps_gen_random_rays_at(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        pixels_x = torch.randint(low=int(0.00*self.W), high=int(1.00*self.W), size=[batch_size])
        pixels_y = torch.randint(low=int(0.00*self.H), high=int(1.00*self.H), size=[batch_size])
        color = self.images[img_idx[0]][img_idx[1]][(pixels_y, pixels_x)]    # batch_size, 3
        mask = self.masks[img_idx[0]][(pixels_y, pixels_x)]      # batch_size, 3
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx[0], None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
        rays_v = torch.matmul(self.pose_all[img_idx[0], None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx[0], None, :3, 3].expand(rays_v.shape) # batch_size, 3
        return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, :1]], dim=-1).cuda()    # batch_size, 10

    def ps_gen_random_rays_at_view_on_all_lights(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        pixels_x = torch.randint(low=int(0.00*self.W), high=int(1.00*self.W), size=[batch_size], device='cpu')
        pixels_y = torch.randint(low=int(0.00*self.H), high=int(1.00*self.H), size=[batch_size], device='cpu')
        images_warmup = self.images_warmup[img_idx,:,pixels_y,pixels_x,:] # nb_light, batch_size, 3-4
        images = self.images[img_idx,:,pixels_y,pixels_x,:] # nb_light, batch_size, 3-4

        mask = self.masks[img_idx][(pixels_y, pixels_x)] # batch_size, 3
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float().cuda()  # batch_size, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape) # batch_size, 3

        return torch.cat([rays_o.cpu(), rays_v.cpu(), mask[:, :1].cpu()], dim=-1).cuda(), images_warmup.cuda(), images.cuda(), pixels_x.cuda(), pixels_y.cuda()

    def gen_random_rays_at(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        pixels_x = torch.randint(low=int(0.00*self.W), high=int(1.00*self.W), size=[batch_size])
        pixels_y = torch.randint(low=int(0.00*self.H), high=int(1.00*self.H), size=[batch_size])
        color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
        mask = self.masks[img_idx][(pixels_y, pixels_x)]      # batch_size, 3
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape) # batch_size, 3
        return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, :1]], dim=-1).cuda()    # batch_size, 10

    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        """
        Interpolate pose between two cameras.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[0, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        trans = self.pose_all[idx_0, :3, 3] * (1.0 - ratio) + self.pose_all[idx_1, :3, 3] * ratio
        pose_0 = self.pose_all[idx_0].detach().cpu().numpy()
        pose_1 = self.pose_all[idx_1].detach().cpu().numpy()
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        rot = torch.from_numpy(pose[:3, :3]).cuda()
        trans = torch.from_numpy(pose[:3, 3]).cuda()
        rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far

    def image_at(self, idx, resolution_level):
        img = cv.imread(self.images_lis[idx])
        return cv.resize(img, (self.W // resolution_level, self.H // resolution_level))
    
    def normal_at(self, idx, resolution_level):
        normals = load_normal(self.normals_lis[idx]).reshape([-1, 3])
        pose = self.pose_all[idx].detach().cpu().numpy()
        normals_world = np.matmul(pose[:3, :3], normals.T).T.reshape([self.H, self.W, 3])
        return cv.resize(normals_world, (self.W // resolution_level, self.H // resolution_level))
    
    def image_at_ps(self, idv, idl, resolution_level):
        img_warmup = self.images_warmup[idv,idl,:,:,:3].cpu().detach().numpy()
        img = self.images[idv,idl,:,:,:3].cpu().detach().numpy()
        return cv.resize(img_warmup, (self.W // resolution_level, self.H // resolution_level)), cv.resize(img, (self.W // resolution_level, self.H // resolution_level))
