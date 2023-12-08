import os
import logging
import argparse
import numpy as np
import cv2 as cv
import trimesh
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from tqdm import tqdm
from pyhocon import ConfigFactory
from models.dataset import Dataset, save_image, save_normal
from models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF
from models.renderer import NeuSRenderer


class Runner:
    def __init__(self, conf_path, mode='train_rnb', case='CASE_NAME', is_continue=False, no_albedo=False):
        self.device = torch.device('cuda')

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        self.base_exp_dir = self.conf['general.base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.dataset = Dataset(self.conf['dataset'],no_albedo)
        self.iter_step = 0

        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.warm_up_iter = self.conf.get_int('train.warm_up_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)

        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight')
        self.is_continue = is_continue
        self.mode = mode
        self.no_albedo = self.dataset.no_albedo
        self.model_list = []
        self.writer = None

        # Networks
        params_to_train = []
        self.nerf_outside = NeRF(**self.conf['model.nerf']).to(self.device)
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)
        self.color_depth = self.conf["model.rendering_network"]["d_out"]
        
        params_to_train += list(self.nerf_outside.parameters())
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        if not self.no_albedo:
            params_to_train += list(self.color_network.parameters())

        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)

        self.renderer = NeuSRenderer(self.nerf_outside,
                                     self.sdf_network,
                                     self.deviation_network,
                                     self.color_network,
                                     **self.conf['model.neus_renderer'])
        
        self.renderer.color_depth = self.color_depth

        # Load checkpoint
        latest_model_name = None
        if is_continue:
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]

        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

        # Backup codes and configs for debug
        if self.mode[:5] == 'train':
            self.file_backup()


    def train_rnb(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step
        image_perm = self.get_image_perm()
        torch.random.manual_seed(0)
        for iter_i in tqdm(range(res_step)):
            torch.random.manual_seed(iter_i)
            cbn = image_perm[self.iter_step % len(image_perm)]
            data, true_rgb_warmup, true_rgb, pixels_x, pixels_y = self.dataset.ps_gen_random_rays_at_view_on_all_lights(cbn, self.batch_size)

            rays_o, rays_d, mask = data[:, :3], data[:, 3: 6], data[:, 6: 7]
            near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)

            background_rgb = None
            if self.use_white_bkgd:
                background_rgb = torch.ones([1, 3])

            if self.mask_weight > 0.0:
                mask = (mask > 0.5).float()
            else:
                mask = torch.ones_like(mask)
            mask_sum = mask.sum() + 1e-5

            if self.iter_step < self.warm_up_iter:
                true_rgb = true_rgb_warmup

                lights_dir = self.dataset.light_directions_warmup[cbn, :, :].cuda()
                lights_dir = lights_dir.reshape(self.dataset.n_lights,1,1,3)

                render_out = self.renderer.render_rnb_warmup(rays_o, rays_d, near, far, lights_dir,
                                background_rgb=background_rgb,
                                cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                no_albedo=self.no_albedo)
                
            else:
                lights_dir = self.dataset.light_directions[cbn, :, pixels_y, pixels_x, :].cuda() # [n_lights, batch_size, 3]
                lights_dir = lights_dir.reshape(self.dataset.n_lights,self.batch_size,1,3)

                render_out = self.renderer.render_rnb(rays_o, rays_d, near, far, lights_dir,
                                background_rgb=background_rgb,
                                cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                no_albedo=self.no_albedo)
            
            color_fine = render_out['color_fine']
            s_val = render_out['s_val']
            cdf_fine = render_out['cdf_fine']
            gradient_error = render_out['gradient_error']
            weight_max = render_out['weight_max']
            weight_sum = render_out['weight_sum']
            normal = render_out['gradients']

            # Loss
            color_error = ((color_fine - true_rgb) * mask[None, :, :]).reshape(-1, self.color_depth)
            # psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb)**2 *  mask[None, :, :]).sum() / ( mask[None, :, :] * 3.0)).sqrt())
            color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / (mask_sum*self.dataset.n_lights)

            eikonal_loss = gradient_error

            mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)

            loss = color_fine_loss +\
                   eikonal_loss * self.igr_weight +\
                   mask_loss * self.mask_weight # 4 * self.mask_weight ??????????????

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iter_step += 1
            
            self.writer.add_scalar('Loss/loss', loss, self.iter_step)
            self.writer.add_scalar('Loss/color_loss', color_fine_loss, self.iter_step)
            self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss, self.iter_step)
            self.writer.add_scalar('Statistics/s_val', s_val.mean(), self.iter_step)
            self.writer.add_scalar('Statistics/cdf', (cdf_fine[:, :1] * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar('Statistics/weight_max', (weight_max * mask).sum() / mask_sum, self.iter_step)
            # self.writer.add_scalar('Statistics/psnr', psnr, self.iter_step)

            if self.iter_step % self.report_freq == 0:
                print(self.base_exp_dir)
                print('iter:{:8>d} \nloss = {} \ncolor_loss={} \neikonal_loss={} \nmask_loss={} \nlr={}\n'.format(self.iter_step, loss,
                   color_fine_loss,
                   eikonal_loss * self.igr_weight,
                   mask_loss * self.mask_weight,
                   self.optimizer.param_groups[0]['lr']))
                
            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()

            if self.iter_step % self.val_freq == 0:
                self.validate_image()

            if self.iter_step % self.val_mesh_freq == 0:
                self.validate_mesh()

            self.update_learning_rate()

            if self.iter_step % len(image_perm) == 0:
                image_perm = self.get_image_perm()


    def get_image_perm(self):
        return torch.randperm(self.dataset.n_images)

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor

    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_checkpoint(self,checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        self.iter_step = checkpoint['iter_step']
        self.nerf_outside.load_state_dict(checkpoint['nerf'])
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        if "shading" not in self.mode:
            self.color_network.load_state_dict(checkpoint['color_network_fine'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        logging.info('End')
    
    def save_checkpoint(self):
        checkpoint = {
            'nerf': self.nerf_outside.state_dict(),
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'color_network_fine': self.color_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
        }
        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def validate_image(self, idv=-1, idl=-1, resolution_level=-1):
        if idv < 0:
            idv = np.random.randint(self.dataset.n_images)
        if idl < 0:
            idl = np.random.randint(self.dataset.n_lights)

        print('Validate: iter: {}, camera: {}, light: {}'.format(self.iter_step, idv, idl))

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        rays_o, rays_d, pixels_x, pixels_y = self.dataset.gen_rays_at(idv, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)
        pixels_x = pixels_x.reshape(-1, 1).split(self.batch_size)
        pixels_y = pixels_y.reshape(-1, 1).split(self.batch_size)

        out_rgb_fine = []
        out_normal_fine = []

        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            if self.iter_step < self.warm_up_iter:
                lights_dir = self.dataset.light_directions_warmup[idv, idl, :].cuda()
                lights_dir = lights_dir.reshape(1,1,1,3)

                render_out = self.renderer.render_rnb_warmup(rays_o_batch, rays_d_batch, near, far, lights_dir,
                                background_rgb=background_rgb,
                                cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                no_albedo=self.no_albedo)
                
            else:
                lights_dir = self.dataset.light_directions[idv, idv, pixels_y, pixels_x, :].cuda()
                lights_dir = lights_dir.reshape(1,self.batch_size,1,3)

                render_out = self.renderer.render_rnb(rays_o_batch, rays_d_batch, near, far, lights_dir,
                                background_rgb=background_rgb,
                                cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                no_albedo=self.no_albedo)

            def feasible(key): return (key in render_out) and (render_out[key] is not None)

            if feasible('color_fine'):
                out_rgb_fine.append(render_out['color_fine'].squeeze(0).detach().cpu().numpy())
            if feasible('gradients') and feasible('weights'):
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()
                out_normal_fine.append(normals)
            del render_out

        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1])

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0).reshape([H, W, 3, -1])

        os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)

        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                if self.iter_step < self.warm_up_iter:

                    save_image(os.path.join(self.base_exp_dir,
                                            'validations_fine',
                                            '{:0>8d}_{}_{}_{}.png'.format(self.iter_step, i, idv, idl)),
                                np.concatenate([img_fine[..., i],
                                                self.dataset.image_at_ps(idv, idl, resolution_level=resolution_level)[0]]))
                else:
                    save_image(os.path.join(self.base_exp_dir,
                                            'validations_fine',
                                            '{:0>8d}_{}_{}_{}.png'.format(self.iter_step, i, idv, idl)),
                                np.concatenate([img_fine[..., i],
                                                self.dataset.image_at_ps(idv, idl, resolution_level=resolution_level)[1]]))
            if len(out_normal_fine) > 0:
                if self.iter_step < self.warm_up_iter:
                    save_normal(os.path.join(self.base_exp_dir,
                                            'normals',
                                            '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idv)),
                                np.concatenate([normal_img[..., i],
                                                self.dataset.normal_at(idv, resolution_level=resolution_level)]))
                else:
                    save_normal(os.path.join(self.base_exp_dir,
                                            'normals',
                                            '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idv)),
                                np.concatenate([normal_img[..., i],
                                                self.dataset.normal_at(idv, resolution_level=resolution_level)]))

    def render_novel_image(self, idx_0, idx_1, ratio, resolution_level):
        """
        Interpolate view between two cameras.
        """
        rays_o, rays_d = self.dataset.gen_rays_between(idx_0, idx_1, ratio, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())

            del render_out

        img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 255).clip(0, 255).astype(np.uint8)
        return img_fine

    def validate_mesh(self, world_space=False, resolution=128, threshold=0.0):
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)

        vertices, triangles =\
            self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step)))

        logging.info('End')

    def validate_mesh_texture(self, resolution=128, threshold=0.0):
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)

        vertices, triangles = \
            self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)
        vertices_ws = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        # Get Albedo

        cut_size = 100000
        vertices_tensor = torch.tensor(vertices).type(torch.float32).split(cut_size)
        albedo = np.empty(vertices.shape)
        for k in range(len(vertices_tensor)):
            vt = vertices_tensor[k]
            sdf_nn_output = self.sdf_network(vt)
            sdf = sdf_nn_output[:, :1]
            feature_vector = sdf_nn_output[:, 1:]

            gradients = self.sdf_network.gradient(vt).squeeze()
            albedo[k*cut_size:k*cut_size+vt.shape[0],:] = np.clip(1.00*self.color_network(vt, gradients,gradients, feature_vector).cpu().detach().numpy()[:,[2,1,0]],0,1)

        mesh = trimesh.Trimesh(vertices_ws, triangles, vertex_colors=albedo)
        mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step)))

        logging.info('End')

    def interpolate_view(self, img_idx_0, img_idx_1):
        images = []
        n_frames = 60
        for i in range(n_frames):
            print(i)
            images.append(self.render_novel_image(img_idx_0,
                                                  img_idx_1,
                                                  np.sin(((i / n_frames) - 0.5) * np.pi) * 0.5 + 0.5,
                          resolution_level=4))
        for i in range(n_frames):
            images.append(images[n_frames - i - 1])

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        video_dir = os.path.join(self.base_exp_dir, 'render')
        os.makedirs(video_dir, exist_ok=True)
        h, w, _ = images[0].shape
        writer = cv.VideoWriter(os.path.join(video_dir,
                                             '{:0>8d}_{}_{}.mp4'.format(self.iter_step, img_idx_0, img_idx_1)),
                                fourcc, 30, (w, h))

        for image in images:
            writer.write(image)

        writer.release()


if __name__ == '__main__':
    print('Hello Wooden')

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='')
    parser.add_argument('--no_albedo', default=False, action="store_true")
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)

    # BASE
    if args.mode == 'train_rnb' :
        runner = Runner(args.conf, args.mode, args.case, args.is_continue, args.no_albedo)
        runner.train_rnb()

    elif args.mode == 'validate_mesh':
        runner = Runner(args.conf, args.mode, args.case, args.is_continue)
        runner.validate_mesh(world_space=True, resolution=512, threshold=args.mcube_threshold)
    elif args.mode == 'validate_mesh_texture':
        runner = Runner(args.conf, args.mode, args.case, args.is_continue)
        runner.validate_mesh_texture(world_space=True, resolution=512, threshold=args.mcube_threshold)
    elif args.mode == "validate_image_ps" :
        runner = Runner(args.conf, args.mode, args.case, args.is_continue)
        runner.validate_image_ps()
    elif args.mode == "validate_image_normal_integration" :
        runner = Runner(args.conf, args.mode, args.case, args.is_continue)
        runner.validate_image_normal_integration()

    # CVPR 2024

    elif args.mode == 'train_RnB_lights_optimal' :
        runner = Runner(args.conf, args.mode, args.case, args.is_continue)
        runner.train_RnB_lights_optimal()
        runner.validate_mesh(world_space=True, resolution=512, threshold=args.mcube_threshold)
    elif args.mode == 'train_RnB_lights_optimal_worelu' :
        runner = Runner(args.conf, args.mode, args.case, args.is_continue)
        runner.train_RnB_lights_optimal_worelu()
        runner.validate_mesh(world_space=True, resolution=512, threshold=args.mcube_threshold)
    elif args.mode == 'train_mvps_normal_integration':
        runner = Runner(args.conf, args.mode, args.case, args.is_continue)
        runner.train_mvps_normal_integration()
        runner.validate_mesh(world_space=True, resolution=512, threshold=args.mcube_threshold)
    elif args.mode == 'train_mvps_normal_integration_light_optimal':
        runner = Runner(args.conf, args.mode, args.case, args.is_continue)
        runner.train_mvps_normal_integration_light_optimal()
        runner.validate_mesh(world_space=True, resolution=512, threshold=args.mcube_threshold)
    elif args.mode == 'train_mvps_normal_integration_light_optimal_worelu':
        runner = Runner(args.conf, args.mode, args.case, args.is_continue)
        runner.train_mvps_normal_integration_light_optimal_worelu()
        runner.validate_mesh(world_space=True, resolution=512, threshold=args.mcube_threshold)
