import torch
from easydict import EasyDict as edict
import os
from torch.utils.data import DataLoader
import time
import tqdm
import imageio
from collections import OrderedDict
import socket
import numpy as np
import re
import math

import misc
import datasets
import models
import options

from misc.utils import log, visualize_depth
from misc.metrics import EvalTools
from misc.train_helpers import summarize_metrics, summarize_loss, set_requires_grad
from datasets import datas_dict
from models import models_dict
from misc import utils
#sim3
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True, stride=None):
    mu1 = F.conv2d(img1, window, padding = (window_size-1)//2, groups = channel, stride=stride)
    mu2 = F.conv2d(img2, window, padding = (window_size-1)//2, groups = channel, stride=stride)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = (window_size-1)//2, groups = channel, stride=stride) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = (window_size-1)//2, groups = channel, stride=stride) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = (window_size-1)//2, groups = channel, stride=stride) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 3, size_average = True, stride=3):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.stride = stride
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        """
        img1, img2: torch.Tensor([b,c,h,w])
        """
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average, stride=self.stride)


def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)
class S3IM(torch.nn.Module):
    r"""Implements Stochastic Structural SIMilarity(S3IM) algorithm.
    It is proposed in the ICCV2023 paper  
    `S3IM: Stochastic Structural SIMilarity and Its Unreasonable Effectiveness for Neural Fields`.

    Arguments:
        kernel_size (int): kernel size in ssim's convolution(default: 4)
        stride (int): stride in ssim's convolution(default: 4)
        repeat_time (int): repeat time in re-shuffle virtual patch(default: 10)
        patch_height (height): height of virtual patch(default: 64)
        patch_width (height): width of virtual patch(default: 64)
    """
    def __init__(self, kernel_size=4, stride=4, repeat_time=10, patch_height=32, patch_width=32):
        super(S3IM, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.repeat_time = repeat_time
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.ssim_loss = SSIM(window_size=self.kernel_size, stride=self.stride)
    def forward(self, src_vec, tar_vec):
        src_vec = src_vec.squeeze(0)
        tar_vec = tar_vec.squeeze(0)
        loss = 0.0
        index_list = []
        for i in range(self.repeat_time):
            if i == 0:
                tmp_index = torch.arange(len(tar_vec))
                index_list.append(tmp_index)
            else:
                ran_idx = torch.randperm(len(tar_vec))
                index_list.append(ran_idx)
        res_index = torch.cat(index_list)
        tar_all = tar_vec[res_index]
        src_all = src_vec[res_index]
        #print("111222")
        #print(tar_all.size()) #10240 ,3
        #print(src_all.size()) #10240 ,3
        #print("111")
        tar_patch = tar_all.permute(1, 0).reshape(1, 3, self.patch_height, self.patch_width * self.repeat_time)
        src_patch = src_all.permute(1, 0).reshape(1, 3, self.patch_height, self.patch_width * self.repeat_time) #permute
        loss = (1 - self.ssim_loss(src_patch, tar_patch))
        return loss

#sim3
class Coach():
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.n_src_views = opts.n_src_views
        self.epoch_start = 0
        self.iter_start = 0
        os.makedirs(opts.output_path, exist_ok=True)

    def MSE_loss(self, pred, label=0):
        loss = (pred.contiguous() - label) ** 2
        return loss.mean()

    def load_dataset(self, splits):
        # load training data
        log.info(f"loading datasets...")
        for split in splits:
            if getattr(self.opts, f'data_{split}', None):
                if split == 'test':
                    data_opts_list = [v for _, v in self.opts.data_test.items()]
                    self.test_loaders = []
                else:
                    data_opts_list = [getattr(self.opts, f'data_{split}')]

                for data_opts in data_opts_list:
                    if data_opts is None:
                        continue
                    scene_list = getattr(data_opts, "scene_list", None)
                    test_views_method = getattr(data_opts, "test_views_method", "nearest")
                    nf_mode = getattr(data_opts, 'nf_mode', 'avg')
                    eval_mode = getattr(data_opts, 'eval_mode', 'mvsnerf')
                    n_add_train_views = getattr(data_opts, 'n_add_train_views', 2)
                    cur_dataset = datas_dict[data_opts.dataset_name](data_opts.root_dir, split,
                                                                     n_views=self.n_src_views, img_wh=data_opts.img_wh, max_len=data_opts.max_len,
                                                                     scene_list=scene_list, test_views_method=test_views_method,
                                                                     nf_mode=nf_mode, eval_mode=eval_mode, n_add_train_views=n_add_train_views)

                    bs = self.opts.batch_size
                    if split == 'train' and 'rays' in data_opts.dataset_name:
                        bs = self.opts.nerf.rand_rays_train
                    cur_loader = DataLoader(cur_dataset, shuffle=(split == 'train'), num_workers=data_opts.num_workers,
                                            batch_size=bs, pin_memory=True)
                    if split == 'test':
                        self.test_loaders.append(cur_loader)
                    else:
                        setattr(self, f"{split}_loader", cur_loader)
                    log.info(f"  * loaded {split} set of {data_opts.dataset_name}")

    def build_networks(self):
        log.info("building networks...")
        self.model = models_dict[self.opts.model](self.opts).to(self.opts.device)
        if self.opts.encoder.pretrain_weight and (not self.opts.load) and (not self.opts.resume):
            utils.load_gmflow_checkpoint(self.model.feat_enc, self.opts.encoder.pretrain_weight, self.opts.device,
                                         gmflow_n_blocks=self.opts.encoder.num_transformer_layers)
            log.info(f"loaded gmflow pretrained weight for encoder from {self.opts.encoder.pretrain_weight}.")

        if len(self.opts.gpu_ids) > 1:  # Use multi GPU training
            self.model.feat_enc = torch.nn.DataParallel(self.model.feat_enc, self.opts.gpu_ids)
            self.model.nerf_dec = torch.nn.DataParallel(self.model.nerf_dec, self.opts.gpu_ids)

    def setup_optimizer(self):
        log.info("setting up optimizers...")
        # load trainable params
        optim_params = []
        lr_lists = []
        if self.opts.optim.lr_enc > 0:  # do not tune encoder for per-scene fine tuning
            optim_params.append(dict(params=self.model.feat_enc.parameters(), lr=self.opts.optim.lr_enc))
            lr_lists.append(self.opts.optim.lr_enc)
        else:
            set_requires_grad(self.model.feat_enc, False)

        if self.opts.optim.lr_dec > 0:
            optim_params.append(dict(params=self.model.nerf_dec.parameters(), lr=self.opts.optim.lr_dec))
            lr_lists.append(self.opts.optim.lr_dec)
        else:
            set_requires_grad(self.model.nerf_dec, False)

        # set up optimizer
        optim_type = self.opts.optim.algo.type
        optim_kwargs = {k: v for k, v in self.opts.optim.algo.items() if k != "type"}
        self.optim = getattr(torch.optim, optim_type)(optim_params, **optim_kwargs)
        info = f"  * {optim_type} optimizer (" + ', '.join([f'{k}={v}' for k, v in optim_kwargs.items()]) + ')'
        log.info(info)

        # set up scheduler if needed
        self.sched_type = None
        if self.opts.optim.sched:
            sched_type = self.opts.optim.sched.type
            sched_kwargs = {k: v for k, v in self.opts.optim.sched.items() if k != "type"}
            info = f"  * {sched_type} scheduler"
            if sched_type == 'OneCycleLR':  # set additional param accordingly
                assert hasattr(self, 'train_loader'), "Must initialize the training data, to calculate total steps for OneCycleLR"
                steps_per_epoch = len(self.train_loader) // (self.opts.batch_size // len(self.opts.gpu_ids))
                sched_kwargs.update(dict(epochs=self.opts.max_epoch, steps_per_epoch=steps_per_epoch, max_lr=lr_lists))

            self.sched_type = sched_type
            self.sched = getattr(torch.optim.lr_scheduler, sched_type)(self.optim, **sched_kwargs)
            info = info + ' (' + ', '.join([f'{k}={v}' for k, v in sched_kwargs.items()]) + ')'
            log.info(info)

    def restore_checkpoint(self):
        epoch_start, iter_start = 0, 0
        if self.opts.resume:
            log.info("resuming from previous checkpoint...")
            ckpt_path = os.path.join(self.opts.output_path, 'models', 'latest.pth')
            if not os.path.isfile(ckpt_path):
                log.warn(f"can NOT find previous checkpoints at {ckpt_path}")
                log.warn("start training from scratch.")
            else:
                optims_scheds = {x: getattr(self, x) for x in ['optim', 'sched'] if hasattr(self, x)}
                epoch_start, iter_start = utils.restore_checkpoint(self.model, ckpt_path=ckpt_path,
                                                                   device=self.opts.device, log=log, resume=True,
                                                                   optims_scheds=optims_scheds)
        elif self.opts.load is not None:
            log.info("loading weights from checkpoint {}...".format(self.opts.load))
            epoch_start, iter_start = utils.restore_checkpoint(self.model, ckpt_path=self.opts.load, device=self.opts.device, log=log)
        else:
            log.info("initializing weights from scratch...")
        self.epoch_start = epoch_start or 0
        self.iter_start = iter_start or 0

    def setup_visualizer(self):
        log.info("setting up visualizers...")
        if self.opts.tb:
            from torch.utils import tensorboard
            self.tb = tensorboard.SummaryWriter(log_dir=self.opts.output_path, flush_secs=10)

    def train_model(self):
        # before training
        log.title("TRAINING START")
        self.timer = edict(start=time.time(), it_mean=None)
        self.it = self.iter_start
        self.ep = self.epoch_start
        self.val_it = math.ceil(self.opts.freq.val_it * len(self.train_loader)) if self.opts.freq.val_it > 0 else self.opts.freq.val_it
        self.test_it = math.ceil(self.opts.freq.test_it * len(self.train_loader)) if self.opts.freq.test_it > 0 else self.opts.freq.test_it
        self.ckpt_it = math.ceil(self.opts.freq.ckpt_it * len(self.train_loader)) if self.opts.freq.ckpt_it > 0 else self.opts.freq.ckpt_it

        # training
        if getattr(self.opts, "sanity_check", False) and self.it == 0:
            if self.val_it > 0:
                self.validate_model(iter=self.it, is_sanity_check=True)
            if self.opts.freq.test_ep > 0:
                self.test_model(ep=0, send_log=False, save_images=False, is_sanity_check=True)

        for self.ep in range(self.epoch_start, self.opts.max_epoch):
            self.train_epoch()

        # after training
        if self.opts.tb:
            self.tb.flush()
            self.tb.close()
        log.title("TRAINING DONE")

    def train_epoch(self):
        # before train epoch
        self.model.train()
        # train epoch
        tqdm_bar = tqdm.tqdm(self.train_loader, desc="training epoch {}".format(self.ep + 1), leave=False)
        for batch_idx, batch in enumerate(tqdm_bar):
            # train iteration
            if self.opts.resume and self.ep * len(self.train_loader) + batch_idx < self.iter_start:
                continue

            var = edict(batch)
            var = utils.move_to_device(var, self.opts.device)
            loss = self.train_iteration(var)
            tqdm_bar.set_postfix(it=self.it, loss="{:.3f}".format(loss.all))

            if self.sched_type == 'OneCycleLR':
                self.sched.step()

        # after train epoch
        lr_dict = self.get_cur_lrates()
        if self.opts.freq.log_ep > 0 and (self.ep + 1) % self.opts.freq.log_ep == 0:
            log.loss_train(self.opts, self.ep+1, lr_dict, loss.all, self.timer)

        if self.sched_type is not None and self.sched_type != 'OneCycleLR':
            self.sched.step()

        if self.opts.freq.val_ep > 0 and (self.ep + 1) % self.opts.freq.val_ep == 0:
            self.validate_model(iter=self.it)

        if self.ep >= self.opts.freq.test_ep_start and self.opts.freq.test_ep > 0 and (self.ep + 1) % self.opts.freq.test_ep == 0:
            self.test_model(ep=self.ep+1, send_log=True, save_images=self.opts.save_test_image)

        if self.opts.freq.ckpt_ep > 0 and (self.ep + 1) % self.opts.freq.ckpt_ep == 0:
            self.save_checkpoint(ep=self.ep+1, it=self.it, backup_ckpt=True)

    def train_iteration(self, var):
        # before train iteration
        self.timer.it_start = time.time()

        # train iteration
        self.optim.zero_grad()
        var_pred = self.model(var, mode="train")
        loss = self.compute_loss(var_pred, var, mode="train")
        loss = summarize_loss(loss, self.opts.loss_weight)
        loss.all.backward()
        if self.opts.optim.clip_enc is not None:
            torch.nn.utils.clip_grad_norm_(self.model.feat_enc.parameters(), self.opts.optim.clip_enc)
        self.optim.step()

        # after train iteration
        self.it += 1
        self.timer.it_end = time.time()
        utils.update_timer(self.opts, self.timer, self.ep, len(self.train_loader))
        if self.opts.freq.scalar > 0 and self.it % self.opts.freq.scalar == 0:
            cur_lrates = self.get_cur_lrates()
            self.log_scalars(loss, self.opts.loss_weight, lrates=cur_lrates, step=self.it, split="train")
        if self.ckpt_it > 0 and self.it % self.ckpt_it == 0:
            self.save_checkpoint(ep=self.ep, it=self.it, backup_ckpt=False)
        if self.val_it > 0 and self.it % self.val_it == 0:
            self.validate_model(iter=self.it)
        if self.test_it > 0 and self.it % self.test_it == 0:
            self.test_model(ep=self.ep, send_log=True, save_images=self.opts.save_test_image)

        return loss

    def compute_loss(self, pred, src, mode=None):
        loss = edict()
        if 'train_color' in src:
            target_gt = src['train_color']
        else:
            batch_size, n_views, n_chnl = src.images.shape[:3]
            #print(src.images.size())
            #print("rrr")
            assert n_views == (self.n_src_views + 1), "Make sure the last views are provided as the GT target view"
            target_gt = src.images[:, -1].reshape(batch_size, n_chnl, -1).permute(0, 2, 1)  # (b, h*w, 3)
            if getattr(self.opts.nerf, f"rand_rays_{mode}") and mode in ["train"]:
                target_gt = target_gt[:, pred.ray_idx]
        # compute image losses
        if self.opts.loss_weight.render is not None:
            a = pred.rgb / 255.
            b = target_gt / 255.
            a = a ** 2.2
            b = b ** 2.2
            a = torch.log(a + 0.1)
            b = torch.log(b + 0.1)
            simloss=S3IM()

            loss.render = self.MSE_loss(pred.rgb, target_gt)+0.1*self.MSE_loss(a, b)+0.1*simloss(pred.rgb, target_gt)


        return loss

    @torch.no_grad()
    def log_scalars(self, loss=None, loss_weight=None, metric=None, lrates=None, step=0, split="train"):
        if loss is not None:
            for key, value in loss.items():
                if key == "all":
                    continue
                if loss_weight[key] is not None:
                    self.tb.add_scalar("{0}/loss_{1}".format(split, key), value, step)
        if metric is not None:
            for key, value in metric.items():
                mean_value = np.array(value).mean()
                self.tb.add_scalar("{0}/{1}".format(split, key), mean_value, step)
        if lrates is not None:
            for key, value in lrates.items():
                self.tb.add_scalar("{0}/{1}".format('lrate', key), value, step)

    @torch.no_grad()
    def get_cur_lrates(self):
        lr_enc = self.opts.optim.lr_enc
        lr_dec = self.opts.optim.lr_dec
        if self.opts.optim.sched:
            if lr_enc > 0:
                lr_enc = self.sched.get_last_lr()[0]
            if lr_dec > 0:
                lr_dec = self.sched.get_last_lr()[-1]
        lr_dict = dict(enc=lr_enc, dec=lr_dec)

        return lr_dict

    def save_checkpoint(self, ep=0, it=0, backup_ckpt=True):
        save_train_info = True

        checkpoint = dict(model=self.model.state_dict())
        if save_train_info:
            train_info = dict(optim=self.optim.state_dict())
            if self.sched_type is not None:
                train_info.update(dict(sched=self.sched.state_dict()))
            checkpoint.update(train_info)

        utils.save_checkpoint(self.opts.output_path, checkpoint, ep=ep, it=it, backup_ckpt=backup_ckpt)

    def send_results(self, msg, reset_status=False, log_msg=True):
        if hasattr(self, "tg_trainnet_bot"):
            if reset_status:
                self.tg_trainnet_bot.reset_msg()
            # first message, append machine and experiment name
            if self.tg_trainnet_bot.msg_id is None and len(self.tg_trainnet_bot.msg_text) == 0:
                header = '<b>#%s #%s</b>\n<b>%s</b>, ' % (socket.gethostname(), self.opts.name.replace('/', '_'), time.strftime("%m%d-%H:%M"))
            else:
                header = '<b>%s</b>, ' % (time.strftime("%m%d-%H:%M"))
            msg = header + msg
            self.tg_trainnet_bot(msg)
        if log_msg:
            log.metric_test(re.sub('<[^<]+?>', '', msg.split('\n')[-1]))

    @torch.no_grad()
    def validate_model(self, iter=None, is_sanity_check=False):
        assert hasattr(self, 'val_loader'), "please load validation dataset."
        self.model.eval()
        data_outdir = os.path.join(self.opts.output_path, 'validation')
        os.makedirs(data_outdir, exist_ok=True)
        eval_tools = EvalTools(device=self.opts.device)
        metrics_dict = {k: [] for k in eval_tools.support_metrics}

        tqdm_loader = tqdm.tqdm(self.val_loader, desc="validating", leave=False)
        for batch_idx, batch in enumerate(tqdm_loader):
            if is_sanity_check and batch_idx > 0:
                break

            var = edict(batch)
            batch_size = var.images.shape[0]
            var = utils.move_to_device(var, self.opts.device)
            var_pred = self.model(var, mode="val")

            # save image and depth
            img_hw = batch['img_wh'][0].numpy().tolist()[::-1]
            pred_rgb = var_pred['rgb'].reshape(batch_size, *img_hw, -1)
            pred_depth = var['depth'].reshape(batch_size, *img_hw)
            for batch_idx, cur_rgb in enumerate(pred_rgb):
                pred_rgb_nb = (cur_rgb.detach().cpu().numpy() * 255).astype('uint8')
                gt_rgb_nb = (var.images[batch_idx, -1].permute(1, 2, 0).detach().cpu().numpy() * 255).astype('uint8')
                # visualize depth
                minmax = batch['near_fars'][batch_idx, -1].detach().cpu().numpy().tolist()
                depth_vis = visualize_depth(pred_depth[batch_idx], minmax)[0]
                depth_vis = (depth_vis.permute(1, 2, 0).detach().cpu().numpy() * 255).astype('uint8')
                img_vis = np.concatenate([depth_vis, pred_rgb_nb, gt_rgb_nb], axis=1)

                out_name = f"{batch['scene'][batch_idx]}_view{batch['view_ids'][batch_idx][-1]}_it{iter}.jpg"
                imageio.imwrite(os.path.join(data_outdir, out_name), img_vis)

            for batch_idx, cur_rgb in enumerate(pred_rgb):
                pred_rgb_nb = cur_rgb.detach().cpu().numpy()
                gt_rgb_nb = var.images[batch_idx, -1].permute(1, 2, 0).detach().cpu().numpy()  # h,w,3
                if 'dtu' in self.val_loader.dataset.get_name():
                    assert 'depth' in batch, "Must provide 'depth' of target view for validation"
                    depth = batch['depth'][batch_idx].detach().cpu().numpy()
                    image_mask = depth == 0
                else:
                    image_mask = None
                eval_tools.set_inputs(pred_rgb_nb, gt_rgb_nb, image_mask)
                cur_metrics = eval_tools.get_metrics()
                for k, v in cur_metrics.items():
                    metrics_dict[k].append(v)

        self.log_scalars(metric=metrics_dict, step=iter, split="val")
        self.model.train()

    @torch.no_grad()
    def test_model(self, ep=None, send_log=True, save_images=True, leave_tqdm=False, is_sanity_check=False):
        assert hasattr(self, 'test_loaders'), "Must load the test data for testing."
        test_outroot = os.path.join(self.opts.output_path, 'test')
        os.makedirs(test_outroot, exist_ok=True)
        eval_tools = EvalTools(device=self.opts.device)
        metrics_dict = {}

        self.model.eval()
        for data_loader in self.test_loaders:
            dataname = data_loader.dataset.get_name()
            metrics_dict[dataname] = OrderedDict()
            data_outdir = os.path.join(test_outroot, dataname)
            os.makedirs(data_outdir, exist_ok=True)
            if dataname == 'blender':
                self.model.nerf_setbg_opaque = True

            tqdm_desc = f"testing {dataname}" if ep is None else f"testing {dataname} [epoch {ep}]"
            for batch_idx, batch in enumerate(tqdm.tqdm(data_loader, desc=tqdm_desc, leave=leave_tqdm)):
                if is_sanity_check and batch_idx > 0:
                    break

                var = edict(batch)
                var = utils.move_to_device(var, self.opts.device)
                var = self.model(var, mode="test")

                # save image
                batch_size = var['images'].shape[0]
                img_hw = batch['img_wh'][0].numpy().tolist()[::-1]
                pred_rgb = var['rgb'].reshape(batch_size, *img_hw, -1)
                pred_depth = var['depth'].reshape(batch_size, *img_hw)
                if save_images:
                    for batch_idx, cur_rgb in enumerate(pred_rgb):
                        pred_rgb_nb = (cur_rgb.detach().cpu().numpy() * 255).astype('uint8')
                        gt_rgb_nb = (var.images[batch_idx, -1].permute(1, 2, 0).detach().cpu().numpy() * 255).astype('uint8')
                        # visualize depth
                        if self.opts.vis_depth:
                            minmax = batch['near_fars'][batch_idx, -1].detach().cpu().numpy().tolist()
                            depth_vis = visualize_depth(pred_depth[batch_idx], minmax)[0]
                            depth_vis = (depth_vis.permute(1, 2, 0).detach().cpu().numpy() * 255).astype('uint8')
                            img_vis = np.concatenate([depth_vis, pred_rgb_nb, gt_rgb_nb], axis=1)
                        else:
                            img_vis = np.concatenate([pred_rgb_nb, gt_rgb_nb], axis=1)

                        src_ids_str = '_'.join([f'{x:02d}' for x in batch['view_ids'][batch_idx][:self.n_src_views]])
                        out_name = f"{batch['scene'][batch_idx]}_view{batch['view_ids'][batch_idx][-1]:02d}_src{src_ids_str}.jpg"
                        if hasattr(self, 'it'):
                            out_name = f"it{self.it}_{out_name}"
                        if ep is not None:
                            out_name = f"ep{ep}_{out_name}"
                        imageio.imwrite(os.path.join(data_outdir, out_name), img_vis)

                # log metric
                for batch_idx, cur_rgb in enumerate(pred_rgb):
                    pred_rgb_nb = cur_rgb.detach().cpu().numpy()
                    gt_rgb_nb = var.images[batch_idx, -1].permute(1, 2, 0).detach().cpu().numpy()  # h,w,3
                    if 'depth' in batch:
                        depth = batch['depth'][batch_idx].detach().cpu().numpy()
                        image_mask = depth == 0
                    else:
                        image_mask = None
                    eval_tools.set_inputs(pred_rgb_nb, gt_rgb_nb, image_mask)
                    report_full_scores = getattr(getattr(self.opts.data_test, dataname), "report_full_scores", False)
                    cur_metrics = eval_tools.get_metrics(return_full=report_full_scores)
                    pred_img_id = f"{var.scene[batch_idx]}_{var.view_ids[batch_idx, -1]:03d}"
                    metrics_dict[dataname][pred_img_id] = cur_metrics
                    # print(f"{var.scene[batch_idx]}_{var.view_ids[batch_idx, -1]:03d}", cur_metrics)
            # reset params
            self.model.nerf_setbg_opaque = False
        sum_dict = summarize_metrics(metrics_dict, test_outroot, ep=ep)
        log_msg = f"{self.ep:02d},{self.it:06d};" if hasattr(self, 'ep') and hasattr(self, 'it') else ""
        for cur_dataname, cur_datametric in sum_dict.items():
            metric_avg = {k: np.array(v).mean() for k, v in cur_datametric.items()}
            log_msg = log_msg + f" {cur_dataname.upper()[0]}: {metric_avg['PSNR']:.2f}, {metric_avg['SSIM']:.3f}, {metric_avg['LPIPS']:.3f},"
            if hasattr(self, 'tb'):
                self.log_scalars(metric=metric_avg, step=ep, split=cur_dataname)
        log.metric_test(re.sub('<[^<]+?>', '', log_msg.split('\n')[-1]))
        self.model.train()

    @torch.no_grad()
    def test_model_video(self, ep=None, leave_tqdm=False):
        assert hasattr(self, 'test_loaders'), "Must load the test data for testing."
        test_outroot = os.path.join(self.opts.output_path, 'test_videos')
        os.makedirs(test_outroot, exist_ok=True)

        self.model.eval()
        for data_loader in self.test_loaders:
            dataname = data_loader.dataset.get_name()
            data_outdir = os.path.join(test_outroot, dataname)
            os.makedirs(data_outdir, exist_ok=True)

            # set rendering parameters
            if 'dtu' in dataname:
                self.model.nerf_setbg_opaque = False
                render_path_mode = 'interpolate'
            elif dataname == 'blender':
                self.model.nerf_setbg_opaque = True
                render_path_mode = 'interpolate'
            elif dataname == 'llff':
                self.model.nerf_setbg_opaque = False
                render_path_mode = 'spiral'
            elif dataname == 'colmap':
                self.model.nerf_setbg_opaque = False
                render_path_mode = self.opts.data_test.colmap.render_path_mode
            else:
                raise Exception(f"Unknown dataset for rendering video {dataname}")

            tqdm_desc = f"testing {dataname}" if ep is None else f"testing {dataname} [epoch {ep}]"
            for batch in tqdm.tqdm(data_loader, desc=tqdm_desc, leave=leave_tqdm):
                var = edict(batch)
                var = utils.move_to_device(var, self.opts.device)
                var = self.model(var, mode="test",
                                 render_video=self.opts.nerf.render_video, render_path_mode=render_path_mode)

                # save videos and images
                batch_size = var['images'].shape[0]
                img_hw = batch['img_wh'][0].numpy().tolist()[::-1]
                pred_rgb = var['rgb'].reshape(batch_size, self.opts.nerf.video_n_frames, *img_hw, -1)
                pred_depth = var['depth'].reshape(batch_size, self.opts.nerf.video_n_frames, *img_hw)
                for batch_idx, cur_rgb in enumerate(pred_rgb):
                    pred_rgb_nb = (cur_rgb.detach().cpu().numpy() * 255).astype('uint8')
                    if self.opts.vis_depth:
                        minmax = batch['near_fars'][batch_idx, -1].detach().cpu().numpy().tolist()
                        depth_vis = []
                        for pred_depth_frame in pred_depth[batch_idx]:
                            cur_depth_vis = visualize_depth(pred_depth_frame, minmax)[0]
                            cur_depth_vis = (cur_depth_vis.permute(1, 2, 0).detach().cpu().numpy() * 255).astype('uint8')
                            depth_vis.append(cur_depth_vis)
                        depth_vis = np.stack(depth_vis, axis=0)
                        img_vis = np.concatenate([pred_rgb_nb, depth_vis], axis=2)
                    else:
                        img_vis = pred_rgb_nb

                    src_ids_str = '_'.join([f'{x:02d}' for x in batch['view_ids'][batch_idx][:self.n_src_views]])
                    out_name = f"{batch['scene'][batch_idx]}_view{batch['view_ids'][batch_idx][-1]:02d}_src{src_ids_str}"
                    if ep is not None:
                        out_name = f"ep{ep}_{out_name}"
                    pred_rgb_nb_list = [img_vis[x] for x in range(pred_rgb_nb.shape[0])]
                    # save frames
                    if getattr(self.opts.nerf, "save_frames", False):
                        for f_idx, frame in enumerate(pred_rgb_nb_list):
                            imageio.imwrite(os.path.join(data_outdir, f"{out_name}_f{f_idx}.jpg"), frame)
                    # save video
                    utils.write_video(os.path.join(data_outdir, f"{out_name}.mp4"), pred_rgb_nb_list,
                                      getattr(self.opts.nerf, "video_pts_rates", 2.0))
                    # save gif if needed
                    if getattr(self.opts.nerf, "save_gif", False):
                        imageio.mimsave(os.path.join(data_outdir, f"{out_name}.gif"), pred_rgb_nb_list, fps=12)
                    # save the src images for reference
                    imgs_src_vis = batch['images'][batch_idx, :self.n_src_views].detach().permute(0, 2, 3, 1).cpu().numpy() * 255
                    imgs_src_vis = np.concatenate([imgs_src_vis[i] for i in range(self.n_src_views)], axis=1).astype('uint8')
                    imageio.imwrite(os.path.join(data_outdir, f"{out_name}.jpg"), imgs_src_vis)

        self.model.train()
