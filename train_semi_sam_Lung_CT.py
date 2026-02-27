import os
import math

os.environ['CUDA_VISIBLE_DEVICES'] = "2"
import torch.optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset, Dataset, SubsetRandomSampler
import torch.utils.data
import random
import shutil
from skimage import measure
import skimage
import logging
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
import matplotlib
from segment_anything import sam_model_registry
import matplotlib.patches as patches

matplotlib.use('Agg')
import matplotlib.pylab as plt
from torchvision.utils import make_grid
from skimage import measure, io
from augmention import generate_unsup_data
from CE_Net import Our_Semic_Seg
from torch.nn import functional as F
import numpy as np
import utils
from data_folder import DataFolder, PairedUnlabeledDataset, SAMPreprocess, PairedUnlabeledWithSAMMask, \
    PairedUnlabeledWithSAMMask_v1
from hausdorff_loss import HausdorffERLoss
from options_semi_Lung_CT import Options
from my_transforms import get_transforms
from loss import LossVariance, dice_loss, FlowLoss
from torch import nn
from copy import deepcopy
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import cv2
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
from torch.cuda.amp import GradScaler, autocast
import yaml
import argparse
from utils2 import clip_grad_norm, bits_per_dim
from functools import partial
from attack import attack
from scipy.signal import find_peaks
from torchvision.ops import box_iou
from PIL import Image

medsam_model = sam_model_registry["vit_b"](checkpoint="work_dir/MedSAM/medsam_vit_b.pth")
medsam_model = medsam_model.cuda()
medsam_model.eval()

shapley_weights = {'w_T': 0.5, 'w_S': 0.5}
# 更新频率：每多少个 iteration 更新一次（按你数据量调）
SHAPLEY_UPDATE_ITERS = 10  # 例如每 300 iter 更新一次
SHAPLEY_WARMUP_EPOCH = 30 # 前 300 epoch 不更新
SHAPLEY_UNLAB_BATCHES = 5  # 每次估计聚合多少个 unlabeled batch（降方差）
SHAPLEY_FINETUNE_STEPS = 3

# 平滑（EMA）参数
SHAPLEY_EMA_BETA = 0.1  # 越小越稳（0.05~0.2）
SHAPLEY_CLIP_LO, SHAPLEY_CLIP_HI = 0.2, 0.8  # 防止极端 0.99/0.01


def shapley_smooth_update(new_w_T: float, beta=SHAPLEY_EMA_BETA,
                          lo=SHAPLEY_CLIP_LO, hi=SHAPLEY_CLIP_HI):
    """
    new_w_T: 本次估计出来的 raw 权重（0~1）
    返回：更新后的 (w_T, w_S)
    """
    global shapley_weights
    # 1) clip raw
    new_w_T = float(max(lo, min(hi, new_w_T)))
    new_w_S = 1.0 - new_w_T

    # 2) EMA 平滑
    old_w_T = float(shapley_weights.get('w_T', 0.5))
    w_T = (1 - beta) * old_w_T + beta * new_w_T
    w_T = float(max(lo, min(hi, w_T)))
    w_S = 1.0 - w_T

    shapley_weights['w_T'] = w_T
    shapley_weights['w_S'] = w_S
    return w_T, w_S


def get_random_loader(dataset, num_samples, batch_size=8, num_workers=4):
    indices = random.sample(range(len(dataset)), num_samples)
    sampler = SubsetRandomSampler(indices)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                        num_workers=num_workers, drop_last=False)
    return loader


def intersect_boxes(clip_boxes, student_boxes, iou_thresh=0.3):
    if len(clip_boxes) == 0 or len(student_boxes) == 0:
        return []

    clip_tensor = torch.tensor(clip_boxes).float()
    student_tensor = torch.tensor(student_boxes).float()

    iou = box_iou(student_tensor, clip_tensor)  # [M, N]

    consistent_boxes = []
    for i in range(student_tensor.size(0)):
        if iou[i].max() > iou_thresh:
            consistent_boxes.append(student_boxes[i])
    return consistent_boxes


class DataFolderSingle(Dataset):
    def __init__(self, img_path, weight_path, label_path, transform):
        self.img_path = img_path
        self.weight_path = weight_path
        self.label_path = label_path
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        img = Image.open(self.img_path).convert('L')
        weight = Image.open(self.weight_path).convert('L')
        label = Image.open(self.label_path).convert('L')
        img, weight, label = self.transform([img, weight, label])
        return img, weight, label


def main():
    global opt, best_iou, num_iter, tb_writer, logger, logger_results, cfg, scaler_nf, scaler, best_iou
    best_iou = 0
    # cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    opt = Options(isTrain=True)
    opt.parse()
    opt.save_options()
    torch.backends.cudnn.enabled = False

    tb_writer = SummaryWriter('{:s}/tb_logs'.format(opt.train['save_dir']))

    # set up logger
    logger, logger_results = setup_logging(opt)
    opt.print_options(logger)

    model = Our_Semic_Seg(3, 3)
    teacher_model = deepcopy(model)
    for p in teacher_model.parameters():
        p.requires_grad = False

    # model = nn.DataParallel(model,device_ids=[0])
    model = model.cuda()
    teacher_model = teacher_model.cuda()

    torch.backends.cudnn.benchmark = True

    # ----- define optimizer ----- #
    optimizer = torch.optim.Adam(model.parameters(), opt.train['lr'], betas=(0.9, 0.99),
                                 weight_decay=opt.train['weight_decay'])

    # ----- define criterion ----- #


    global mseloss
    mseloss = torch.nn.MSELoss(reduction='mean').cuda()
    criterion = torch.nn.NLLLoss(reduction='none').cuda()

    if opt.train['alpha'] > 0:
        logger.info('=> Using variance term in loss...')
        global criterion_var
        criterion_var = LossVariance()

    data_transforms = {'train': get_transforms(opt.transform['train']),
                       'train1': get_transforms(opt.transform['train1']),
                       'val': get_transforms(opt.transform['val']),
                       'sam_input': get_transforms(opt.transform['sam_input']),
                       'sam_mask': get_transforms(opt.transform['sam_mask'])}

    # ----- load data ----- #
    dsets = {}
    for x in ['train', 'val']:
        img_dir = '{:s}/{:s}'.format(opt.train['img_dir'], x)
        target_dir = '{:s}/{:s}'.format(opt.train['label_dir'], x)
        dir_list = [img_dir, target_dir]
        if opt.dataset == 'Lung_CT':
            post_fix = ['anno_label.jpg']
        else:
            post_fix = ['.png']
        num_channels = [3, 3]
        dsets[x] = DataFolder(dir_list, post_fix, num_channels, data_transforms[x])

    img_dir = os.path.join(opt.train['img_dir'], 'unlabeltrain1')
    sam_input_dir = os.path.join(opt.train['img_dir'], 'sam_input')
    sam_mask_dir = os.path.join(opt.train['img_dir'], 'sam_mask')
    # Transform：一个做增强，一个只做ToTensor
    transform_aug = data_transforms['train1']  # 带随机增强
    transform_orig = SAMPreprocess(output_size=(1024, 1024))  # 只做 ToTensor

    # 加载数据
    paired_dataset = PairedUnlabeledWithSAMMask_v1(
        dir_aug=img_dir,
        dir_orig=sam_input_dir,
        dir_sam_mask=sam_mask_dir,
        transform_aug=transform_aug,
        transform_orig=transform_orig,
        transform_sam_mask=data_transforms['sam_mask']
    )
    train_loader = DataLoader(dsets['train'], batch_size=8, shuffle=True,
                              num_workers=24, drop_last=False)
    train_loader1 = DataLoader(paired_dataset, batch_size=8, shuffle=True,
                               num_workers=24, drop_last=False)

    # ----- optionally load from a checkpoint for validation or resuming training ----- #
    if opt.train['checkpoint']:
        if os.path.isfile(opt.train['checkpoint']):
            logger.info("=> loading checkpoint '{}'".format(opt.train['checkpoint']))
            checkpoint = torch.load(opt.train['checkpoint'])
            opt.train['start_epoch'] = checkpoint['epoch']
            best_iou = checkpoint['best_iou']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                        .format(opt.train['checkpoint'], checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(opt.train['checkpoint']))

    # ----- training and validation ----- #
    for epoch in range(opt.train['start_epoch'], opt.train['num_epochs']):
        # train for one epoch or len(train_loader) iterations
        logger.info('Epoch: [{:d}/{:d}]'.format(epoch + 1, opt.train['num_epochs']))

        sampled_val_loader = get_random_loader(dsets['val'], 1800)
        sampled_val_loader1 = get_random_loader(dsets['val'], 20)
        train_results = train(train_loader, train_loader1, model, teacher_model, optimizer,
                              criterion, epoch, sampled_val_loader, sampled_val_loader1)
        train_loss, train_loss_ce, train_loss_var, train_unsup_loss, train_sam_loss, train_pixel_acc, train_iou = train_results

        # evaluate on validation set
        with torch.no_grad():
            # 学生模型进行验证
            val_loss, val_pixel_acc, val_iou = validate(sampled_val_loader, model, criterion)

            # 教师模型进行验证
            val_loss_t, val_pixel_acc_t, val_iou_t = validate(sampled_val_loader, teacher_model, criterion, False)

            # 比较学生和教师模型的 val_iou，取较大者
            if val_iou >= val_iou_t:
                val_iou_a = val_iou
            else:
                val_iou_a = val_iou_t
            is_best = val_iou_a > best_iou
            best_iou = max(val_iou_a, best_iou)

            if (val_iou_a >= 0.85):
                is_second = val_iou_a
                cp_flag = (epoch + 1) % opt.train['checkpoint_freq'] == 0
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_iou': best_iou,
                    'optimizer': optimizer.state_dict(),
                }, epoch, is_best, opt.train['save_dir'], cp_flag, is_second)


        # save the training results to txt files
        logger_results.info('{:d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
                            .format(epoch + 1, train_loss, train_loss_ce, train_loss_var, train_pixel_acc,
                                    train_iou, val_loss, val_pixel_acc, val_iou))
        # tensorboard logs
        tb_writer.add_scalars('epoch_losses',
                              {'train_loss': train_loss, 'train_loss_ce': train_loss_ce,
                               'train_unsup_loss': train_unsup_loss,
                               'train_sam_loss': train_sam_loss,
                               'train_loss_var': train_loss_var, 'val_loss': val_loss}, epoch)

        # tb_writer.add_scalars('student_epoch_accuracies',
        #                       {'train_pixel_acc': train_pixel_acc, 'train_iou': train_iou,
        #                        'val_pixel_accB': val_pixel_acc, 'val_iouB': val_iou,
        #                        'val_pixel_accA': val_pixel_acc1, 'val_iouA': val_iou1}, epoch)

        tb_writer.add_scalars('student_epoch_accuracies',
                              {'train_pixel_acc': train_pixel_acc, 'train_iou': train_iou,
                               'val_pixel_acc': val_pixel_acc, 'val_iou': val_iou}, epoch)

        # tb_writer.add_scalars('iou_history_weighted_iou',
        #                       {'train_pixel_acc': train_pixel_acc, 'train_iou': train_iou,
        #                        'weighted_iou_student': weighted_iou_student,
        #                        'weighted_iou_teacher': weighted_iou_teacher}, epoch)

    tb_writer.close()




def train(train_loader, train_loader1, model, teacher_model, optimizer, criterion, epoch, sampled_val_loader,
          sampled_val_loader1):
    global shapley_weights

    if 'shapley_weights' not in globals():
        shapley_weights = {'w_T': 0.5, 'w_S': 0.5}

    ite = 0
    IsSamOutput = False
    # list to store the average loss and iou for this epoch
    results = utils.AverageMeter(7)
    # switch to train mode

    label_iter = iter(train_loader1)

    for i2, sample in enumerate(train_loader):
        ite += 1
        input, target = sample

        if torch.max(target) == 255:
            target = target / 255
        if target.dim() == 4:
            target = target.squeeze(1)
        try:
            input1, orig_img, sam_mask = next(label_iter)
        except StopIteration:
            label_iter = iter(train_loader1)
            input1, orig_img, sam_mask = next(label_iter)

        input_var = input.cuda()
        target_var = target.cuda()

        input_var1 = input1.cuda()

        # compute teacher_model,teacher_model predicts on all data
        teacher_model.eval()
        with torch.no_grad():
            out_unlabeled, output_rep = teacher_model(input_var1)
            # Get predictions of original unlabeled data

            output_ema_u = F.softmax(out_unlabeled, dim=1)
            logits_u_aug, label_u = torch.max(output_ema_u, dim=1)


            # === 只计算类别为1的置信度 ===
            confidence_maps = []
            for bi in range(output_ema_u.shape[0]):  # 遍历 batch
                class1_prob = output_ema_u[bi, 1]  # 取类别1的概率图 [H, W]
                entropy = - (class1_prob * torch.log(class1_prob + 1e-6) + (1 - class1_prob) * torch.log(
                    1 - class1_prob + 1e-6))  # binary entropy
                entropy = entropy / math.log(2)  # 归一化到 [0,1]
                confidence_map = 1 - entropy  # 置信度越高越好
                gamma = 0.5  # gamma < 1 会拉高整体置信度分布
                confidence_map = confidence_map ** gamma
                confidence_maps.append(confidence_map)
            confidence_maps = torch.stack(confidence_maps, dim=0)  # [B, H, W]

            teacher_boxes_list = get_boxes_from_mask(label_u, min_area=0)
            teacher_boxes_1024_list = normalize_boxes(teacher_boxes_list, H=512, W=512)
            if epoch < 30:
                bclip_boxes_list = get_boxes_from_mask_batch(sam_mask, min_area=0)
                bclip_boxes_1024_list = normalize_boxes(bclip_boxes_list, H=512, W=512)

            # === 每个教师 box 计算平均熵置信度（只基于类别1） ===
            box_confidences_batch = []
            for i in range(len(teacher_boxes_list)):
                boxes = teacher_boxes_list[i]
                conf_map = confidence_maps[i]  # [H, W]

                confs = []
                for box in boxes:
                    x1, y1, x2, y2 = [int(v) for v in box]
                    region = conf_map[y1:y2, x1:x2]
                    if region.numel() == 0:
                        conf = 0.0
                    else:
                        conf = region.mean().item()
                    confs.append(conf)
                # print("置信度列表（没加权图像）:", confs)
                box_confidences_batch.append(confs)

                # === 提升与 BiomedCLIP box 有重叠的 box 的置信度（仅限前300轮）===
            if epoch < 30:
                boosted_confidences_batch = []
                for i, (teacher_boxes, teacher_confs, bclip_boxes) in enumerate(
                        zip(teacher_boxes_1024_list, box_confidences_batch, bclip_boxes_1024_list)):
                    boosted_confs = []
                    for box, conf in zip(teacher_boxes, teacher_confs):
                        has_match = any(calc_iou(box, sam_box) > 0.1 for sam_box in bclip_boxes)
                        if has_match:
                            boosted_confs.append(min(conf + 0.3, 1.0))  # 提升置信度，最大不超过1.0
                        else:
                            boosted_confs.append(conf)
                    # print("置信度列表（当前图像）:", boosted_confs)
                    boosted_confidences_batch.append(boosted_confs)
            else:
                boosted_confidences_batch = box_confidences_batch  # 直接使用原始置信度，不提升
                # print("置信度列表（当前图像）NO:", boosted_confidences_batch)
            # === 置信度过滤 ===
            if epoch < 10:
                conf_thresh = 0.30  # 初期放宽
            elif epoch < 30:
                conf_thresh = 0.40
            elif epoch < 70:
                conf_thresh = 0.60
            else:
                conf_thresh = 0.65

            final_boxes_list = []
            for boxes, boosted_confs in zip(teacher_boxes_1024_list, boosted_confidences_batch):
                filtered_boxes = [box for box, conf in zip(boxes, boosted_confs) if conf >= conf_thresh]
                final_boxes_list.append(filtered_boxes)

            if not all(len(b) == 0 for b in final_boxes_list):
                with torch.no_grad():
                    image_embedding = medsam_model.image_encoder(orig_img.cuda())  # (1, 256, 64, 64)

                medsam_seg = batch_medsam_inference(medsam_model, image_embedding, final_boxes_list, H=512, W=512)
                if epoch % 10 == 0:
                    save_medsam_masks(medsam_seg, epoch,
                                      save_dir='/mnt/1abf867b-1b73-4a66-82de-c7fd1d9441b1/YJ/Lung_CT/2_label/semi_sam_Shapley_new_v2')
                IsSamOutput = True
            else:
                print("⚠️ Skipped MedSAM inference due to empty boxes in batch.")



            input_var1_aug, label_u_aug, logits_u_aug = generate_unsup_data(input_var1, label_u.clone(),
                                                                            logits_u_aug.clone(), mode="classmix")


        # compute student_model
        model.train()
        unsup_loss_pt = 0.0
        consistency_loss = 0.0

        out_labeled, out_unlabeled, res_head_l, res_head_u = model(input_var, input_var1_aug)
        log_prob_maps = F.log_softmax(out_labeled, dim=1)
        loss_map = criterion(log_prob_maps, target_var)
        loss_CE = loss_map.mean()



        unsup_loss = compute_unsupervised_loss_conf_weight(label_u_aug.clone(), 100, out_unlabeled)



        if IsSamOutput == True:
            un_log_prob_maps = F.log_softmax(out_unlabeled, dim=1)
            medsam_seg = medsam_seg.squeeze(1).long()
            sam_loss = criterion(un_log_prob_maps, medsam_seg).mean()
        else:
            sam_loss = 0

        # sam_loss = compute_unsupervised_loss_conf_weight(medsam_seg.clone(), 100, out_unlabeled)


        if opt.train['alpha'] != 0:
            prob_maps = F.softmax(out_labeled, dim=1)

            # label instances in target
            target_labeled = torch.zeros(target.size()).long()
            for k in range(target.size(0)):
                target_labeled[k] = torch.from_numpy(measure.label(target[k].numpy() == 1))
                # utils.show_figures((target[k].numpy(), target[k].numpy()==1, target_labeled[k].numpy()))
            loss_var = criterion_var(prob_maps, target_labeled.cuda())
            # print(2*loss_total_mse)
            # loss = loss_CE + opt.train['alpha'] * loss_var + unsup_loss + 0.65*sam_loss
            loss = loss_CE + opt.train['alpha'] * loss_var + shapley_weights['w_T'] * unsup_loss + shapley_weights[
                'w_S'] * sam_loss


        else:
            loss_var = torch.ones(1) * -1
            loss = loss_CE + unsup_loss

        pred = np.argmax(log_prob_maps.data.cpu().numpy(), axis=1)
        metrics = utils.accuracy_pixel_level(pred, target.numpy())
        pixel_accu, iou = metrics[0], metrics[1]


        result = [loss, loss_CE, loss_var, unsup_loss, sam_loss, pixel_accu, iou]
        results.update(result, input.size(0))
        # compute gradient and do SGD step

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        global_step = epoch * len(train_loader) + i2

        model, teacher_model = update_ema_variables(model, teacher_model, global_step=global_step, base_alpha=0.99)
        # 在 epoch 内或结束后，做周期性更新（例如每 20 epoch）
        # -------- Shapley update: low-frequency + multi-batch aggregation + EMA smoothing --------

        if epoch >= SHAPLEY_WARMUP_EPOCH and ((epoch - SHAPLEY_WARMUP_EPOCH) % 5 == 0):
            try:
                raw_w_T, raw_w_S = compute_shapley_weights_finetune(
                    model=model,
                    teacher_model=teacher_model,
                    medsam_model=medsam_model,
                    labeled_loader=train_loader,
                    unlabeled_loader=train_loader1,
                    sampled_val_loader=sampled_val_loader1,
                    criterion_ce=criterion,
                    epoch=epoch,
                    finetune_steps=SHAPLEY_FINETUNE_STEPS,
                    unlabeled_batches=SHAPLEY_UNLAB_BATCHES,  # ✅聚合多个 unlabeled batch
                    finetune_lr=1e-4,
                    device=next(model.parameters()).device
                )

                # ✅EMA 平滑 + clip 防极端
                smooth_w_T, smooth_w_S = shapley_smooth_update(raw_w_T)

                logger.info(f"[ShapleySmooth@step{global_step}] raw(wT,wS)=({raw_w_T:.3f},{raw_w_S:.3f}) "
                            f"-> smooth(wT,wS)=({smooth_w_T:.3f},{smooth_w_S:.3f})")

                tb_writer = SummaryWriter('{:s}/tb_logs'.format(opt.train['save_dir']))
                tb_writer.add_scalars('shapley_weights/raw',
                                      {'w_T': float(raw_w_T), 'w_S': float(raw_w_S)},
                                      global_step)
                tb_writer.add_scalars('shapley_weights/smooth',
                                      {'w_T': float(smooth_w_T), 'w_S': float(smooth_w_S)},
                                      global_step)

            except Exception as e:
                logger.info(f"[Shapley@step{global_step}] update failed, keep previous weights. err={e}")

                # tb_writer.close()

        del input_var, target_var, log_prob_maps, loss

        if i % opt.train['log_interval'] == 0:
            logger.info('\tIteration:[{:d}/{:d}]'
                        '\tLoss {r[0]:.4f}'
                        '\tLoss_CE {r[1]:.4f}'
                        '\tLoss_var {r[2]:.4f}'
                        '\tunsup_loss {r[3]:.4f}'
                        '\tsam_loss {r[4]:.4f}'
                        '\tpixel_accu {r[5]:.4f}'
                        '\tIoU {r[6]:.4f}'.format(i, len(train_loader), r=results.avg))

    logger.info('\t => Train_Avg:Loss_total {r[0]:.4f}'
                '\tLoss_CE {r[1]:.4f}'
                '\tLoss_var {r[2]:.4f}'
                '\tunsup_loss {r[3]:.4f}'
                '\tsam_loss {r[4]:.4f}'
                '\tpixel_accu {r[5]:.4f}'
                '\tIoU {r[6]:.4f}'.format(epoch, opt.train['num_epochs'], r=results.avg))

    return results.avg


def compute_consistency_loss(out_unlabeled, out_all_unlabeled_pt, mask=None):
    if mask is not None:
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        mse = F.mse_loss(out_unlabeled * mask, out_all_unlabeled_pt * mask, reduction='sum')
        norm = mask.sum() * out_unlabeled.shape[1]
        loss = mse / (norm + 1e-6)
    else:
        loss = F.mse_loss(out_unlabeled, out_all_unlabeled_pt)
    return loss


def compute_shapley_weights_finetune(model, teacher_model, medsam_model,
                                     labeled_loader, unlabeled_loader, sampled_val_loader,
                                     criterion_ce, epoch,
                                     finetune_steps=3, unlabeled_batches=2,
                                     finetune_lr=1e-4, device='cuda'):
    """
    近似 Shapley：基于短期（finetune_steps）在 unlabeled 数据上微调来估计效用值。
    返回：w_T, w_S （两个归一化权重）
    参数说明：
      - model, teacher_model, medsam_model: 原始模型和 teacher, sam
      - labeled_loader: 用于微调时的有标签小批（你训练里每epoch随机抽的一张可复用）
      - unlabeled_loader: DataLoader of unlabeled data (train_loader1)
      - val_loader: 验证集 DataLoader（全是有 mask 的）
      - criterion_ce: 你用的 CE 损失函数（如 variable `criterion`）
      - finetune_steps: 每种配置下的微调 step 数（通常 2-5）
      - unlabeled_batches: 每 step 使用多少个 unlabeled batch（控制多样性）
    """

    device = device if device is not None else next(model.parameters()).device

    # baseline: 当前模型在 val 上的 IoU (不微调)
    val_base = validate(sampled_val_loader, model, criterion_ce)[2]  # validate 返回 [loss, pix_acc, iou]

    # helper：短期微调并评估（loss_config 控制是否使用 teacher/sam）
    def local_finetune_and_eval(use_teacher=False, use_sam=False):
        # 复制模型和优化器
        tmp_model = deepcopy(model).to(device)
        tmp_model.train()
        tmp_opt = torch.optim.Adam(tmp_model.parameters(), lr=finetune_lr)

        # iterators
        lab_iter = iter(labeled_loader)
        unl_iter = iter(unlabeled_loader)

        any_sam_output = False  # 记录是否出现过 sam 输出（否则 sam 情况无效）

        for step in range(finetune_steps):
            # 获取 1 个 labeled（你的 train_loader 是 batch_size=1）以保持 forward 接口一致
            try:
                input_lab, target_lab = next(lab_iter)
            except StopIteration:
                lab_iter = iter(labeled_loader)
                input_lab, target_lab = next(lab_iter)

            # 获取若干 unlabeled batch（取 unlabeled_batches 个 batch 叠在一起）
            unl_inputs = []
            unl_orig_imgs = []
            unl_sam_masks = []
            for _ in range(unlabeled_batches):
                try:
                    in1, orig_img, sam_mask = next(unl_iter)
                except StopIteration:
                    unl_iter = iter(unlabeled_loader)
                    in1, orig_img, sam_mask = next(unl_iter)
                unl_inputs.append(in1)
                unl_orig_imgs.append(orig_img)
                unl_sam_masks.append(sam_mask)
            # concat along batch dim
            input_lab = input_lab.to(device)

            if torch.max(target_lab) == 255:
                target_lab = target_lab / 255
            if target_lab.dim() == 4:
                target_lab = target_lab.squeeze(1)
            target_lab = target_lab.long().to(device)

            input_unl = torch.cat([x.to(device) for x in unl_inputs], dim=0)  # [B_unl, C, H, W]
            orig_imgs = torch.cat([x.to(device) for x in unl_orig_imgs], dim=0)  # 用于 SAM
            sam_masks = torch.cat([x for x in unl_sam_masks], dim=0)  # 可能用不上

            # --- 得到 teacher 的伪标签（按你训练流程） ---
            teacher_model.eval()
            with torch.no_grad():
                out_unlabeled_teacher, _ = teacher_model(input_unl)
                # prob_teacher = F.softmax(out_unlabeled_teacher, dim=1)
                # _, label_u = torch.max(prob_teacher, dim=1)  # pseudo labels
                prob_teacher = sigmoid_prob(out_unlabeled_teacher)  # [B,H,W]
                label_u = (prob_teacher > 0.5).to(torch.uint8)  # [B,H,W] 0/1

                # 计算 boxes + medsam mask（如果需要）
                teacher_boxes_list = get_boxes_from_mask(label_u, min_area=0)
                teacher_boxes_1024_list = normalize_boxes(teacher_boxes_list, H=256, W=256)

                final_boxes_list = teacher_boxes_1024_list  # 这里只用教师 box（可复用你训练里的筛选逻辑）

            # 如果启用 SAM，则运行一次 batch_medsam_inference（可能返回空 mask）
            medsam_seg = None
            if use_sam:
                with torch.no_grad():
                    try:
                        image_embedding = medsam_model.image_encoder(orig_imgs)  # (B, C, 64, 64)
                        medsam_seg = batch_medsam_inference(medsam_model, image_embedding,
                                                            final_boxes_list, H=256, W=256)  # 返回 [B, H, W]
                        # squeeze channel and cast to long later when used
                        any_sam_output = any_sam_output or (medsam_seg is not None and medsam_seg.sum() > 0)
                    except Exception as e:
                        # SAM 可能失败或 boxes 为空；将 medsam_seg 置为 None
                        medsam_seg = None

            # --- tmp_model 前向（沿用你的 forward 接口： labeled + unlabeled） ---
            # 注意：你的 model.forward 接受 (input_var, input_var1_aug)
            # 我们这里把 labeled 作为 input_var，unlabeled 作为 input_var1_aug
            out_labeled_tmp, out_unlabeled_tmp, _, _ = tmp_model(input_lab, input_unl)

            # --- 计算损失（基于 flags） ---
            # supervised
            log_prob_maps = F.log_softmax(out_labeled_tmp, dim=1)
            loss_map = criterion_ce(log_prob_maps, target_lab)
            loss_CE = loss_map.mean()


            # teacher unsup loss
            teacher_unsup = 0.0
            if use_teacher:
                # 这里采用你已有的置信度加权无监督函数（student logits 为 out_unlabeled_tmp）
                try:
                    # 生成简单的 label_u_aug：不做 classmix，为近似用
                    # label_u is teacher argmax per pixel
                    label_u_clone = label_u.clone().long()
                    teacher_unsup = compute_unsupervised_loss_conf_weight(label_u_clone, 100, out_unlabeled_tmp)
                except Exception as e:
                    teacher_unsup = 0.0

            # sam loss
            sam_l = 0.0
            if use_sam and (medsam_seg is not None):
                try:
                    p_g = F.softmax(out_unlabeled_tmp, dim=1)[:, 1]  # [B,H,W]
                    sam_target = medsam_seg.float().to(device)  # 0/1
                    sam_l = F.binary_cross_entropy(p_g, sam_target)
                except Exception as e:
                    sam_l = 0.0

            total_loss = loss_CE + (teacher_unsup if use_teacher else 0.0) + (sam_l if use_sam else 0.0)

            # backward step on tmp_model
            tmp_opt.zero_grad()
            total_loss.backward()
            tmp_opt.step()

        # finetune done -> evaluate tmp_model on val_loader
        val_iou = validate(sampled_val_loader, tmp_model, criterion_ce)[2]
        # 如果 use_sam=True 且在 finetune 期间没有有效的 sam 输出，这个 val_iou 没什么意义
        if use_sam and (not any_sam_output):
            # 退回到 baseline（表示 SAM 在这些 unlabeled samples 上没有产出）
            return val_base
        return val_iou

    # 计算三种配置的 val score（baseline 已得）
    val_T = local_finetune_and_eval(use_teacher=True, use_sam=False)
    val_S = local_finetune_and_eval(use_teacher=False, use_sam=True)
    val_TS = local_finetune_and_eval(use_teacher=True, use_sam=True)

    # Shapley 计算（2-player closed form）
    phi_T = 0.5 * ((val_T - val_base) + (val_TS - val_S))
    phi_S = 0.5 * ((val_S - val_base) + (val_TS - val_T))

    # 防止负值 / 归一化
    phi_T = max(phi_T, 1e-6)
    phi_S = max(phi_S, 1e-6)
    w_T = phi_T / (phi_T + phi_S)
    w_S = phi_S / (phi_T + phi_S)

    # 日志
    # print(f"[Shapley@epoch{epoch}] val_base={val_base:.4f}, val_T={val_T:.4f}, val_S={val_S:.4f}, val_TS={val_TS:.4f}")
    # print(f"[Shapley@epoch{epoch}] phi_T={phi_T:.6f}, phi_S={phi_S:.6f}, w_T={w_T:.3f}, w_S={w_S:.3f}")
    logger.info(
        f"[Shapley@epoch{epoch}] "
        f"val_base={val_base:.4f}, val_T={val_T:.4f}, val_S={val_S:.4f}, val_TS={val_TS:.4f}"
    )
    logger.info(
        f"[Shapley@epoch{epoch}] "
        f"phi_T={phi_T:.6f}, phi_S={phi_S:.6f}, w_T={w_T:.3f}, w_S={w_S:.3f}"
    )

    if epoch >= 1:
        import csv, os

        val_log_path = "/mnt/1abf867b-1b73-4a66-82de-c7fd1d9441b1/YJ/Lung_CT/2_label/semi_sam_Shapley_new_v2/shapley_val_log.csv"
        phi_log_path = "/mnt/1abf867b-1b73-4a66-82de-c7fd1d9441b1/YJ/Lung_CT/2_label/semi_sam_Shapley_new_v2/shapley_phi_log.csv"

        # 如果文件不存在，则先写表头
        if not os.path.exists(val_log_path):
            with open(val_log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "val_base", "val_T", "val_S", "val_TS"])

        if not os.path.exists(phi_log_path):
            with open(phi_log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "phi_T", "phi_S", "w_T", "w_S"])

        # 写入当前epoch结果
        with open(val_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, val_base, val_T, val_S, val_TS])

        with open(phi_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, phi_T, phi_S, w_T, w_S])

    return w_T, w_S


def compute_unsupervised_loss_conf_weight(target, percent, pred_teacher):
    """
    计算伪标签的无监督损失（带置信度权重）
    ----------
    target: LongTensor, [B, H, W]
        标签(0=背景, 1=腺体, 2=轮廓)，伪标签可能覆盖其上
    percent: int
        百分比，用于筛选低置信度像素 (0-100)
    pred_teacher: FloatTensor, [B, C, H, W]
        教师模型的预测 logits
    """

    batch_size, num_class, h, w = pred_teacher.shape

    with torch.no_grad():
        # 1. softmax 得到类别概率
        prob = torch.softmax(pred_teacher, dim=1)
        conf, ps_label = torch.max(prob, dim=1)  # conf: 最大类别概率; ps_label: 预测类别索引

        conf = conf.detach()

        # 2. 计算置信度阈值
        #   percent=100 时取最小值，只剔除置信度最低的点
        conf_thresh = np.percentile(
            conf.cpu().numpy().flatten(), 100 - percent
        )

        # 3. 生成掩码：置信度 ≤ 阈值的像素被忽略
        thresh_mask = conf.le(conf_thresh).bool()

        # 标记这些像素为 ignore_index=255
        target = target.clone()
        target[thresh_mask] = 255

        # 4. 计算权重：归一化到像素总数
        valid_pixel_num = torch.sum(target != 255)
        weight = batch_size * h * w / (valid_pixel_num + 1e-6)

    # 5. 逐像素交叉熵损失，忽略 index=255 的像素
    loss_ = weight * F.cross_entropy(pred_teacher, target, ignore_index=255, reduction='none')

    # 6. 用置信度做加权（归一化到总的 valid 像素数）
    conf = (conf + 1.0) / (conf + 1.0).sum() * (valid_pixel_num + 1e-6)

    loss = torch.mean(conf * loss_)

    return loss

def validate(val_loader, model, criterion, Flag=True):
    # list to store the losses and accuracies: [loss, pixel_acc, iou ]
    results = utils.AverageMeter(3)

    # switch to evaluate mode
    model.eval()

    for i, sample in enumerate(val_loader):
        input, target = sample

        if torch.max(target) == 255:
            target = target / 255
        if target.dim() == 4:
            target = target.squeeze(1)

        target_var = target.cuda()

        size = opt.train['input_size'][0]
        overlap = opt.train['val_overlap']
        output = utils.split_forward(model, input, size, overlap, opt.model['out_c'])

        output1 = F.softmax(output, dim=1)

        log_prob_maps = F.log_softmax(output, dim=1)

        loss_map = criterion(log_prob_maps, target_var)

        loss_CE = loss_map.mean()

        if opt.train['alpha'] != 0:
            prob_maps = F.softmax(output, dim=1)

            target_labeled = torch.zeros(target.size()).long()
            for k in range(target.size(0)):
                target_labeled[k] = torch.from_numpy(measure.label(target[k].numpy() == 1))
                # utils.show_figures((target[k].numpy(), target[k].numpy()==1, target_labeled[k].numpy()))
            loss_var = criterion_var(prob_maps, target_labeled.cuda())
            loss = loss_CE + opt.train['alpha'] * loss_var
        else:
            loss = loss_CE

        # measure accuracy and record loss
        pred = np.argmax(log_prob_maps.data.cpu().numpy(), axis=1)
        metrics = utils.accuracy_pixel_level(pred, target.numpy())
        pixel_accu = metrics[0]
        iou = metrics[1]

        results.update([loss.item(), pixel_accu, iou])

        del output, target_var, log_prob_maps, loss

    if Flag == True:
        logger.info('\t=> S_Model_Val Avg:   Loss {r[0]:.4f}\tPixel_Acc {r[1]:.4f}'
                    '\tIoU {r[2]:.4f}'.format(r=results.avg))
    else:
        logger.info('\t=> T_Model_ValB Avg   Loss {r[0]:.4f}\tPixel_Acc {r[1]:.4f}'
                    '\tIoU {r[2]:.4f}'.format(r=results.avg))

    return results.avg

def get_boxes_from_mask(mask_tensor_batch, min_area=20, max_boxes_per_image=70):
    """
    输入: mask_tensor_batch: torch.Tensor, shape [B, H, W]
    输出: list of np.ndarray，每个元素 shape 为 [N_i, 4]
    """
    mask_batch = mask_tensor_batch.cpu().numpy().astype(np.uint8)
    all_boxes = []
    for mask in mask_batch:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append([x, y, x + w, y + h])
        # 限制最多框数量
        boxes = boxes[:max_boxes_per_image]
        all_boxes.append(np.array(boxes))
    return all_boxes


def get_boxes_from_mask_batch(sam_masks, min_area=20, max_boxes_per_image=70):
    """
    输入: sam_masks 是形状 [B, 3, H, W] 的 Tensor，像素值为 0. 或 1.
    输出: 每个样本提取出的 bounding boxes 列表
    """
    boxes_all = []

    for i in range(sam_masks.shape[0]):
        mask = sam_masks[i]  # 取出第 i 个 mask，形状 [3, H, W]
        mask = mask[0]  # 只取第一通道即可，形状 [H, W]
        mask = (mask * 255).cpu().numpy().astype(np.uint8)  # 转为 uint8 格式

        # findContours 要求是 C-contiguous 的
        mask = np.ascontiguousarray(mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append([x, y, x + w, y + h])
        boxes = boxes[:max_boxes_per_image]
        boxes_all.append(boxes)

    return boxes_all


def calc_iou(box1, box2):
    """
    计算两个 box 之间的 IoU（交并比）
    box1, box2: (x1, y1, x2, y2)
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # 交集的面积
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    # 各自的面积
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 并集的面积
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


# def get_boxes_from_mask(mask_tensor_batch):
#     """
#     输入: mask_tensor_batch: torch.Tensor, shape [B, H, W], 每个为伪标签 mask
#     输出: list of np.ndarray，每个元素 shape 为 [N_i, 4]，表示每个 mask 中的多个 box
#     """
#     mask_batch = mask_tensor_batch.cpu().numpy().astype(np.uint8)
#     all_boxes = []
#     for mask in mask_batch:
#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         boxes = []
#         for cnt in contours:
#             x, y, w, h = cv2.boundingRect(cnt)
#             boxes.append([x, y, x + w, y + h])
#         all_boxes.append(np.array(boxes))  # shape [N_i, 4]
#     return all_boxes  # list of length B

def normalize_boxes(boxes_list, H, W, target_size=1024):
    norm_boxes_list = []
    for boxes in boxes_list:
        if boxes is None or len(boxes) == 0:
            norm_boxes_list.append(np.zeros((0, 4)))  # 或者跳过处理
            continue
        boxes = np.array(boxes)
        boxes_1024 = boxes / np.array([W, H, W, H]) * target_size
        norm_boxes_list.append(boxes_1024)
    return norm_boxes_list


def save_medsam_masks(medsam_seg, epoch, save_dir="./vis_medsam_masks"):
    """
    保存 batch 的 medsam_seg 到本地 PNG 文件。

    参数：
    - medsam_seg: Tensor, shape [B, H, W]，推理出的伪标签 mask（通常为0/1）
    - epoch: 当前 epoch，用于保存文件夹命名
    - save_dir: 根目录路径
    """
    folder = os.path.join(save_dir, f'epoch_{epoch:04d}')
    os.makedirs(folder, exist_ok=True)

    masks_np = medsam_seg.detach().cpu().numpy()  # [B, H, W]

    for i, mask in enumerate(masks_np):
        mask_img = (mask * 255).astype(np.uint8)  # 转为灰度图（0/255）
        Image.fromarray(mask_img).save(os.path.join(folder, f'medsam_mask_{i:03d}.png'))

    print(f"✅ Saved MedSAM masks to: {folder}")


@torch.no_grad()
def batch_medsam_inference(medsam_model, image_embeddings, box_batch_list, H, W):
    """
    对一个 batch 的图像 embedding 和对应 box 列表，逐图推理并合成 segmentation mask。

    参数：
    - medsam_model: 已加载的 MedSAM 模型
    - image_embeddings: torch.Tensor，shape [B, C, 64, 64]
    - box_batch_list: list of np.ndarray，每个 [N_i, 4]，为每张图的 box（已经归一化为 1024）
    - H, W: 原始图像尺寸

    返回：
    - seg_masks: torch.Tensor，shape [B, H, W]，每张图的二值 mask
    """
    B = image_embeddings.size(0)
    seg_masks = []

    for i in range(B):
        boxes_1024 = box_batch_list[i]  # numpy array, shape [N_i, 4]
        # print(type(boxes_1024), type(boxes_1024[0]))
        boxes_1024 = np.array(boxes_1024)

        if boxes_1024.size == 0:
            # 没有框：返回全零 mask
            empty_mask = torch.zeros((1, H, W), device=image_embeddings.device)
            seg_masks.append(empty_mask)
            continue

        img_embed_i = image_embeddings[i].unsqueeze(0)  # [1, C, 64, 64]
        all_masks = []

        for box in boxes_1024:
            box_tensor = torch.tensor(box, dtype=torch.float32, device=img_embed_i.device).unsqueeze(0).unsqueeze(
                0)  # [1, 1, 4]

            sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
                points=None,
                boxes=box_tensor,
                masks=None,
            )

            low_res_logits, _ = medsam_model.mask_decoder(
                image_embeddings=img_embed_i,
                image_pe=medsam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            low_res_pred = torch.sigmoid(low_res_logits)  # [1, 1, 256, 256]
            upsampled_pred = F.interpolate(
                low_res_pred,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )  # [1, 1, H, W]

            mask_bin = (upsampled_pred > 0.5).float()  # [1, 1, H, W]
            mask_bin = mask_bin.squeeze(1)  # [1, H, W]，保持 batch dim
            all_masks.append(mask_bin)

        # 合并多个 mask（按位或）
        merged_mask = torch.stack(all_masks, dim=0).max(dim=0)[0]  # shape: [1, H, W]
        seg_masks.append(merged_mask)

    # 所有 seg_masks 应为 [1, H, W]，拼接为 [B, H, W]
    return torch.cat(seg_masks, dim=0)  # shape: [B, H, W]


def featuremap_visual(feature, i, out_dir, save_feature=True, show_feature=True, feature_title=None, num_ch=-1, nrow=8,
                      padding=10, pad_value=1):
    # feature = feature.detach().cpu()
    b, c, h, w = feature.shape
    for w in range(0, 8):
        feature_t = feature[w]
        # feature = feature.unsqueeze(1)

        if c > num_ch > 0:
            feature_t = feature_t[:num_ch]

        # img = torchvision.utils.make_grid(feature, nrow=nrow, padding=padding, pad_value=pad_value)
        feature_t = feature_t.squeeze(0)
        img = feature_t.detach().cpu().numpy()
        images = img
        cv2.imwrite(out_dir + '/{:s}_prob_inside_{:s}.png'.format(str(i), str(w)), images.astype(np.uint8))


def save_checkpoint(state, epoch, is_best, save_dir, cp_flag, is_second):
    cp_dir = '{:s}/checkpoints'.format(save_dir)
    if not os.path.exists(cp_dir):
        os.mkdir(cp_dir)
    filename = '{:s}/checkpoint.pth.tar'.format(cp_dir)
    torch.save(state, filename)
    if cp_flag:
        shutil.copyfile(filename, '{:s}/checkpoint_{:d}.pth.tar'.format(cp_dir, epoch + 1))
    if is_best:
        shutil.copyfile(filename, '{:s}/checkpoint_best.pth.tar'.format(cp_dir))
    # if is_top15:
    #     shutil.copyfile(filename, os.path.join(cp_dir, f'checkpoint_top15_epoch{epoch+1:03d}_iou{state["best_iou"]:.4f}.pth.tar'))
    if is_second:
        shutil.copyfile(filename, '{:s}/checkpoint_{:d}_demo.pth.tar'.format(cp_dir, epoch + 1))


def setup_logging(opt):
    mode = 'a' if opt.train['checkpoint'] else 'w'

    # create logger for training information
    logger = logging.getLogger('train_logger')
    logger.setLevel(logging.DEBUG)
    # create console handler and file handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler('{:s}/train.log'.format(opt.train['save_dir']), mode=mode)
    file_handler.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s\t%(message)s', datefmt='%Y-%m-%d %I:%M')
    # add formatter to handlers
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    # add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # create logger for epoch results
    logger_results = logging.getLogger('results')
    logger_results.setLevel(logging.DEBUG)
    file_handler2 = logging.FileHandler('{:s}/epoch_results.txt'.format(opt.train['save_dir']), mode=mode)
    file_handler2.setFormatter(logging.Formatter('%(message)s'))
    logger_results.addHandler(file_handler2)

    logger.info('***** Training starts *****')
    logger.info('save directory: {:s}'.format(opt.train['save_dir']))
    if mode == 'w':
        logger_results.info('epoch\ttrain_loss\ttrain_loss_CE\ttrain_loss_var\ttrain_acc\ttrain_iou\t'
                            'val_loss\tval_acc\tval_iou')

    return logger, logger_results


def update_ema_variables(model, model_teacher, global_step, base_alpha=0.99):
    # 建议用 global_step 而不是 epoch（更平滑）
    alpha = min(1.0 - 1.0 / float(global_step + 1), base_alpha)

    # 1) EMA 更新参数
    for p_t, p in zip(model_teacher.parameters(), model.parameters()):
        p_t.data.mul_(alpha).add_(p.data, alpha=1 - alpha)

    # 2) 同步 BN 等 buffers（running_mean/var 等）
    for b_t, b in zip(model_teacher.buffers(), model.buffers()):
        b_t.data.copy_(b.data)

    return model, model_teacher




def sigmoid_prob(logits):
    # logits: [B,1,H,W] or [B,H,W]
    if logits.dim() == 4:
        return torch.sigmoid(logits[:, 0])
    return torch.sigmoid(logits)

def dice_loss_binary_from_logits(logits, target, eps=1e-6):
    """
    logits: [B,1,H,W] or [B,H,W]
    target: [B,H,W] float {0,1}
    """
    p = sigmoid_prob(logits)  # [B,H,W]
    target = target.float()
    inter = (p * target).sum(dim=(1,2))
    denom = (p + target).sum(dim=(1,2))
    dice = (2 * inter + eps) / (denom + eps)
    return (1 - dice).mean()


def iou_binary(pred_bin, gt_bin, eps=1e-6):
    """
    pred_bin / gt_bin: numpy or torch [B,H,W] 0/1
    """
    if isinstance(pred_bin, np.ndarray):
        pred = torch.from_numpy(pred_bin).float()
    else:
        pred = pred_bin.float()
    if isinstance(gt_bin, np.ndarray):
        gt = torch.from_numpy(gt_bin).float()
    else:
        gt = gt_bin.float()

    inter = (pred * gt).sum(dim=(1,2))
    union = (pred + gt - pred*gt).sum(dim=(1,2))
    return ((inter + eps) / (union + eps)).mean().item()




def compute_unsupervised_loss(predict, target, ignore_mask):
    target[ignore_mask == 255] = 255
    loss = F.cross_entropy(predict, target, ignore_index=255)

    return loss


if __name__ == '__main__':
    main()