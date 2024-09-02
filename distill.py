import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from tqdm import tqdm
from utils import get_dataset, get_network, get_eval_pool, evaluate_synset, get_time, DiffAugment, ParamDiffAug, one_hot_CrossEntropy
import wandb
import copy
import random
from reparam_module import ReparamModule
# import cv2
import warnings
import math
import matplotlib.pyplot as plt
from io import BytesIO


warnings.filterwarnings("ignore", category=DeprecationWarning)
seed = 3407
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed) 
torch.backends.cudnn.deterministic = True
    
def main(args):
    if args.zca and args.texture:
        raise AssertionError("Cannot use zca and texture together")

    if args.texture and args.pix_init == "real":
        print("WARNING: Using texture with real initialization will take a very long time to smooth out the boundaries between images.")

    if args.max_experts is not None and args.max_files is not None:
        args.total_experts = args.max_experts * args.max_files

    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))

    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    eval_it_pool = np.arange(0, args.Iteration + 1, args.eval_it).tolist()
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)
    if args.model=='pretrained':
        model_eval_pool = get_eval_pool(args.eval_mode, 'ResNet18', 'ResNet18')
    else:
        model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    im_res = im_size[0]

    args.im_size = im_size

    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []
    print(accs_all_exps)
    data_save = []

    if args.dsa:
        # args.epoch_eval_train = 1000
        args.dc_aug_param = None

    args.dsa_param = ParamDiffAug()

    dsa_params = args.dsa_param
    if args.zca:
        zca_trans = args.zca_trans
    else:
        zca_trans = None

    wandb.init(sync_tensorboard=False,
               project="DatasetDistillation",
               job_type="CleanRepo",
               config=args,
               )

    args = type('', (), {})()

    for key in wandb.config._items:
        setattr(args, key, wandb.config._items[key])

    args.dsa_param = dsa_params
    args.zca_trans = zca_trans

    if args.batch_syn is None:
        if num_classes * args.ipc <256:
            args.batch_syn = num_classes * args.ipc
        else:
            args.batch_syn = 256

    args.distributed = torch.cuda.device_count() > 1
    print(args.distributed)

    print('Hyper-parameters: \n', args.__dict__)
    print('Evaluation model pool: ', model_eval_pool)

    ''' organize the real dataset '''
    images_all = []
    labels_all = []
    indices_class = [[] for c in range(num_classes)]
    print("BUILDING DATASET")
    for i in tqdm(range(len(dst_train))):
        sample = dst_train[i]
        images_all.append(torch.unsqueeze(sample[0], dim=0))
        labels_all.append(class_map[torch.tensor(sample[1]).item()])

    for i, lab in tqdm(enumerate(labels_all)):
        indices_class[lab].append(i)
    images_all = torch.cat(images_all, dim=0).to("cpu")
    labels_all = torch.tensor(labels_all, dtype=torch.long, device="cpu")

    # for c in range(num_classes):
    #     print('class c = %d: %d real images'%(c, len(indices_class[c])))

    for ch in range(channel):
        print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))


    def get_images(c, n):  # get random n images from class c
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        return images_all[idx_shuffle]
    
    ''' initialize the synthetic data '''
    if args.low_rank_control:
        label_syn = torch.tensor([np.ones(args.ipc*args.control_size)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9] len=num_classes*ipc*control_size
    else:
        label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1)
    if args.texture:
        image_syn = torch.randn(size=(num_classes * args.ipc, channel, im_size[0]*args.canvas_size, im_size[1]*args.canvas_size), dtype=torch.float)
    else:
        image_syn = torch.randn(size=(num_classes * args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float)

    syn_lr = torch.tensor(args.lr_teacher).to(args.device)

    if args.pix_init == 'real':
        print('initialize synthetic data from random real images')
        if args.texture:
            for c in range(num_classes):
                for i in range(args.canvas_size):
                    for j in range(args.canvas_size):
                        image_syn.data[c * args.ipc:(c + 1) * args.ipc, :, i * im_size[0]:(i + 1) * im_size[0],
                        j * im_size[1]:(j + 1) * im_size[1]] = torch.cat(
                            [get_images(c, 1).detach().data for s in range(args.ipc)])

        if args.mixup:
            image_syn_patch = []
        
            # _, C, H, W = image_syn_patch[0].shape
            # image_syn_data = torch.cat(image_syn_patch).reshape(num_classes, args.ipc * args.factor * args.factor, C, H, W)
            # n_crop = args.factor**2
            # label_syn = F.one_hot(label_syn, num_classes)
            # random_indices = np.random.permutation(label_syn.shape[0])
            # image_syn_data = image_syn_data[random_indices]
            # label_syn = label_syn[random_indices]
            for c in range(num_classes):
                image_syn_patch = get_images(c , args.ipc * args.factor * args.factor * 3).detach().data
                s = 32 // args.factor
                remained = 32 % args.factor
                n = args.ipc
                k = 0
                h_loc = 0
                for i in range(args.factor):
                    h_r = s + 1 if i < remained else s
                    w_loc = 0
                    for j in range(args.factor):
                        for l in range(3):
                            w_r = s + 1 if j < remained else s
                            if l==0:
                                image_syn_patch_part = F.interpolate(image_syn_patch[(k*3) * n  : (k*3+1) * n], size=(h_r, w_r))
                            else:
                                image_syn_patch_part[:,l,...] = F.interpolate(image_syn_patch[(k*3 + l) * n  : (k*3 + l + 1) * n], size=(h_r, w_r))[:,l,...]
                        k+=1
                        image_syn.data[n * c:n * (c + 1), :, h_loc:h_loc + h_r,
                                        w_loc:w_loc + w_r] = image_syn_patch_part
                        w_loc += w_r
                    h_loc += h_r
            # print(label_syn)
        else:
            for c in range(num_classes):
                image_syn.data[c * args.ipc:(c + 1) * args.ipc] = get_images(c, args.ipc).detach().data
    else:
        print('initialize synthetic data from random noise')


    ''' training '''
    image_syn = image_syn.detach().to(args.device).requires_grad_(True)
    print('img:',image_syn.shape)
    syn_lr = syn_lr.detach().to(args.device).requires_grad_(True)
    if args.soft_label:
        if args.temperature:
            temperature = torch.tensor(1,dtype=torch.float).to(args.device).requires_grad_(True)
        # soft_items = torch.from_numpy(np.array([np.ones(args.ipc)*i for i in range(num_classes)])).long().view(-1)
        # if args.low_rank_control:
        #     soft_items = torch.cat([soft_items for _ in range(args.control_size)])
        # soft_items = torch.nn.functional.one_hot(soft_items).float()
        # label_syn_soft = torch.tensor(soft_items).detach().to(args.device).requires_grad_(True)
        # label_syn_soft = torch.randn((args.control_size*args.ipc*args.ipc*5, num_classes)).detach().to(args.device).requires_grad_(True)
        if args.spilt:
            label_syn_soft = torch.randn((args.control_size*args.ipc*args.ipc*(int)((args.factor*(args.factor+1)*(2*args.factor+1))//6), num_classes)).detach().to(args.device).requires_grad_(True)
        else:
            label_syn_soft = torch.randn((args.control_size*args.ipc*(int)((args.factor*(args.factor+1)*(2*args.factor+1))//6), num_classes)).detach().to(args.device).requires_grad_(True)
            # label_syn_soft = torch.randn((args.ipc*(int)((args.factor*(args.factor+1)*(2*args.factor+1))//6), num_classes)).detach().to(args.device).requires_grad_(True)
            # label_syn_soft = label_syn_soft.repeat(args.control_size, 1).detach().to(args.device).requires_grad_(True)
        lr_soft = torch.tensor(args.lr_soft, dtype = torch.float)
        optimizer_label = torch.optim.SGD([label_syn_soft], lr=lr_soft, momentum=0.5)
        optimizer_label.zero_grad()
        print("label_syn_soft:", label_syn_soft.shape)
    if args.low_rank:
        image_syn_a = torch.randn(size=(num_classes * args.ipc, channel, im_size[0], args.rank_size), dtype=torch.float).detach().to(args.device).requires_grad_(True)
        image_syn_b = torch.randn(size=(num_classes * args.ipc, channel, args.rank_size, im_size[1]), dtype=torch.float).detach().to(args.device).requires_grad_(True)
        optimizer_img = torch.optim.SGD([image_syn_a, image_syn_b], lr=args.lr_img, momentum=0.5)
    
    elif args.low_rank_control:
        if args.spilt:
            image_syn_a = torch.randn(size=(args.ipc, channel, im_size[0]//args.downsample, args.rank_size), dtype=torch.float).detach().to(args.device).requires_grad_(True)  ### rank_size代表基向量的维度r,channel代表论文中的k，control_size代表论文中的m 
            image_syn_b = torch.randn(size=(args.ipc, channel, args.rank_size, im_size[1]//args.downsample), dtype=torch.float).detach().to(args.device).requires_grad_(True)
            image_control = torch.randn(size=(args.control_size, args.ipc * args.ipc, channel, args.rank_size, args.rank_size), dtype=torch.float).detach().to(args.device).requires_grad_(True) ### 这是基向量
            print(image_syn_a.shape, image_syn_b.shape)
        else:
            if not args.soft_label:
                image_syn_a = torch.randn(size=(num_classes, args.ipc, channel, im_size[0]//args.downsample, args.rank_size), dtype=torch.float).detach().to(args.device).requires_grad_(True)
                image_syn_b = torch.randn(size=(num_classes, args.ipc, channel, args.rank_size, im_size[1]//args.downsample), dtype=torch.float).detach().to(args.device).requires_grad_(True)
                image_control = torch.randn(size=(num_classes, args.control_size*args.ipc, 3, args.rank_size, args.rank_size), dtype=torch.float).detach().to(args.device).requires_grad_(True)
            else:
                image_syn_a = torch.randn(size=(args.ipc, channel, im_size[0]//args.downsample, args.rank_size), dtype=torch.float).detach().to(args.device).requires_grad_(True)
                image_syn_b = torch.randn(size=(args.ipc, channel, args.rank_size, im_size[1]//args.downsample), dtype=torch.float).detach().to(args.device).requires_grad_(True)
                image_control = torch.randn(size=(args.ipc, args.control_size, 3, args.rank_size, args.rank_size), dtype=torch.float).detach().to(args.device).requires_grad_(True)
                # image_control = torch.randn(size=(args.control_size, 1, args.rank_size, args.rank_size), dtype=torch.float).detach().to(args.device).requires_grad_(True)
                # image_control = torch.randn(size=(args.control_size, channel, args.rank_size, args.rank_size), dtype=torch.float).detach().to(args.device).requires_grad_(True)
        optimizer_img = torch.optim.SGD([image_syn_a, image_syn_b], lr=args.lr_img, momentum=0.5)
        optimizer_control = torch.optim.SGD([image_control], lr=args.lr_control, momentum=0.5)
    
    elif args.dictionary:
        label_syn_base = torch.randn((args.ipc*num_classes, num_classes)).detach().to(args.device).requires_grad_(True)
        image_syn_query = torch.randn(size=(num_classes, args.rank_size), dtype=torch.float).detach().to(args.device).requires_grad_(True)
        image_syn_memory = torch.randn(size=(args.rank_size, 3*im_size[0]*im_size[1]//(args.downsample*args.downsample)), dtype=torch.float).detach().to(args.device).requires_grad_(True)
        optimizer_img = torch.optim.SGD([label_syn_base, image_syn_query, image_syn_memory], lr=args.lr_img, momentum=0.5)
    else:
        if args.soft_label:
            label_syn_soft = torch.randn((args.ipc*num_classes, num_classes)).detach().to(args.device).requires_grad_(True)
            lr_soft = torch.tensor(args.lr_soft, dtype = torch.float)
            optimizer_label = torch.optim.SGD([label_syn_soft], lr=lr_soft, momentum=0.5)
            optimizer_label.zero_grad()
        optimizer_img = torch.optim.SGD([image_syn], lr=args.lr_img, momentum=0.5)
    
    optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.5)
    optimizer_lr.zero_grad()
    if args.temperature:
        optimizer_temp = torch.optim.SGD([temperature], lr=args.lr_temp, momentum=0.5)
        optimizer_temp.zero_grad()
    optimizer_img.zero_grad()
    if args.low_rank_control:
        optimizer_control.zero_grad()
    B, C, H ,W = image_syn.shape
    if args.color:
        color = []
        color_param_list = []
        for i in range(args.ipc*num_classes*3*args.control_size):
            color_item = nn.Sequential(
                nn.Conv2d(1,3,1).to(args.device)
            ).to(args.device)
            color.append(color_item)
            for name, param in color_item.named_parameters():
                color_param_list.append(param)
        optimizer_color = torch.optim.SGD(color_param_list, lr=1000)
        optimizer_color.zero_grad()
    if args.weight:
        weight = torch.zeros(B,args.factor,args.factor).detach().to(args.device).requires_grad_(True)
        optimizer_weight = torch.optim.SGD([weight], lr=args.lr_weight, momentum=0.5)
        optimizer_weight.zero_grad()
    if args.mlp:
        mlp = torch.nn.Linear(32*32, 32*32).to(args.device)
        optimizer_mlp = torch.optim.SGD(mlp.parameters(), lr=args.lr_mlp, momentum=0.5)
        optimizer_mlp.zero_grad()

    criterion = nn.CrossEntropyLoss().to(args.device)
    print('%s training begins'%get_time())

    expert_dir = os.path.join(args.buffer_path, args.dataset)
    if args.dataset == "ImageNet":
        expert_dir = os.path.join(expert_dir, args.subset, str(args.res))
    if args.dataset in ["CIFAR10", "CIFAR100"] and not args.zca:
        expert_dir += "_NO_ZCA"
    expert_dir = os.path.join(expert_dir, args.model)
    print("Expert Dir: {}".format(expert_dir))

    if args.load_all:
        buffer = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            buffer = buffer + torch.load(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))

    else:
        expert_files = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            expert_files.append(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))
        file_idx = 0
        expert_idx = 0
        random.shuffle(expert_files)
        if args.max_files is not None:
            expert_files = expert_files[:args.max_files]
        print("loading file {}".format(expert_files[file_idx]))
        buffer = torch.load(expert_files[file_idx])
        if args.max_experts is not None:
            buffer = buffer[:args.max_experts]
        random.shuffle(buffer)

    best_acc = {m: 0 for m in model_eval_pool}

    best_std = {m: 0 for m in model_eval_pool}

    for it in range(0, args.Iteration+1):
        save_this_it = False

        # writer.add_scalar('Progress', it, it)
        wandb.log({"Progress": it}, step=it)
        ''' Evaluate synthetic data '''
        if it in eval_it_pool and it!=0:
            for model_eval in model_eval_pool:
                print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))
                if args.dsa:
                    print('DSA augmentation strategy: \n', args.dsa_strategy)
                    print('DSA augmentation parameters: \n', args.dsa_param.__dict__)
                else:
                    print('DC augmentation parameters: \n', args.dc_aug_param)

                accs_test = []
                accs_train = []
                for it_eval in range(args.num_eval):
                    net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model

                    eval_labs = label_syn
                    with torch.no_grad():
                        if args.low_rank:
                            image_save = torch.matmul(image_syn_a, image_syn_b)
                        elif args.low_rank_control:
                            image_save = get_image(image_syn_a, image_syn_b, image_control, args.control_size, args.spilt,  args.soft_label)
                        elif args.dictionary:
                            image_save = get_image_dict(image_syn_query, image_syn_memory, label_syn_base, args.downsample)
                        else:
                            image_save = image_syn
                    # color_clip = torch.clip(color, 0, 255/torch.max(image_save).detach()) # avoid any unaware modification
                    if args.color:
                        image_syn_eval, label_syn_eval, color_eval = copy.deepcopy(image_save.detach()), copy.deepcopy(eval_labs.detach()), copy.deepcopy(color) 
                        image_syn_eval, label_syn_eval = image_syn_eval, torch.cat([label_syn_eval for _ in range(args.control_size)])
                        image_syn_eval, label_syn_eval  = decode_zoom_multi(image_syn_eval, label_syn_eval, args.factor, color_eval, 'test')
                        print(image_syn_eval.shape, label_syn_eval.shape)
                    elif args.low_rank_control:
                        image_syn_eval, label_syn_eval= copy.deepcopy(image_save.detach()), copy.deepcopy(eval_labs.detach())
                        # image_syn_eval, label_syn_eval = image_syn_eval, torch.cat([label_syn_eval for _ in range(args.control_size*args.ipc)])
                        image_syn_eval, label_syn_eval  = decode_zoom_multi(image_syn_eval, label_syn_eval, args.factor, None, 'test')
                        print(image_syn_eval.shape, label_syn_eval.shape)
                    elif args.dictionary:
                        image_syn_eval, label_syn_eval= copy.deepcopy(image_save.detach()), copy.deepcopy(eval_labs.detach())
                        image_syn_eval, label_syn_eval  = decode_zoom_multi(image_syn_eval, label_syn_eval, args.factor, None, 'test')
                        print(image_syn_eval.shape, label_syn_eval.shape)
                    else:
                        image_syn_eval, label_syn_eval= copy.deepcopy(image_save.detach()), copy.deepcopy(eval_labs.detach())
                        image_syn_eval, label_syn_eval = image_syn_eval, label_syn_eval
                    # print(label_syn_eval)
                    # print(label_syn_eval.dtype)

                    args.lr_net = syn_lr.item()
                    if args.soft_label:
                        if args.temperature:
                            _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, copy.deepcopy(label_syn_soft.detach()), testloader, args, texture=args.texture, temperature=temperature)
                        else:
                            _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, copy.deepcopy(label_syn_soft.detach()), testloader, args, texture=args.texture)
                    else:
                        _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, label_syn_eval , testloader, args, texture=args.texture)
                    accs_test.append(acc_test)
                    accs_train.append(acc_train)
                accs_test = np.array(accs_test)
                accs_train = np.array(accs_train)
                acc_test_mean = np.mean(accs_test)
                accc_best = np.max(accs_test)
                acc_test_std = np.std(accs_test)
                if accc_best > best_acc[model_eval]:
                    best_acc[model_eval] = accc_best
                    best_std[model_eval] = acc_test_std
                    save_this_it = True
                print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs_test), model_eval, acc_test_mean, acc_test_std))
                wandb.log({'Accuracy/{}'.format(model_eval): acc_test_mean}, step=it)
                wandb.log({'Max_Accuracy/{}'.format(model_eval): best_acc[model_eval]}, step=it)
                wandb.log({'Std/{}'.format(model_eval): acc_test_std}, step=it)
                wandb.log({'Max_Std/{}'.format(model_eval): best_std[model_eval]}, step=it)


        if it in eval_it_pool and (save_this_it or it % 1000 == 0):
            with torch.no_grad():
                if args.low_rank:
                    image_save = torch.matmul(image_syn_a, image_syn_b).cuda()
                elif args.low_rank_control:
                    image_save = get_image(image_syn_a, image_syn_b, image_control, args.control_size, args.spilt, args.soft_label)
                    image_save = image_save.cuda()
                elif args.dictionary:
                    image_save = get_image_dict(image_syn_query, image_syn_memory, label_syn_base, args.downsample)
                    image_save = image_save.cuda()
                else:
                    image_save = image_syn.cuda()

                save_dir = os.path.join(".", "logged_files", args.dataset, wandb.run.name)

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                torch.save(image_save.cpu(), os.path.join(save_dir, "images_{}.pt".format(it)))
                torch.save(label_syn.cpu(), os.path.join(save_dir, "labels_{}.pt".format(it)))

                if save_this_it:
                    torch.save(image_save.cpu(), os.path.join(save_dir, "images_best.pt".format(it)))
                    torch.save(label_syn.cpu(), os.path.join(save_dir, "labels_best.pt".format(it)))
                if args.low_rank:
                    wandb.log({"Pixels": wandb.Histogram(torch.nan_to_num(torch.matmul(image_syn_a, image_syn_b).detach().cpu()))}, step=it)
                else:
                    wandb.log({"Pixels": wandb.Histogram(torch.nan_to_num(image_syn.detach().cpu()))}, step=it)

                if image_save.shape[0]<500 or args.force_save:
                    upsampled = image_save
                    if args.dataset != "ImageNet":
                        upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                        upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                    grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                    # heatmap = torch.nn.functional.normalize(weight.reshape(B, C, -1), dim=-1).reshape(B, C, H, W).permute(0,2,3,1).detach().cpu().numpy()
                    wandb.log({"Synthetic_Images": wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)
                    wandb.log({'Synthetic_Pixels': wandb.Histogram(torch.nan_to_num(image_save.detach().cpu()))}, step=it)

                    for clip_val in [2.5]:
                        std = torch.std(image_save)
                        mean = torch.mean(image_save)
                        upsampled = torch.clip(image_save, min=mean-clip_val*std, max=mean+clip_val*std)
                        if args.dataset != "ImageNet":
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                        grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                        wandb.log({"Clipped_Synthetic_Images/std_{}".format(clip_val): wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)
                        # if args.low_rank_control:
                        #     std_control = torch.std(image_control)
                        #     mean_control = torch.mean(image_control)
                        #     upsampled_control = torch.clip(image_control.reshape(-1,3,8,8), min=mean_control-clip_val*std_control, max=mean_control+clip_val*std_control)
                        #     grid_control = torchvision.utils.make_grid(upsampled_control, nrow=10, normalize=True, scale_each=True)
                        #     wandb.log({"Clipped_Synthetic_Control/std_{}".format(clip_val): wandb.Image(torch.nan_to_num(grid_control.detach().cpu()))}, step=it)
                        
                        #     image_base = torch.matmul(image_syn_a, image_syn_b).detach()
                        #     std_base = torch.std(image_base)
                        #     mean_base = torch.mean(image_base)
                        #     upsampled_base = torch.clip(image_base, min=mean_base-clip_val*std_base, max=mean_base+clip_val*std_base)
                        #     grid_base = torchvision.utils.make_grid(upsampled_base.reshape(-1,3,32,32), nrow=10, normalize=True, scale_each=True)
                        #     wandb.log({"Clipped_Synthetic_base/std_{}".format(clip_val): wandb.Image(torch.nan_to_num(grid_base.detach().cpu()))}, step=it)
                    if args.zca:
                        image_save = image_save.to(args.device)
                        image_save = args.zca_trans.inverse_transform(image_save)
                        image_save.cpu()

                        torch.save(image_save.cpu(), os.path.join(save_dir, "images_zca_{}.pt".format(it)))

                        upsampled = image_save
                        if args.dataset != "ImageNet":
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                        grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                        wandb.log({"Reconstructed_Images": wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)
                        wandb.log({'Reconstructed_Pixels': wandb.Histogram(torch.nan_to_num(image_save.detach().cpu()))}, step=it)

                        for clip_val in [2.5]:
                            std = torch.std(image_save)
                            mean = torch.mean(image_save)
                            upsampled = torch.clip(image_save, min=mean - clip_val * std, max=mean + clip_val * std)
                            if args.dataset != "ImageNet":
                                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                            grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                            wandb.log({"Clipped_Reconstructed_Images/std_{}".format(clip_val): wandb.Image(
                                torch.nan_to_num(grid.detach().cpu()))}, step=it)
                        # if args.color:
                        #     clip_val = 2.5
                        #     std = torch.std(image_save)
                        #     mean = torch.mean(image_save)
                        #     upsampled = torch.clip(image_save, min=mean-clip_val*std, max=mean+clip_val*std)
                        #     color_channel = []
                        #     B = 10
                        #     ipc = upsampled.shape[0]//B
                        #     for k in range(3):
                        #         img_col = upsampled[:, k:k+1, ...].reshape(ipc,B,1,32,32).detach()
                        #         for i in range(ipc):
                        #             for j in range(B):
                        #                 color_channel.append(color[k*ipc*B + i*B + j](img_col[i:i+1,j,...]))
                        #     color_channel_grid = torchvision.utils.make_grid(torch.cat(color_channel).detach(), nrow=10, normalize=True, scale_each=True)
                        #     wandb.log({"color_channel_grid": wandb.Image(torch.nan_to_num(color_channel_grid.detach().cpu()))}, step=it)
        # if args.color:
        #     wandb.log({"Color_max": torch.max(color.detach().cpu())}, step=it)
        #     wandb.log({"Color_mean": torch.mean(color.detach().cpu())}, step=it)
        #     wandb.log({"Color_min": torch.min(color.detach().cpu())}, step=it)
        wandb.log({"Synthetic_LR": syn_lr.detach().cpu()}, step=it)
        if args.temperature:
            wandb.log({"Temperature": temperature.detach().cpu()}, step=it)


        ### khy 注释
        # if args.soft_label and it%100==0 and args.factor==1:
        #     # num = [0 for _ in range(num_classes)]
        #     array = np.argmax(label_syn_soft.detach().cpu().data.numpy(), axis=-1)
        #     # for i in range(label_syn_soft.shape[0]):
        #     #     num[array[i]] += 1
        #     # for i in range(num_classes):
        #     #     wandb.log({f'Nums_class_{i}': num[i]}, step=it)

        #     label_count = array.reshape(args.control_size,args.ipc)
        #     equal = 0
        #     for i in range(args.control_size):
        #         if label_count[i,0]==label_count[i,1]:
        #             equal += 1
        #     wandb.log({f'equal_class': equal}, step=it)



            # for i in range(args.ipc):
            #     label_count_i = label_count[:,:,i].reshape(-1)
            #     values = [0 for _ in range(num_classes)]
            #     for j in range(label_count_i.shape[0]):
            #         values[label_count_i[j]] += 1
            #     categories = ["Class1","Class2","Class3","Class4","Class5","Class6","Class7","Class8","Class9","Class10"]
            #     colors = ['red','green','blue','cyan','magenta','yellow','black','purple','gray','orange']
            #     bars = plt.bar(categories, values, color=colors)
            #     for bar in bars:
            #         yval = bar.get_height()
            #         plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom')
            #     plt.title("My Bar Chart")
            #     plt.xlabel("Categories")
            #     plt.ylabel("Values")
            #     plt.savefig("bar_chart.png")
            #     plt.close()
            #     wandb.log({f'num_bar_chart{i}': wandb.Image("bar_chart.png")}, step=it)


        # weight_norm = torch.nn.functional.normalize(weight.reshape(B, C, -1), dim=-1).reshape(B, C, H, W)
        student_net = get_network(args.model, channel, num_classes, im_size, dist=False).to(args.device)  # get a random model

        student_net = ReparamModule(student_net)

        if args.distributed:
            student_net = torch.nn.DataParallel(student_net)

        student_net.train()

        num_params = sum([np.prod(p.size()) for p in (student_net.parameters())])

        if args.load_all:
            expert_trajectory = buffer[np.random.randint(0, len(buffer))]
        else:
            expert_trajectory = buffer[expert_idx]
            expert_idx += 1
            if expert_idx == len(buffer):
                expert_idx = 0
                file_idx += 1
                if file_idx == len(expert_files):
                    file_idx = 0
                    random.shuffle(expert_files)
                print("loading file {}".format(expert_files[file_idx]))
                if args.max_files != 1:
                    del buffer
                    buffer = torch.load(expert_files[file_idx])
                if args.max_experts is not None:
                    buffer = buffer[:args.max_experts]
                random.shuffle(buffer)
                
        if args.max_start_epoch>3 and it<2000:
            start_epoch = int(3 + (args.max_start_epoch-3)*np.log10(9/2000*it + 1))
        else:
            start_epoch = args.max_start_epoch
        if args.dataset=="MSTAR" and it<100:
            start_epoch = 1
        start_epoch = np.random.randint(0, start_epoch)

        starting_params = expert_trajectory[start_epoch]

        target_params = expert_trajectory[start_epoch+args.expert_epochs]
        target_params = torch.cat([p.data.to(args.device).reshape(-1) for p in target_params], 0)
        student_params = [torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0).requires_grad_(True)]

        starting_params = torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0)


        if args.weight:
            weight_norm = args.factor * args.factor * torch.nn.functional.softmax(weight.reshape(B, -1), dim=-1).reshape(B, 1, args.factor, args.factor).expand((B,C,args.factor,args.factor))
            if args.patch:
                # avg_pool = torch.nn.AdaptiveAvgPool2d((H//args.patch_size,W//args.patch_size))
                resize  = nn.Upsample(size=(32,32), mode='bilinear')
                # upsample = torch.nn.UpsamplingNearest2d(scale_factor=args.patch_size)
                weight_norm_patch = resize(weight_norm)
                image_extra = (image_syn*(1-weight_norm_patch)).detach()
                syn_images = image_syn*(weight_norm_patch) + image_extra
                syn_images, label_syn_multi = decode_zoom_multi(syn_images, label_syn, args.factor)
            else:
                image_extra = (image_syn*(1-weight_norm)).detach()
                syn_images = image_syn*(weight_norm) + image_extra
        elif args.mlp:
            B, C, H ,W = image_syn.shape
            image_mlp = mlp(image_syn.reshape(B,C,-1)).reshape(B,C,H,W)
            image_extra = (image_syn - image_mlp).detach()
            syn_images = image_mlp + image_extra
        else:
            # syn_images = image_syn
            if args.color:
                # color_norm = 32*32*torch.nn.functional.softmax(color.reshape(color.shape[0],color.shape[1],-1), dim=-1).reshape(color.shape)
                # color_clip = torch.clip(color, 0, 255/torch.max(image_syn).detach())
                if args.low_rank_control:
                    syn_images = get_image(image_syn_a, image_syn_b, image_control, args.control_size, args.spilt,  args.soft_label)
                    syn_images, label_syn_mutil  = decode_zoom_multi(syn_images, label_syn, args.factor, color, 'train')
                else:
                    # print('image_syn:',image_syn.shape)
                    syn_images, label_syn_mutil  = decode_zoom_multi(image_syn, label_syn, args.factor, color, 'train')
            elif args.low_rank:
                syn_images, label_syn_mutil = torch.matmul(image_syn_a, image_syn_b), label_syn
            elif args.low_rank_control:
                syn_images = get_image(image_syn_a, image_syn_b, image_control, args.control_size, args.spilt, args.soft_label)
                syn_images, label_syn_mutil  = decode_zoom_multi(syn_images, label_syn, args.factor, None, 'train')
                # print(syn_images.shape, label_syn_soft.shape)
            elif args.dictionary:
                syn_images = get_image_dict(image_syn_query, image_syn_memory, label_syn_base, args.downsample)
                syn_images, label_syn_mutil  = decode_zoom_multi(syn_images, label_syn, args.factor, None, 'train')
            else:
                syn_images, label_syn_mutil  = image_syn, label_syn
        


        y_hat = label_syn_mutil.to(args.device)
        if args.soft_label:
            y_soft = label_syn_soft.to(args.device)

        param_loss_list = []
        param_dist_list = []
        indices_chunks = []

        for step in range(args.syn_steps):

            if not indices_chunks:
                indices = torch.randperm(len(syn_images))
                indices_chunks = list(torch.split(indices, args.batch_syn))
            these_indices = indices_chunks.pop()


            x = syn_images[these_indices]
            
            # print(these_indices)
            # print(y_hat.shape)
            if args.soft_label:
                this_y_soft = y_soft[these_indices]
            else:
                this_y = y_hat[these_indices]
            if args.texture:
                x = torch.cat([torch.stack([torch.roll(im, (torch.randint(im_size[0]*args.canvas_size, (1,)), torch.randint(im_size[1]*args.canvas_size, (1,))), (1,2))[:,:im_size[0],:im_size[1]] for im in x]) for _ in range(args.canvas_samples)])
                this_y = torch.cat([this_y for _ in range(args.canvas_samples)])

            if args.dsa and (not args.no_aug):
                x = DiffAugment(x, args.dsa_strategy, param=args.dsa_param)

            if args.distributed:
                forward_params = student_params[-1].unsqueeze(0).expand(torch.cuda.device_count(), -1)
            else:
                forward_params = student_params[-1]
            x = student_net(x, flat_param=forward_params)
            
            # p = torch.softmax(p_logits, dim=-1)
            # q = torch.softmax(q_logits, dim=-1)
            # log_p = torch.log(p + 1e-10)
            # kl_div = torch.sum(p * (log_p - torch.log(q + 1e-10)), dim=-1)

            if args.soft_label:
                if args.temperature:
                    T = temperature
                else:
                    T = 1
                p = torch.softmax(x/T, dim=-1)
                log_p = torch.log(p + 1e-10)
                q = torch.softmax(this_y_soft/T, dim=-1)
                kl_loss = torch.sum(p * (log_p - torch.log(q + 1e-10)), dim=-1).mean()
                this_y = torch.argmax(this_y_soft.detach(), axis=-1)
                # kl_loss = F.kl_div(log_probs, this_y_soft, reduction='batchmean')
                loss = kl_loss + 0.2*criterion(x, this_y)
                # loss = kl_loss
                # if args.low_rank_control:
                #     diff_matrix = these_indices.view(-1, 1) - these_indices.view(1, -1)
                #     mask = (diff_matrix % args.ipc == 0).to(torch.float32).cuda()
                #     cl = contrastive_loss(x, mask)
            else:
                loss = criterion(x, this_y)
            grad = torch.autograd.grad(loss, student_params[-1], create_graph=True)[0]

            # student_params.append(student_params[-1] - grad * grad_weight) 
            student_params.append(student_params[-1] - syn_lr * grad) 



        param_loss = torch.tensor(0.0).to(args.device)
        param_dist = torch.tensor(0.0).to(args.device)
    

        if args.classifier:
            param_loss += torch.nn.functional.mse_loss(student_params[-1][-20490:], target_params[-20490:], reduction="sum")
            param_dist += torch.nn.functional.mse_loss(starting_params[-20490:], target_params[-20490:], reduction="sum")
        else:
            # total_grad = student_params[-1] - starting_params
            # target_grad = target_params - starting_params
            # _, idx = torch.topk(target_grad.abs(), 1000)
            # selected_total = total_grad[idx]
            # selected_target = target_grad[idx]
            param_loss += torch.nn.functional.mse_loss(student_params[-1], target_params, reduction="sum")
            param_dist += torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")

        param_loss_list.append(param_loss)
        param_dist_list.append(param_dist)


        param_loss /= num_params
        param_dist /= num_params
        param_loss /= param_dist
        grand_loss = param_loss
        # print(selected_total, selected_target)
        # grand_loss = F.cosine_similarity(selected_total.unsqueeze(0), selected_target.unsqueeze(0)).pow(2)
        # if args.low_rank_control:
        #     index = torch.arange(label_syn_soft.shape[0])
        #     diff_matrix = index.view(-1, 1) - index.view(1, -1)
        #     mask = (diff_matrix % args.ipc == 0).to(torch.float32).cuda()
        #     cl = contrastive_loss(label_syn_soft, mask)
        
        optimizer_img.zero_grad()
        optimizer_lr.zero_grad()
        if args.temperature:
            optimizer_temp.zero_grad()
        if args.color:
            optimizer_color.zero_grad()
        if args.low_rank_control:
            optimizer_control.zero_grad()
        if args.soft_label:
            optimizer_label.zero_grad()
        
        # if grand_loss>1 and it>100:
        #     continue
        # else:
        grand_loss.backward()
        if args.low_rank_control:
            optimizer_img.step()
            optimizer_control.step()
        else:
            optimizer_img.step()

        # if args.soft_label and lr_soft>0.1*args.lr_soft:
        #     lr_soft = args.lr_soft - args.lr_soft*(it/1000)
        #     optimizer_label.step()
        if args.soft_label:
            optimizer_label.step()
        if args.temperature:
            optimizer_temp.step()
        # optimizer_lr.step()
        if args.color:
            optimizer_color.step()
            
        wandb.log({"Grand_Loss": (grand_loss).detach().cpu(),
                    # "Contrastive_loss": cl.detach().cpu(),
                   "Start_Epoch": start_epoch})

        for _ in student_params:
            del _

        if it%10 == 0:
            print('%s iter = %04d, loss = %.4f' % (get_time(), it, grand_loss.item()))

    wandb.finish()

def decode_zoom(img, target, factor, color, mode):
    """Uniform multi-formation
    """
    # target = target.reshape(factor**2, img.shape[0]//factor**2, -1)
    h = img.shape[-1]
    # print('img shape:',img.shape)
    remained = h % factor

    # if remained > 0:
    #     img = F.pad(img, pad=(0, factor - remained, 0, factor - remained), value=0.5)
    s_crop = math.ceil(h / factor)
    n_crop = factor**2
    resize = nn.Upsample(size=(img.shape[-1],img.shape[-1]), mode='bilinear')
    # print(img.shape[-1])
    cropped = []
    for i in range(factor):
        for j in range(factor):
            h_loc = i * s_crop
            w_loc = j * s_crop
            # if mode !="test":
            cropped.append(resize(img[:, :, h_loc:h_loc + s_crop, w_loc:w_loc + s_crop]))
            # for k in range(img.shape[1]):
                # if args.color:
                #     B = 10
                #     img_col = resize(img[:, k:k+1, h_loc:h_loc + s_crop, w_loc:w_loc + s_crop]).reshape(B,img.shape[0]//B,1,32,32)
                #     img_col_expand = []
                #     ipc = img.shape[0]//B
                #     for m in range(B):
                #         for l in range(ipc):
                #             img_col_expand.append(color[k*B*ipc + l*B + m](img_col[m,l:l+1,...]))
                #     # print(torch.cat(img_col_expand).shape)
                #     cropped.append(torch.cat(img_col_expand))
                #     # img_col = color(resize(img[:, k:k+1, h_loc:h_loc + s_crop, w_loc:w_loc + s_crop]))
                #     # cropped.append(img_col.reshape(-1,3,32,32))
                # else:
                # cropped.append(resize(img[:, k:k+1, h_loc:h_loc + s_crop, w_loc:w_loc + s_crop]).expand(img.shape[0], img.shape[1], 32, 32))
    cropped = torch.cat(cropped)
    data_dec = cropped
    # target_dec = torch.cat([target for _ in range(n_crop*(4))])
    target_dec = torch.cat([target for _ in range(n_crop)])

    return data_dec, target_dec

def decode_zoom_multi(img, target, factor_max, color, mode):
    """Multi-scale multi-formation
    """
    data_multi = []
    target_multi = []
    range_start = 1
    # print(img.shape)
    for factor in range(1, factor_max + 1):
        decoded = decode_zoom(img, target, factor, color, mode)
        # print(decoded[0].shape)
        data_multi.append(decoded[0])
        target_multi.append(decoded[1])
        # print(decoded[0].shape,decoded[1].shape)
    # data_multi.append(color)
    # target_multi.append(target)
    return torch.cat(data_multi), torch.cat(target_multi)

def get_image(image_syn_a, image_syn_b, image_control, control_size, spilt, soft):
    resize  = nn.Upsample(size=(32,32), mode='bilinear')
    data_multi = []
    if spilt:
        for k in range(control_size):
            for i in range(image_syn_a.shape[0]):
                for j in range(image_syn_b.shape[0]):
                    data_multi.append(torch.matmul(torch.matmul(image_syn_a[i:i+1,...], image_control[k,i*image_syn_b.shape[0]+j,...]), image_syn_b[j:j+1,...]))
    else:
        if not soft:
            Numclass, ipc, C, H, W = image_syn_a.shape
            for i in range(Numclass):
                for j in range(ipc):
                    for k in range(control_size):
                        data_multi.append(torch.matmul(torch.matmul(image_syn_a[i,j:j+1,...], image_control[i,j*control_size+k,...]), image_syn_b[i,j:j+1,...]))
        else:
            for i in range(control_size):
                data_multi.append(torch.matmul(torch.matmul(image_syn_a, image_control[:,i,...]), image_syn_b))
        # for i in range(control_size):
        #     for j in range(image_syn_a.shape[0]):
        #         data = torch.matmul(torch.matmul(image_syn_a[j:j+1,...], image_control[i,...]), image_syn_b[j:j+1,...])
        #         data = resize(data)
        #         data_multi.append(data)
    # for i in range(image_syn_a.shape[0]):
    #     for j in range(image_syn_b.shape[0]):
    #         data_multi.append(torch.matmul(image_syn_a[i:i+1,...],image_syn_b[j:j+1,...]))   
    return torch.cat(data_multi)

def get_image_dict(image_syn_query, image_syn_memory, label_syn_soft, downsample):
    resize  = nn.Upsample(size=(32,32), mode='bilinear')
    data = torch.matmul(torch.matmul(label_syn_soft, image_syn_query), image_syn_memory)
    data = data.reshape(data.shape[0],3,32//downsample,32//downsample)
    return resize(data)

import torch
import torch.nn.functional as F

def contrastive_loss(z, mask, temperature=0.5):
    batch_size = z.size(0)
    z = F.normalize(z, p=2, dim=1)

    similarity_matrix = torch.matmul(z, z.T)
    
    # mask = torch.eye(batch_size).to(z.device)
    positive_samples = similarity_matrix * mask
    negative_samples = similarity_matrix * (1 - mask)
    
    # 计算损失
    numerator = torch.exp(positive_samples / temperature)
    denominator = torch.exp(negative_samples / temperature).sum(dim=1)
    loss = -torch.log(numerator / denominator)
    
    return loss.mean()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')

    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')

    parser.add_argument('--subset', type=str, default='imagenette', help='ImageNet subset. This only does anything when --dataset=ImageNet')

    parser.add_argument('--model', type=str, default='ConvNet', help='model')

    parser.add_argument('--ipc', type=int, default=10, help='image(s) per class')

    parser.add_argument('--eval_mode', type=str, default='S',
                        help='eval_mode, check utils.py for more info')

    parser.add_argument('--num_eval', type=int, default=3, help='how many networks to evaluate on')

    parser.add_argument('--eval_it', type=int, default=100, help='how often to evaluate')

    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=10000, help='how many distillation steps to perform')

    parser.add_argument('--lr_img', type=float, default=100000, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_lr', type=float, default=0.0001, help='learning rate for updating... learning rate')
    parser.add_argument('--lr_weight', type=float, default=10, help='learning rate for updating weight')
    parser.add_argument('--lr_score', type=float, default=100, help='learning rate for updating weight')
    parser.add_argument('--lr_soft', type=float, default=100, help='learning rate for updating weight')
    parser.add_argument('--lr_control', type=float, default=100, help='learning rate for updating weight')
    parser.add_argument('--lr_temp', type=float, default=1e-02, help='learning rate for updating weight')
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='initialization for synthetic learning rate')

    parser.add_argument('--lr_init', type=float, default=0.01, help='how to init lr (alpha)')

    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_syn', type=int, default=64, help='should only use this if you run out of VRAM')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')

    parser.add_argument('--pix_init', type=str, default='real', choices=["noise", "real"],
                        help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')

    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')

    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')

    parser.add_argument('--data_path', type=str, default='{path_to_dataset}', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='{path_to_buffer_storage}', help='buffer path')

    parser.add_argument('--expert_epochs', type=int, default=2, help='how many expert epochs the target params are')
    parser.add_argument('--syn_steps', type=int, default=30, help='how many steps to take on synthetic data')
    parser.add_argument('--max_start_epoch', type=int, default=18, help='max epoch we can start at')

    parser.add_argument('--zca', action='store_true', help="do ZCA whitening")
    parser.add_argument('--weight', action='store_true', help="do weight")
    parser.add_argument('--mlp', action='store_true', help="do mlp")
    parser.add_argument('--lr_mlp', default=100, help="do mlp")
    parser.add_argument('--load_all', action='store_true', help="only use if you can fit all expert trajectories into RAM")

    parser.add_argument('--no_aug', type=bool, default=False, help='this turns off diff aug during distillation')

    parser.add_argument('--texture', action='store_true', help="will distill textures instead")
    parser.add_argument('--canvas_size', type=int, default=2, help='size of synthetic canvas')
    parser.add_argument('--canvas_samples', type=int, default=1, help='number of canvas samples per iteration')


    parser.add_argument('--max_files', type=int, default=None, help='number of expert files to read (leave as None unless doing ablations)')
    parser.add_argument('--max_experts', type=int, default=None, help='number of experts to read per file (leave as None unless doing ablations)')
    parser.add_argument('--seed', type=int, default=3407, help='random seed')

    parser.add_argument('--force_save', action='store_true', help='this will save images for 50ipc')
    parser.add_argument('--patch', action='store_true', help='this will weight as patch')
    parser.add_argument('--patch_size', type=int, default=2, help='patch_size')
    parser.add_argument('--factor', type=int, default=1, help='patch_size')
    parser.add_argument('--rank_size', type=int, default=8, help='rank_size')
    parser.add_argument('--control_size', type=int, default=144, help='rank_size')
    parser.add_argument('--mixup', action='store_true', help='init mix')
    parser.add_argument('--score', action='store_true', help='scorenet')
    parser.add_argument('--color', action='store_true', help='scorenet')
    parser.add_argument('--low_rank', action='store_true', help='low_rank')
    parser.add_argument('--low_rank_control', action='store_true', help='low_rank_control')
    parser.add_argument('--dictionary', action='store_true', help='dict')
    parser.add_argument('--spilt', action='store_true', help='class_spilt')
    parser.add_argument('--multi', action='store_true', help='multi_low_rank')
    parser.add_argument('--cross_opt', action='store_true', help='cross_opt')
    parser.add_argument('--classifier', action='store_true', help='classifier only')
    parser.add_argument('--soft_label', action='store_true', help='add soft label')
    parser.add_argument('--temperature', action='store_true', help='add soft label')
    parser.add_argument('--downsample', type=int, default=1, help='upsample')
    args = parser.parse_args()

    main(args)


