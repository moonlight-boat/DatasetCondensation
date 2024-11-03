import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
import csv
import os
from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug
def argmin(array):
    return torch.argsort(torch.tensor(array), descending=True)
from torchvision.utils import save_image, make_grid

def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='SS', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--num_exp', type=int, default=5, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=20, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data') # it can be small for speeding up with little performance drop
    parser.add_argument('--Iteration', type=int, default=4000, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=1.0, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='real', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
    args = parser.parse_args()
    args.method = 'DM'
    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = False if args.dsa_strategy in ['none', 'None'] else True
    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    eval_it_pool = np.arange(0, args.Iteration+1, args.Iteration).tolist() if args.eval_mode == 'S' or args.eval_mode == 'SS' else [args.Iteration] # The list of iterations when we evaluate models and record results.
    print('eval_it_pool: ', eval_it_pool)
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)
    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []
    data_save = []
    for exp in range(args.num_exp):
        # args.ipc =  exp + 1
        print('\n================== Exp %d ==================\n '%exp)
        print('Hyper-parameters: \n', args.__dict__)
        print('Evaluation model pool: ', model_eval_pool)
        ''' organize the real dataset '''
        images_all = []
        labels_all = []
        indices_class = [[] for c in range(num_classes)]
        images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        images_all = torch.cat(images_all, dim=0).to(args.device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)
        for c in range(num_classes):
            print('class c = %d: %d real images'%(c, len(indices_class[c])))
        def get_images(c, n): # get random n images from class c
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle]
        for ch in range(channel):
            print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))
        ''' initialize the synthetic data '''
        image_syn = torch.randn(size=(num_classes*args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
        label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]
        # if args.init == 'real':
        #     for c in range(num_classes):
        #         image_syn.data[c*args.ipc:(c+1)*args.ipc] = get_images(c, args.ipc).detach().data
        # else:
        #     print('initialize synthetic data from random noise')
        # class_image_counts = [6,10,17,19,10,5,15,12,2,4]
        # class_image_counts = [14, 8, 5, 1, 6, 14, 10, 6, 17, 19]
        # class_image_counts = [14, 13, 12, 11, 10, 10, 9, 8, 7, 6]
        # class_image_counts = [9, 11, 12, 15, 13, 10, 9,7, 6, 8]
        # class_image_counts = [8,8,11,14,11,13,9,9,6,11]
        # class_image_counts = [8,8,11,14,11,13,9,9,6,11]
        # class_image_counts =[12,12,9,6,9,7,11,11,14,9]
        class_image_counts =[60,62,43,31,43,34,55,56,70,46]
        class_image_counts = np.array(class_image_counts) * 1
        print(class_image_counts)
        # 初始化空的 image_syn 和 label_syn
        total_images = sum(class_image_counts)
        image_syn = torch.randn(size=(total_images, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
        label_syn = torch.zeros(total_images, dtype=torch.long, device=args.device)  # 用于存储每个类的标签
        # 初始化图像和标签
        if args.init == 'real':
            start_idx = 0
            for c, num_images in enumerate(class_image_counts):
                # 生成属于类 c 的 num_images 张图像
                new_images = get_images(c, num_images).detach().data
                # 将生成的图像赋值到 image_syn 中的对应位置
                image_syn.data[start_idx:start_idx + num_images] = new_images
                # 将标签对应设置为类 c
                label_syn[start_idx:start_idx + num_images] = c
                
                # 更新索引，指向下一个类别的起始位置
                start_idx += num_images
        else:
            print('initialize synthetic data from random noise')
        ''' training '''
        optimizer_img = torch.optim.SGD([image_syn, ], lr=args.lr_img, momentum=0.5) # optimizer_img for synthetic data
        optimizer_img.zero_grad()
        print('%s training begins'%get_time())
        csv_file = os.path.join(args.save_path, 'class_acc_test_results_ipc_ship_cat.csv')
        # 如果文件不存在，则创建文件并写入表头
        if not os.path.exists(csv_file):
            with open(csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                header = ['Iteration', 'Total_Accuracy'] + [f'Class_{i+1}_Acc' for i in range(num_classes)]+["合成图片总数"]
                writer.writerow(header)
        for it in range(args.Iteration+1):
            ''' Evaluate synthetic data '''
            if it in eval_it_pool and it !=0:
                for model_eval in model_eval_pool:
                    print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))
                    print('DSA augmentation strategy: \n', args.dsa_strategy)
                    print('DSA augmentation parameters: \n', args.dsa_param.__dict__)
                    accs = []
                    class_acc_tests= []
                    for it_eval in range(args.num_eval):
                        net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model
                        image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach()) # avoid any unaware modification
                        _, acc_train, acc_test,class_acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args,return_class_acc=True)
                        accs.append(acc_test)
                        class_acc_tests.append(class_acc_test)
                    class_image_counts = []
                    for c in range(num_classes):
                        class_image_counts.append((label_syn_eval == c).sum().item())  # 计算每个类别的图片数量
                    class_acc_test = sum(class_acc_tests)/len(class_acc_tests)
                    with open(csv_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        row = [it, np.mean(accs)] + list(class_acc_test) + [image_syn_eval.shape[0]]
                        writer.writerow(row)
                        writer.writerow(['']+[''] + class_image_counts + [''])
                    print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs)))
                    if it == args.Iteration: # record the final results
                        accs_all_exps[model_eval] += accs
                ''' visualize and save '''
                save_name = os.path.join(args.save_path, 'vis_%s_%s_%s_%dipc_exp%d_iter%d.png'%(args.method, args.dataset, args.model, args.ipc, exp, it))
                image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
                for ch in range(channel):
                    image_syn_vis[:, ch] = image_syn_vis[:, ch]  * std[ch] + mean[ch]
                image_syn_vis[image_syn_vis<0] = 0.0
                image_syn_vis[image_syn_vis>1] = 1.0
                # save_image(image_syn_vis, save_name, nrow=args.ipc) # Trying normalize = True/False may get better visual effects.
                images_per_class = []
                for c in range(num_classes):
                    indices = (label_syn == c).nonzero(as_tuple=True)[0]
                    class_images = image_syn[indices]
                    images_per_class.append(class_images)
                all_images = torch.cat(images_per_class, dim=0)
                grid = make_grid(all_images, nrow=args.ipc*10)
                save_image(grid, save_name)
                if image_syn.shape[0] > args.ipc * 10 * num_classes:
                    # 按照 class_acc_test 从高到低的顺序获取类别索引
                    list_c = argmin(class_acc_test)  # 例如 [2, 0, 1, 3] 表示第2类准确率最高,依此类推
                    # 枚举排序后的每个类别 c，并根据排名增加不同数量的图像
                    for rank, c in enumerate(list_c):
                        # 根据排名计算要增加的图像数量 (num_classes - rank)，rank 为类 c 的排名
                        num_images_to_add = num_classes - rank -1 
                        # 找到属于类 c 的所有图像的索引
                        indices = (label_syn == c).nonzero(as_tuple=True)[0]
                        if len(indices) > 0:
                            # 从原来的 synimage 中复制属于类 c 的图像
                            original_image_syn = image_syn[indices]
                            # 随机选择 original_image_syn 中的一个图像
                            random_idx = torch.randint(0, original_image_syn.shape[0], (1,)).item()
                            selected_image_syn = original_image_syn[random_idx].unsqueeze(0)  # 选中的图像，保持维度一致
                            # 复制该图像 num_images_to_add 次
                            new_image_syn = selected_image_syn.repeat(num_images_to_add, 1, 1, 1)
                            new_label_syn = torch.full((new_image_syn.shape[0],), c, dtype=torch.long, device=image_syn.device)
                            # 将新生成的图像和标签插入到原有的合成数据中
                            insert_pos = indices[-1].item() + 1
                            image_syn = torch.cat((image_syn[:insert_pos], new_image_syn, image_syn[insert_pos:]), dim=0).clone().detach().requires_grad_(True)
                            label_syn = torch.cat((label_syn[:insert_pos], new_label_syn, label_syn[insert_pos:]), dim=0)
                            # 优化器
                            optimizer_img = torch.optim.SGD([image_syn, ], lr=args.lr_img, momentum=0.5)  # optimizer_img for synthetic data
                            optimizer_img.zero_grad()

            ''' Train synthetic data '''
            net = get_network(args.model, channel, num_classes, im_size).to(args.device) # get a random model
            net.train()
            for param in list(net.parameters()):
                param.requires_grad = False
            embed = net.module.embed if torch.cuda.device_count() > 1 else net.embed # for GPU parallel
            loss_avg = 0
            ''' update synthetic data '''
            if 'BN' not in args.model: # for ConvNet
                loss = torch.tensor(0.0).to(args.device)
                for c in range(num_classes):
                    img_real = get_images(c, args.batch_real)
                    # img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))
                    indices = (label_syn == c).nonzero(as_tuple=True)[0]
                    # 获取这些索引对应的图像
                    img_syn = image_syn[indices].reshape((len(indices), channel, im_size[0], im_size[1]))
                    if args.dsa:
                        seed = int(time.time() * 1000) % 100000
                        img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                        img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)
                    output_real = embed(img_real).detach()
                    output_syn = embed(img_syn)
                    loss += (class_image_counts[c]/sum(class_image_counts))*torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0))**2)
                    # weight = torch.exp(-1* class_image_counts[c] / sum(class_image_counts))
                    # loss += weight *torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0))**2)
            else: # for ConvNetBN
                images_real_all = []
                images_syn_all = []
                loss = torch.tensor(0.0).to(args.device)
                for c in range(num_classes):
                    img_real = get_images(c, args.batch_real)
                    img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))

                    if args.dsa:
                        seed = int(time.time() * 1000) % 100000
                        img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                        img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                    images_real_all.append(img_real)
                    images_syn_all.append(img_syn)

                images_real_all = torch.cat(images_real_all, dim=0)
                images_syn_all = torch.cat(images_syn_all, dim=0)

                output_real = embed(images_real_all).detach()
                output_syn = embed(images_syn_all)

                loss += torch.sum((torch.mean(output_real.reshape(num_classes, args.batch_real, -1), dim=1) - torch.mean(output_syn.reshape(num_classes, args.ipc, -1), dim=1))**2)
            optimizer_img.zero_grad()
            loss.backward()
            optimizer_img.step()
            loss_avg += loss.item()
            loss_avg /= (num_classes)
            if it%10 == 0:
                print('%s iter = %05d, loss = %.4f' % (get_time(), it, loss_avg))
            if it == args.Iteration: # only record the final results
                data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
                torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, }, os.path.join(args.save_path, 'res_%s_%s_%s_%dipc.pt'%(args.method, args.dataset, args.model, args.ipc)))
    print('\n==================== Final Results ====================\n')
    for key in model_eval_pool:
        accs = accs_all_exps[key]
        print('Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%'%(args.num_exp, args.model, len(accs), key, np.mean(accs)*100, np.std(accs)*100))



if __name__ == '__main__':
    main()

