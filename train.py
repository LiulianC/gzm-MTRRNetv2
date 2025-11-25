import os
import argparse
import torch
import warnings
import torch.nn.functional as F
from tqdm import tqdm
import time
from torchvision.utils import save_image 
from multiprocessing import freeze_support
import datetime
import csv

from dataset.quality_index import *
from dataset.new_dataset1 import *
from torch.utils.data import ConcatDataset

from MTRRNet import MTRREngine
from early_stop import EarlyStopping
from customloss import CustomLoss
from torch import amp
scaler = amp.GradScaler()
from MTRR_option import build_train_opts, get_lr_map, build_optimizer_and_scheduler, build_early_stopping
from set_seed import set_seed 
from torchvision.utils import make_grid
from util.csv import write_csv_row, _to_float
from util.eval_util import _collect_and_zero_probs, eval_no_dropout
from util.color_enhance import hist_match_batch_tensor
from psdLoss.spec_loss_pack import SpecularityNetLossPack

warnings.filterwarnings('ignore')
opts = build_train_opts()
opts.batch_size_train = opts.batch_size_train
opts.batch_size_test = opts.batch_size_test
opts.shuffle = True
opts.display_id = -1  
opts.num_workers = 0

opts.always_print = opts.always_print
opts.debug_monitor_layer_stats = opts.debug_monitor_layer_stats
opts.debug_monitor_layer_grad = opts.debug_monitor_layer_grad


opts.epoch = opts.epoch
opts.sampler_size1 = opts.sampler_size1
opts.sampler_size2 = opts.sampler_size2
opts.sampler_size3 = opts.sampler_size3
opts.sampler_size4 = opts.sampler_size4 
opts.sampler_size5 = opts.sampler_size5
opts.test_size = opts.test_size
opts.model_path = opts.model_path
opts.reset_best = opts.reset_best
opts.color_enhance = opts.color_enhance

opts.scheduler_type = opts.scheduler_type  # 'plateau' or 'cosine'

 

# nohup /home/gzm/cp310pt26/bin/python /home/gzm/gzm-MTRRNetv2/train.py > /home/gzm/gzm-MTRRNetv2/train.log 2>&1 &

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MTRREngine(opts, device)
# model.count_parameters()



fit_datadir = '/home/gzm/gzm-RDNet1/dataset/laparoscope_gen'
fit_data = DSRTestDataset(datadir=fit_datadir, fns='/home/gzm/gzm-RDNet1/dataset/laparoscope_gen_index/train1.txt',size=opts.sampler_size1, enable_transforms=False,if_align=True,real=False, HW=[256,256])

tissue_gen = '/home/gzm/gzm-MTRRVideo/data/tissue_gen'
tissue_gen_data = DSRTestDataset(datadir=tissue_gen, fns='/home/gzm/gzm-MTRRVideo/data/tissue_gen_index/train1.txt',size=opts.sampler_size2, enable_transforms=False,if_align=True,real=False, HW=[256,256])

tissue_dir = '/home/gzm/gzm-MTRRVideo/data/tissue_real'
tissue_data = DSRTestDataset(datadir=tissue_dir,fns='/home/gzm/gzm-MTRRVideo/data/tissue_real_index/train1.txt',size=opts.sampler_size3, enable_transforms=True, unaligned_transforms=False, if_align=True,real=True, HW=[256,256], color_match=False)

VOCroot = "/home/gzm/gzm-RDNet1/dataset/VOC2012"
VOCjson_file = "/home/gzm/gzm-RDNet1/dataset/VOC2012/VOC_results_list.json"
VOCdataset = VOCJsonDataset(VOCroot, VOCjson_file, size=opts.sampler_size4, enable_transforms=False, HW=[256, 256])

HyperKroot = "/home/gzm/gzm-MTRRNetv2/data/EndoData"
HyperKJson = "/home/gzm/gzm-MTRRNetv2/data/EndoData/test.json"
HyperK_data = HyperKDataset(root=HyperKroot, json_path=HyperKJson, start=343, end=369, size=opts.sampler_size5, enable_transforms=True, unaligned_transforms=False, if_align=True, HW=[256,256], flag=None, color_jitter=False)

# 使用ConcatDataset方法合成数据集 能自动跳过空数据集
train_data = ConcatDataset([fit_data, tissue_gen_data, tissue_data, VOCdataset, HyperK_data])
train_loader = torch.utils.data.DataLoader(train_data, batch_size=opts.batch_size_train, shuffle=opts.shuffle, num_workers = opts.num_workers, drop_last=False, pin_memory=True)



test_data_dir1 = '/home/gzm/gzm-MTRRVideo/data/tissue_real'
test_data1 = DSRTestDataset(datadir=test_data_dir1, fns='/home/gzm/gzm-MTRRVideo/data/tissue_real_index/eval1.txt', enable_transforms=False, if_align=True, real=True, HW=[256,256], size=opts.test_size[0], color_match=False)

test_data_dir2 = '/home/gzm/gzm-MTRRVideo/data/hyperK_000'
test_data2 = TestDataset(datadir=test_data_dir2, fns='/home/gzm/gzm-MTRRVideo/data/hyperK_000_list.txt', enable_transforms=False, if_align=True, real=True, HW=[256,256], size=opts.test_size[1])

test_data_dir3 = '/home/gzm/gzm-RDNet1/dataset/laparoscope_gen'
test_data3 = DSRTestDataset(datadir=test_data_dir3, fns='/home/gzm/gzm-RDNet1/dataset/laparoscope_gen_index/eval1.txt', enable_transforms=False, if_align=True, real=True, HW=[256,256], size=opts.test_size[2])

VOCroot1 = "/home/gzm/gzm-RDNet1/dataset/VOC2012"
VOCjson_file1 = "/home/gzm/gzm-RDNet1/dataset/VOC2012/VOC_results_list.json"
VOCdataset1 = VOCJsonDataset(VOCroot1, VOCjson_file1, size=opts.test_size[3], enable_transforms=True, HW=[256, 256])

HyperKroot_test = "/home/gzm/gzm-MTRRNetv2/data/EndoData"
HyperKJson_test = "/home/gzm/gzm-MTRRNetv2/data/EndoData/test.json"
HyperK_data_test = HyperKDataset(root=HyperKroot_test, json_path=HyperKJson_test, start=369, end=370, size=opts.test_size[4], enable_transforms=False, unaligned_transforms=False, if_align=True, HW=[256,256], flag=None, color_jitter=False)

HyperKroot_test = "/home/gzm/gzm-MTRRNetv2/data/EndoData"
HyperKJson_test = "/home/gzm/gzm-MTRRNetv2/data/EndoData/test.json"
HyperK_data_test2 = HyperKDataset(root=HyperKroot_test, json_path=HyperKJson_test, start=371, end=372, size=opts.test_size[5], enable_transforms=False, unaligned_transforms=False, if_align=True, HW=[256,256], flag=None, color_jitter=False)

# print("test data size: {}, {}, {}, {}, {}".format(len(test_data1), len(test_data2), len(test_data3), len(VOCdataset1), len(HyperK_data_test)))

test_data = ConcatDataset([test_data1, test_data2, test_data3, VOCdataset1, HyperK_data_test, HyperK_data_test2])
test_loader = torch.utils.data.DataLoader(test_data, batch_size=opts.batch_size_test, shuffle=False, num_workers=opts.num_workers, drop_last=False, pin_memory=True)



total_train_step = 0
total_test_step = 0

run_times = []

loss_function = CustomLoss().to(device)
PSD_LossFunc = SpecularityNetLossPack(opt=opts).to(device)



# tensorboard_writer = SummaryWriter("./logs")

if __name__ == '__main__':
    # set_seed(42)  # 设置随机种子 使得checkpoint和训练结果可复现

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # log_path = os.path.join('./logs', current_time)
    # os.makedirs(log_path, exist_ok=True)
    train_loss_path = os.path.join("./indexcsv",f"{current_time}_train_loss.csv")
    index_file_path = os.path.join("./indexcsv",f"{current_time}_index.csv")
    os.makedirs('./indexcsv', exist_ok=True)
    os.makedirs('./model_fit', exist_ok=True)

    channel_weights=[]
    spatial_weights=[]
    


    output_dir = './img_results'
    output_dir6 = os.path.join(output_dir,f'./output_test_{current_time}')
    output_dir7 = os.path.join(output_dir,f'./output_train_{current_time}')
    os.makedirs(output_dir, exist_ok=True)
    os.mkdir(output_dir6)
    os.mkdir(output_dir7)


    # 优化器、调度器、LR映射统一由 MTRR_option.py 构建
    optimizer, scheduler, lr_map, _group_stats = build_optimizer_and_scheduler(model.netG_T, opts, profile='train')



    # 定义早停（集中配置）
    early_stopping = build_early_stopping(opts)
    # 记录调度器类型供验证阶段 step 调用
    scheduler_type = opts.scheduler_type

    # 网络load 以及继承上次的epoch和学习参数
    if opts.model_path is not None and os.path.exists(opts.model_path):
        epoch_last_num,best_val_loss,best_val_psnr,best_val_ssim,early_stopping_counter = model.load_checkpoint(optimizer,scheduler)
    else:
        epoch_last_num = None
        best_val_loss = None
        best_val_psnr = None
        best_val_ssim = None
        early_stopping_counter = 0

    if opts.reset_best:
        best_val_loss, best_val_psnr, best_val_ssim = None, None, None

    min_loss = best_val_loss if best_val_loss is not None else float('inf') # 初始loss 尽可能大
    max_psnr = best_val_psnr if best_val_psnr is not None else 0
    max_ssim = best_val_ssim if best_val_ssim is not None else 0
    early_stopping.best_loss = float(min_loss)
    early_stopping.counter = early_stopping_counter

    train_begin=False
    epoch_start_num = 0
    if epoch_last_num is not None:
        if epoch_last_num < opts.epoch :
            train_begin=True
            epoch_start_num=epoch_last_num+1 # 上一轮是n 下一轮要+1
        else:
            print("模型last_epoch>epoch,模型已经训练完毕,不需要继续训练")
            exit(0)

    if opts.num_workers > 0:  # 多线程
        freeze_support()

    for i in range(epoch_start_num, opts.epoch):
        t1 = time.time()
        print("-----------第{}轮训练开始-----------".format(i + 1))
        print(" train data length: {} batch size: {}".format((len(train_loader))*opts.batch_size_train, opts.batch_size_train))
        model.train()

        # 加载新数据 DSRTestDataset 和 HyperKDataset 支持动态采样
        if i % 5 == 0:
            train_data.datasets[0].SampleNewItems()
            train_data.datasets[4].SampleNewItems()
        

        total_train_loss=0
        train_pbar = tqdm(
            train_loader,
            desc="Training",
            total=len(train_loader),
            ncols=170,  # 建议宽度根据指标数量调整
            dynamic_ncols=False,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        )      

        

        # 累计器
        epoch_loss_sum = {}     # 每个loss的总和
        epoch_step_count = 0
        train_loss_total = 0
        print('\n')
        print('current time:',datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        for t, data1 in enumerate(train_pbar):
            # print('\n')
            # print('input mean:',data1['input'].mean())
            # print('input std:',data1['input'].std())
            # print('input shape:',data1['input'].shape)
            model.set_input(data1)
            train_file_name = str(data1['fn']) 




            model.inference()



            visuals = model.get_current_visuals()
            train_input =   visuals['I'].to(device)
            train_ipt =   visuals['Ic'].to(device)
            train_label1 =  visuals['T'].to(device)
            train_label2 =  visuals['R'].to(device)
            # 列表的最后一个元素 shape B C H W
            train_fake_Ts = visuals['fake_Ts']
            train_fake_Rs = visuals['fake_Rs']
            train_rcmaps =  visuals['c_map'].to(device)
 
            # with amp.autocast(device_type='cuda'):


            _, _, _, _, _, all_loss0 = loss_function(train_fake_Ts[0], train_label1, train_ipt, train_rcmaps, train_fake_Rs[0], train_label2)
            _, _, _, _, _, all_loss1 = loss_function(train_fake_Ts[1], train_label1, train_ipt, train_rcmaps, train_fake_Rs[1], train_label2)
            _, _, _, _, _, all_loss2 = loss_function(train_fake_Ts[2], train_label1, train_ipt, train_rcmaps, train_fake_Rs[2], train_label2)
            loss_table, mse_loss, vgg_loss, ssim_loss, loss_spr, all_loss3 = loss_function(train_fake_Ts[3], train_label1, train_ipt, train_rcmaps, train_fake_Rs[3], train_label2)

            # all_loss0, _ = PSD_LossFunc.compute_total(train_fake_Rs[0], train_label1)
            # all_loss1, _ = PSD_LossFunc.compute_total(train_fake_Rs[1], train_label1)
            # all_loss2, loss_table = PSD_LossFunc.compute_total(train_fake_Rs[2], train_label1)
   
            all_loss = 0.5*all_loss0 + 0.5*all_loss1 + 0.5*all_loss2 + 1.0*all_loss3
            train_loss_total += all_loss.item()

            total_train_loss +=all_loss.item()
            if torch.isnan(all_loss2):
                print("⚠️  Loss is NaN! input:", train_file_name)

            for k, v in loss_table.items():
                v_val = _to_float(v)
                epoch_loss_sum[k] = epoch_loss_sum.get(k, 0.0) + v_val
            epoch_step_count += 1



            optimizer.zero_grad()
            # scaler.scale(all_loss).backward()
            all_loss.backward()

            # 防止梯度爆炸
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)


            # 打印每层梯度
            if opts.debug_monitor_layer_grad :
                model.monitor_layer_grad()

            # scaler.step(optimizer)
            # scaler.update()            
            optimizer.step()
            # 可选：同步并短暂休眠，降低平均 GPU 利用率与风扇噪声（不改变训练效果）
            if getattr(opts, 'throttle_ms', 0) > 0:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                time.sleep(opts.throttle_ms / 1000.0)

            
            total_train_step += 1


            # 保存
            grid_input = make_grid(train_input, nrow=train_input.size(0), padding=0, normalize=True)
            grid_fakeT = make_grid(train_fake_Ts[-1], nrow=train_fake_Ts[-1].size(0), padding=0, normalize=True)
            grid_label = make_grid(train_label1, nrow=train_label1.size(0), padding=0, normalize=True)
            grid_cmap = make_grid(train_rcmaps, nrow=train_rcmaps.size(0), padding=0, normalize=True)
            grid_fakeR = make_grid(train_fake_Rs[-1], nrow=train_fake_Rs[-1].size(0), padding=0, normalize=True)
            grid_label2 = make_grid(train_label2, nrow=train_label2.size(0), padding=0, normalize=True)
    
            grid_all = torch.cat([grid_input, grid_fakeT, grid_label, grid_cmap, grid_fakeR, grid_label2], dim=1)

            if i % 3 == 0 and i<11 and total_train_step%10==0:
                save_path = os.path.join(output_dir7, f'epoch{i}+{total_train_step}-loss{all_loss:.4g}train_grid.png')
                save_image(grid_all, save_path)
            elif i%10 == 0 and total_train_step%10==0:
                save_path = os.path.join(output_dir7, f'epoch{i}+{total_train_step}-loss{all_loss:.4g}train_grid.png')
                save_image(grid_all, save_path)





            if i % 1 == 0:
                current_lr = optimizer.param_groups[0]['lr']

            # if total_train_step % 50 == 0:
            #     model.apply_weight_constraints()


            train_pbar.set_postfix({'loss':all_loss.item(),'mseloss':mse_loss.item(), 'vggloss':vgg_loss.item(), 'ssimloss':ssim_loss.item(),'loss_spr':loss_spr.item(),'current_lr': current_lr})
            # train_pbar.set_postfix({'loss':all_loss.item(),'pixeloss':loss_table['pixel'].item(),'current_lr': current_lr})
            train_pbar.update(1)
        train_pbar.close()


        # 一轮结束：计算平均
        epoch_loss_avg = {k: (v / max(1, epoch_step_count)) for k, v in epoch_loss_sum.items()} 

        # 写CSV（每轮一次）
        file_exists = os.path.isfile(train_loss_path)        
        train_fields = ["epoch", "num_steps"] + list(epoch_loss_avg.keys())
        train_row = {"epoch": i + 1, "num_steps": epoch_step_count, **epoch_loss_avg}
        write_csv_row(train_loss_path, train_fields, train_row)               


        total_test_loss = 0
        total_test_step = 0
        total_test_psnr = 0
        total_test_ssim = 0

        model.eval()
        with torch.no_grad():
            with eval_no_dropout(model):
                print("test data length: {} batch size: {}".format(len(test_data),opts.batch_size_test))
                test_pbar = tqdm(
                    test_loader,
                    desc="Validating",
                    total=len(test_loader),
                    ncols=150,  # 建议宽度根据指标数量调整
                    dynamic_ncols=False,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
                )          

                # 指标累计器（加权到样本级）
                sum_psnr = 0.0
                sum_ssim = 0.0
                sum_lmse = 0.0
                sum_ncc  = 0.0
                sample_cnt = 0    

                for n1 , test_data1 in enumerate(test_pbar):
                    model.set_input(test_data1)
                    model.inference()
                    visuals_test = model.get_current_visuals()
                    test_imgs = visuals_test['I'].to(device)
                    test_ipt = visuals_test['Ic'].to(device)
                    test_label1 = visuals_test['T'].to(device)
                    test_label2 = visuals_test['R'].to(device)

                    test_fake_Ts = visuals_test['fake_Ts']
                    
                    test_fake_Rs = visuals_test['fake_Rs']
                    
                    test_rcmaps = visuals_test['c_map'].to(device)
                    

                    _,_,_,_,_,loss = loss_function(test_fake_Ts[-1], test_label1, test_ipt, test_rcmaps, test_fake_Rs[-1], test_label2)
                    # loss,_ = PSD_LossFunc.compute_total(test_fake_Rs[-1], test_label1)

                    total_test_loss += loss.item()

 

                    if opts.color_enhance:
                        test_fake_Ts[-1] = hist_match_batch_tensor(test_fake_Ts[-1], test_imgs)


                    # 计算psnr与ssim与NCC与LMSN
                    index = quality_assess(test_fake_Ts[-1].to('cpu'), test_label1.to('cpu'))
                    file_name, psnr, ssim, lmse, ncc = test_data1['fn'], index['PSNR'], index['SSIM'], index['LMSE'], index['NCC']
                    # 数据集返回时 只要batchsize不为0 就返回的是列表
                    res = {'file':str(file_name),'PSNR':psnr,'SSIM':ssim,'LMSE':lmse,'NCC':ncc}
                    total_test_psnr = total_test_psnr + res['PSNR']
                    total_test_ssim = total_test_ssim + res['SSIM']
                    
                    # 检查文件是否存在，不存在则写入表头
                    file_exists1 = os.path.isfile(index_file_path)


                    file_name = test_data1['fn']          # 可能是字符串或列表
                    psnr = _to_float(index['PSNR'])
                    ssim = _to_float(index['SSIM'])
                    lmse = _to_float(index['LMSE'])
                    ncc  = _to_float(index['NCC'])

                    # 统计该 batch 对应的样本数
                    try:
                        batch_n = len(file_name)          # 列表时
                    except TypeError:
                        batch_n = 1                       # 字符串/标量时

                    # 累计（按样本数加权）
                    sum_psnr += psnr * batch_n
                    sum_ssim += ssim * batch_n
                    sum_lmse += lmse * batch_n
                    sum_ncc  += ncc  * batch_n
                    sample_cnt += batch_n

                    if getattr(opts, 'throttle_ms', 0) > 0:
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        time.sleep(opts.throttle_ms / 1000.0)





                    B = test_imgs.size(0)
                    grid_in  = make_grid(test_imgs, nrow=B, padding=0)
                    grid_out = make_grid(test_fake_Ts[-1], nrow=B, padding=0)
                    grid_tgt = make_grid(test_label1, nrow=B, padding=0)
                    grid_cmap = make_grid(test_rcmaps, nrow=B, padding=0)

                    grid_all = torch.cat([grid_in, grid_out, grid_tgt, grid_cmap], dim=1)  # dim=1 是 H 维度

                    if (i) % 1 == 0 and total_test_step % 1 == 0 and i<=10:                        
                        save_image(grid_all, os.path.join(output_dir6, f'epoch{i}+{total_test_step}+psnr{psnr:.4g}+ssim{ssim:.4g}.png'))

                    elif i>=15 and (i) % 5 == 0 :
                        save_image(grid_all, os.path.join(output_dir6, f'epoch{i}+{total_test_step}+psnr{psnr:.4g}+ssim{ssim:.4g}.png'))





                    total_test_step += 1
                    test_pbar.set_postfix({'loss':loss.item(),'psnr':res['PSNR'], 'ssim':res['SSIM'], 'lmse':res['LMSE'],'ncc': res['NCC']})
                    test_pbar.update(1)

                # 计算验证集“样本平均”指标
                avg_psnr = sum_psnr / max(1, sample_cnt)
                avg_ssim = sum_ssim / max(1, sample_cnt)
                avg_lmse = sum_lmse / max(1, sample_cnt)
                avg_ncc  = sum_ncc  / max(1, sample_cnt)
                # 写CSV（每轮一次）
                val_fields = ["epoch", "num_samples", "PSNR", "SSIM", "LMSE", "NCC"]
                val_row    = {"epoch": i + 1, "num_samples": sample_cnt,
                            "PSNR": avg_psnr, "SSIM": avg_ssim, "LMSE": avg_lmse, "NCC": avg_ncc}
                write_csv_row(index_file_path, val_fields, val_row)

                # 更新学习率（根据调度器类型调用不同的step方法）
                if scheduler_type == 'plateau':
                    scheduler.step(total_test_loss)
                elif scheduler_type == 'cosine':
                    scheduler.step()
                test_pbar.close()

                epoch_num = {"epoch":i}
                # model.state_dict.update(epoch)

        avg_test_loss = total_test_loss / total_test_step
        avg_test_psnr = total_test_psnr / total_test_step
        avg_test_ssim = total_test_ssim / total_test_step
        if avg_test_psnr > max_psnr:
            print(f"psnr from {max_psnr:.5f} improve {avg_test_psnr:.5f} ")
            max_psnr = avg_test_psnr
        else:
            print(f"psnr did not improve : best {max_psnr:.5f} now {avg_test_psnr:.5f} ")
            
        if avg_test_ssim > max_ssim:
            print(f"ssim from {max_ssim:.5f} improve {avg_test_ssim:.5f} ")
            max_ssim = avg_test_ssim
        else:
            print(f"ssim did not improve : best {max_ssim:.5f} now {avg_test_ssim:.5f} ")
        
        

        if model.net_c is not None:
            state = {
                'epoch': i, 
                'net_c': model.net_c.state_dict(),
                'netG_T': model.netG_T.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': avg_test_loss,
                'best_val_psnr': max_psnr,
                'best_val_ssim': max_ssim,
                'early_stopping_counter': early_stopping.counter,
            }
        else:
            state = {
                'epoch': i, 
                'netG_T': model.netG_T.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': avg_test_loss,
                'best_val_psnr': max_psnr,
                'best_val_ssim': max_ssim,                
                'early_stopping_counter': early_stopping.counter,
            }

            
        # 早停检查
        early_stopping(avg_test_loss)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {i}!") 
            break

        t2 = time.time()
        run_times.append(t2 - t1)
        if (i) % 1 == 0:
            print(f'processing the {i + 1} epoch, {(run_times[-1]/60):.4f} mins passed by')

        if opts.training:
            if avg_test_loss<min_loss:
                min_loss = avg_test_loss 
                print(f"New best model at epoch {i} with loss {min_loss:.4f}")
                torch.save(state, "./model_fit/model_{}.pth".format(i + 1))        
            else:
                print(f"Epoch {i} did not improve. Best loss:{min_loss:.4f}  now: {avg_test_loss:.4f}")   

            if (i) % 1 == 0:
                torch.save(state, "./model_fit/model_latest.pth".format(i + 1))
                print("模型已保存")         

        # 清理缓存
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    print("==============训练已经结束==============")
    if opts.training:
        torch.save(state, "./model_fit/model_latest.pth".format(opts.epoch))
        print("模型已保存")

# tensorboard_writer.close()













