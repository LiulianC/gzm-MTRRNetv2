import math
import os.path
import os.path
import random
from os.path import join

import cv2
import numpy as np
import torch.utils.data
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from PIL import Image
from scipy.signal import convolve2d
from scipy.stats import truncnorm
from dataset.image_folder import make_dataset
from dataset.torchdata import Dataset as BaseDataset
from dataset.transforms import to_tensor
from torch.utils.data import Sampler
from torch.utils.data import DataLoader

# 按目标宽度等比缩放图像，保持宽高比，调整高度为最接近的偶数
def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    h = math.ceil(h / 2.) * 2  # round up to even
    return img.resize((w, h), Image.BICUBIC)

# 按目标高度等比缩放图像，保持宽高比，调整宽度为最接近的偶数。
def __scale_height(img, target_height):
    ow, oh = img.size
    if (oh == target_height):
        return img
    h = target_height
    w = int(target_height * ow / oh)
    w = math.ceil(w / 2.) * 2
    return img.resize((w, h), Image.BICUBIC)

# 对输入的两张图像（如退化图像与目标清晰图像）进行 ​​同步或异步的预处理与增强
def paired_data_transforms(img_1, img_2, img_3, unaligned_transforms=False):

    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    # 随机缩放​​：保持宽高比，缩放到 [256, 640] 之间的随机偶数尺寸
    target_size = int(random.randint(256, 640) / 2.) * 2
    ow, oh = img_1.size
    if ow >= oh:
        img_1 = __scale_height(img_1, target_size)
        img_2 = __scale_height(img_2, target_size)
        img_3 = __scale_height(img_3, target_size)
    else:
        img_1 = __scale_width(img_1, target_size)
        img_2 = __scale_width(img_2, target_size)
        img_3 = __scale_width(img_3, target_size)

    # ​​随机水平翻转​​（概率50%）
    if random.random() < 0.5:
        img_1 = TF.hflip(img_1)
        img_2 = TF.hflip(img_2)
        img_3 = TF.hflip(img_3)

    # 随机旋转​​（90°/180°/270°，概率50%）
    if random.random() < 0.5:
        angle = random.choice([90, 180, 270])
        img_1 = TF.rotate(img_1, angle)
        img_2 = TF.rotate(img_2, angle)
        img_3 = TF.rotate(img_3, angle)

    # 随机裁剪​​：随机在（i,j）位置会有小偏移
    i, j, h, w = get_params(img_1, (256, 256)) # （i,j）是左上角坐标 h w是目标大小
    img_1 = TF.crop(img_1, i, j, h, w) # 这里就已经变成目标大小了


    # 异步位移裁剪​​（若启用）：对第二张图像施加轻微的位置偏移
    if unaligned_transforms:
        # print('random shift')
        i_shift = random.randint(-10, 10)
        j_shift = random.randint(-10, 10)
        i += i_shift
        j += j_shift

    img_2 = TF.crop(img_2, i, j, h, w)


    # 异步位移裁剪​​（若启用）：对第二张图像施加轻微的位置偏移
    if unaligned_transforms:
        # print('random shift')
        i_shift = random.randint(-10, 10)
        j_shift = random.randint(-10, 10)
        i += i_shift
        j += j_shift

    img_3 = TF.crop(img_3, i, j, h, w)

    return img_1, img_2, img_3 # 三张图片一样大 




# ReflectionSynthesis 用于 ​​模拟真实场景中的反射效果合成​​
class ReflectionSynthesis(object):
    def __init__(self):
        # Kernel Size of the Gaussian Blurry
        self.kernel_sizes = [5, 7, 9, 11]
        self.kernel_probs = [0.1, 0.2, 0.3, 0.4]

        # Sigma of the Gaussian Blurry
        self.sigma_range = [2, 5]
        self.alpha_range = [0.8, 1.0]
        self.beta_range = [0.4, 1.0]

    def __call__(self, T_, R_):
        T_ = np.asarray(T_, np.float32) / 255.
        R_ = np.asarray(R_, np.float32) / 255.

        # 模拟真实反射的散射效应（如毛玻璃、水面波纹导致的模糊）。
        # 随机选择高斯核尺寸与标准差
        kernel_size = np.random.choice(self.kernel_sizes, p=self.kernel_probs)
        sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])
        kernel = cv2.getGaussianKernel(kernel_size, sigma)
        kernel2d = np.dot(kernel, kernel.T) # 生成二维高斯核
        # 对反射层每个通道进行卷积（模糊）
        for i in range(3):
            R_[..., i] = convolve2d(R_[..., i], kernel2d, mode='same')

        # 生成截断正态分布的随机系数（限制在合理范围）
        a1 = truncnorm((0.82 - 1.109) / 0.118, (1.42 - 1.109) / 0.118, loc=1.109, scale=0.118)
        a2 = truncnorm((0.85 - 1.106) / 0.115, (1.35 - 1.106) / 0.115, loc=1.106, scale=0.115)
        a3 = truncnorm((0.85 - 1.078) / 0.116, (1.31 - 1.078) / 0.116, loc=1.078, scale=0.116)
        
        # 调整透射层各通道强度
        T_[..., 0] *= a1.rvs()  
        T_[..., 1] *= a2.rvs()  
        T_[..., 2] *= a3.rvs()  
        # 反射强度控制​
        b = np.random.uniform(self.beta_range[0], self.beta_range[1])
        T, R = T_, b * R_ # # 反射层强度缩放
        
        if random.random() < 0.7:
            # # 光学叠加模型：I = T + R - T*R (避免过曝)
            I = T + R - T * R

        else:
            # 简单加法 + 自适应校正
            I = T + R
            if np.max(I) > 1:
                m = I[I > 1]
                m = (np.mean(m) - 1) * 1.3
                I = np.clip(T + np.clip(R - m, 0, 1), 0, 1)

        return T_, R_, I




# dataloader.reset() 需要dataset.reset()实现才能实现
class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, *args, **kwargs):
        super(DataLoader, self).__init__(dataset, batch_size, shuffle, *args, **kwargs)
        self.shuffle = shuffle

    def reset(self):
        if self.shuffle:
            print('Reset Dataset...')
            self.dataset.reset()


# dataloader的抽样方法自定义
class CustomSampler(Sampler):
    def __init__(self, size1, size2, size3, samples_size1, samples_size2, samples_size3):
        self.size1 = size1
        self.size2 = size2
        self.size3 = size3
        self.samples_size1 = samples_size1
        self.samples_size2 = samples_size2
        self.samples_size3 = samples_size3

    def __iter__(self):
        # 生成合成风景的随机索引
        indices1 = torch.randperm(self.size1)[:self.samples_size1]
        # 生成合成血管的随机索引，并转换为全局索引
        indices2 = torch.randperm(self.size2)[:self.samples_size2] + self.samples_size1
        # 生成真实数据的随机索引，并转换为全局索引
        indices3 = torch.randperm(self.size3)[:self.samples_size3] + self.samples_size1 + self.samples_size2
        # 合并并打乱索引
        combined_indices = torch.cat([indices1, indices2, indices3])
        combined_indices = combined_indices[torch.randperm(len(combined_indices))]
        return iter(combined_indices.tolist())

    def __len__(self):
        return self.samples_size1 + self.samples_size2 + self.samples_size3






class DSRDataset(BaseDataset):
    def __init__(self, datadir, fns=None, size=None, enable_transforms=True):
        super(DSRDataset, self).__init__()
        self.size = size
        self.datadir = datadir
        self.enable_transforms = enable_transforms # 是否启用数据增强（默认启用）
        sortkey = lambda key: os.path.split(key)[-1]
        self.paths = sorted(make_dataset(datadir, fns), key=sortkey)
        if size is not None:
            self.paths = np.random.choice(self.paths, size)

        self.syn_model = ReflectionSynthesis()
        self.reset(shuffle=False)

    # 决定合成图像 谁是背景 谁是反射
    def reset(self, shuffle=True):
        if shuffle:
            random.shuffle(self.paths)
        num_paths = len(self.paths) // 2 # 打乱所有路径
        self.B_paths = self.paths[0:num_paths] # 前一半作为背景
        self.R_paths = self.paths[num_paths:2 * num_paths] # 后一半作为反射



    # 数据增强​​：若启用，对输入图像对进行同步变换（如缩放、翻转）
    # ​​反射合成​​：调用 syn_model 生成混合图像 M
    # 张量转换​​：将 PIL 图像转换为 [0,1] 范围的张量
    def data_synthesis(self, t_img, r_img):
        if self.enable_transforms:
            t_img, r_img = paired_data_transforms(t_img, r_img)

        t_img, r_img, m_img = self.syn_model(t_img, r_img)

        B = TF.to_tensor(t_img)
        R = TF.to_tensor(r_img)
        M = TF.to_tensor(m_img)

        return B, R, M


    def __getitem__(self, index):
        index_B = index % len(self.B_paths) # 背景路径索引
        index_R = index % len(self.R_paths) # 反射路径索引

        B_path = self.B_paths[index_B]
        R_path = self.R_paths[index_R]

        t_img = Image.open(B_path).convert('RGB') # 背景图像
        r_img = Image.open(R_path).convert('RGB') # 反射图像

        B, R, M = self.data_synthesis(t_img, r_img) # 合成混合图像
        fn = os.path.basename(B_path) # filename
        return {'input': M, 'target_t': B, 'target_r': M-B, 'fn': fn, 'real': False}
        # 混合图像 [C, H, W]    M
        # 背景图像 [C, H, W]    B
        # 残差图像 [C, H, W]    M-B
        # 文件名（用于结果保存）    fn
        # 标记为合成数据（与真实数据区分）  real

    def __len__(self):
        if self.size is not None:
            return min(max(len(self.B_paths), len(self.R_paths)), self.size)
        else:
            return max(len(self.B_paths), len(self.R_paths))










# size：限制加载的数据量 当size=0时 len=0 数据集程序会当成index_out错误自动停止
class DSRTestDataset(BaseDataset):
    def __init__(self, datadir, fns=None, size=None, enable_transforms=False, unaligned_transforms=False,
                 round_factor=1, flag=None, if_align=True, real=False, HW=[256,256],
                 color_match=False, color_jitter=False):
        super(DSRTestDataset, self).__init__()
        self.size = size
        self.datadir = datadir
        self.fns = fns 
        self.enable_transforms = enable_transforms
        self.unaligned_transforms = unaligned_transforms
        self.round_factor = round_factor
        self.flag = flag
        self.if_align = True # if_align
        self.real = real
        self.HW = HW
        # 是否进行颜色直方图匹配：将 m_img 的颜色分布匹配到 t_img
        self.color_match = color_match
        self.color_jitter = color_jitter
        
        self.I_paths = []
        self.R_paths = []
        self.T_paths = []



        if self.fns != None : # 测试集验证机训练集已经分在了txt文件中 好处是不用对图片数据集挑挑拣拣
            with open(self.fns, 'r') as f:
                for line in f :
                    if line.strip():
                        file = line.strip()
                        if file.endswith('-input.png'):
                            self.I_paths.append(os.path.join(self.datadir, file))
                            T = file.replace('-input.png', '-label1.png')
                            self.T_paths.append((os.path.join(self.datadir, T)))
                            R = file.replace('-input.png', '-label2.png')
                            self.R_paths.append((os.path.join(self.datadir, R)))    
                        else:
                            # print("Warning: file %s does not end with '-input.png'" %file)
                            pass

            # print('I_paths num:', len(self.I_paths), 'T_paths num:', len(self.T_paths), 'R_paths num:', len(self.R_paths))

        else: # 采用目录来分测试集验证集
            for file in os.listdir(self.datadir):
                if file.endswith('-input.png'):
                    self.I_paths.append(os.path.join(self.datadir, file))
                    T = file.replace('-input.png', '-label1.png')
                    self.T_paths.append((os.path.join(self.datadir, T)))
                    R = file.replace('-input.png', '-label2.png')
                    self.R_paths.append((os.path.join(self.datadir, R)))
        
        
        self.SampleNewItems()

    def align(self, x1, x2, x3):
        h, w = self.HW[0], self.HW[1]
        h, w = h // 32 * 32, w // 32 * 32
        # h_new, w = h + (32 - h % 32), w + 
        x1 = x1.resize((w, h))
        x2 = x2.resize((w, h))
        x3 = x3.resize((w, h))
        return x1, x2, x3

    def __getitem__(self, index):
        filename = os.path.basename(self.I_paths_s[index]).replace('[', '').replace(']', '').replace("'", '').replace('.png', '')

        m_img = Image.open(self.I_paths_s[index]).convert('RGB')
        try:
            t_img = Image.open(self.T_paths_s[index]).convert('RGB')
        except Exception:
            t_img = m_img.copy()  # 如果没有目标图像，则使用混合图像作为目标
        
        # 可选：将 m_img 的颜色分布匹配到 t_img（直方图匹配）
        if self.color_match:
            try:
                m_img = self._hist_match_rgb(m_img, t_img)
            except Exception as e:
                # 颜色匹配失败则跳过，保持原图
                pass
            
        try:
            r_img = Image.open(self.R_paths_s[index]).convert('RGB')
        except Exception:
            r_img = Image.fromarray(np.clip(np.array(m_img, dtype=np.float32) - np.array(t_img, dtype=np.float32), 0, 255).astype(np.uint8))

        if self.enable_transforms:
            t_img, m_img, r_img = paired_data_transforms(t_img, m_img, r_img, self.unaligned_transforms)

        if self.if_align:
            t_img, m_img, r_img = self.align(t_img, m_img, r_img)

            
        if self.color_jitter:
            # 颜色扰动：默认只对 m_img 生效，并重算 r_img
            m_img, t_img, r_img = _maybe_color_jitter(
                m_img, t_img, r_img,
                p=0.8, brightness=0.2, contrast=0.2, saturation=0.3, hue=0.05,
                apply_to_label=True, recompute_residual=False
            )          

        B = TF.to_tensor(t_img)
        M = TF.to_tensor(m_img)
        R = TF.to_tensor(r_img)

        dic = {'input': M, 'target_t': B, 'fn': filename, 'real': self.real, 'target_r': R, 'Endo':False}
        if self.flag is not None:
            dic.update(self.flag) # 用于将一个字典（或键值对序列）的内容合并到当前字典中
        return dic

    # 返回数据集的实际长度，受 size 参数限制
    def __len__(self):
        if self.size is not None:
            return min(len(self.I_paths), self.size)
        else:
            return len(self.I_paths_s)

    def SampleNewItems(self):
        '''
            每几个epoches从全部数据库重新采样指定数量的数据 为了数据多样性
        '''
        # 随机抽样
        if self.size == 0:
            self.I_paths_s,self.T_paths_s,self.R_paths_s= [], [], []
            pass

        elif self.size is not None and self.size > 0 and self.size <= len(self.I_paths):
            total_length = len(self.I_paths)
            step = total_length // self.size
            
            # 在0到step-1之间随机选择起始点
            start_index = np.random.randint(0, step)
            
            # 生成等距采样索引
            indices = [start_index + i * step for i in range(self.size)]
            
            # 确保索引不越界
            indices = [i if i < total_length else i - total_length for i in indices]
            
            # 应用采样到两个路径列表
            self.I_paths_s = [self.I_paths[i] for i in indices]
            self.T_paths_s = [self.T_paths[i] for i in indices]
            self.R_paths_s = [self.R_paths[i] for i in indices]

        else:
            self.I_paths_s,self.T_paths_s,self.R_paths_s=self.I_paths,self.T_paths,self.R_paths

    # --- 工具：RGB 直方图匹配 ---
    def _hist_match_rgb(self, src_img: Image.Image, ref_img: Image.Image, bins: int = 256) -> Image.Image:
        """
        将 src_img 的颜色/亮度对齐到 ref_img：
        - 先做自适应 gamma 矫正（以中位亮度匹配，避免过度受极端值影响）
        - 再逐通道做基于CDF-逆CDF的平滑直方图匹配（构造0~255的LUT，单调且稳定）
        """
        import numpy as np
        from PIL import Image

        # ---------- 1) 自适应 gamma：按中位亮度匹配 ----------
        def _adaptive_gamma_match(src_arr, ref_arr):
            # 线性空间近似：对8bit值归一化到[0,1]
            s = src_arr.astype(np.float32) / 255.0
            r = ref_arr.astype(np.float32) / 255.0
            # ITU-R BT.709 亮度权重
            w = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
            s_lum = np.clip((s * w).sum(axis=2), 1e-6, 1.0)
            r_lum = np.clip((r * w).sum(axis=2), 1e-6, 1.0)

            s_med = float(np.median(s_lum))
            r_med = float(np.median(r_lum))

            # 目标：s_med^gamma ≈ r_med  =>  gamma = log(r_med)/log(s_med)
            if s_med <= 1e-6 or r_med <= 1e-6:
                gamma = 1.0
            else:
                gamma = np.log(r_med) / np.log(s_med)
                # 限幅避免极端拉伸
                gamma = float(np.clip(gamma, 0.5, 2.0))

            s_corr = np.power(s, gamma)
            s_corr = np.clip(np.rint(s_corr * 255.0), 0, 255).astype(np.uint8)
            return s_corr

        # ---------- 2) 稳定的直方图匹配（CDF→逆CDF，单调插值 + 全量LUT） ----------
        def _match_channel_lut(src_c, ref_c, bins=256):
            """
            为单通道构造0~255的映射LUT：
            LUT[v] = inverse_cdf_ref(cdf_src(v))
            通过固定bins的直方图得到平滑CDF，并用插值实现单调映射。
            """
            # 计算直方图（固定边界在[0,256]，避免空洞）
            bin_edges = np.linspace(0, 256, int(bins) + 1, dtype=np.float64)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5

            # 小的平滑项，防止平坦区导致阶跃
            eps = 1e-8

            s_hist, _ = np.histogram(src_c, bins=bin_edges)
            r_hist, _ = np.histogram(ref_c, bins=bin_edges)

            s_cdf = np.cumsum(s_hist.astype(np.float64) + eps)
            s_cdf /= s_cdf[-1]
            r_cdf = np.cumsum(r_hist.astype(np.float64) + eps)
            r_cdf /= r_cdf[-1]

            # 先把所有整数灰度的CDF求出来（在src的CDF上插值）
            vals = np.arange(256, dtype=np.float64)
            s_cdf_at_vals = np.interp(vals, bin_centers, s_cdf)

            # 再通过ref的 逆CDF（cdf->value）求出目标值
            # 注意：x为CDF（单调非降），y为bin_centers
            lut = np.interp(s_cdf_at_vals, r_cdf, bin_centers)

            # 安全裁剪并量化到[0,255]
            lut = np.clip(np.rint(lut), 0, 255).astype(np.uint8)
            return lut

        # ---------- 流程 ----------
        src = np.asarray(src_img.convert('RGB'))
        ref = np.asarray(ref_img.convert('RGB'))

        # 1) 亮度自适应gamma
        src_corr = _adaptive_gamma_match(src, ref)

        # 2) 逐通道稳定直方图匹配（用 LUT 应用，避免像素级插值抖动）
        out = np.empty_like(src_corr)
        for c in range(3):
            lut = _match_channel_lut(src_corr[..., c], ref[..., c], bins=bins)
            out[..., c] = lut[src_corr[..., c]]

        return Image.fromarray(out, mode='RGB')










# 非 -input -label1
# size：限制加载的数据量
class TestDataset(BaseDataset):
    def __init__(self, datadir, fns=None, size=None, enable_transforms=False, unaligned_transforms=False,
                 round_factor=1, flag=None, if_align=True, real=False, HW=[256,256]):
        super(TestDataset, self).__init__()
        self.size = size
        self.datadir = datadir
        self.fns = fns 
        self.enable_transforms = enable_transforms
        self.unaligned_transforms = unaligned_transforms
        self.round_factor = round_factor
        self.flag = flag
        self.if_align = True # if_align
        self.real = real
        self.HW = HW
        
        self.I_paths = []
        self.R_paths = []
        self.T_paths = []



        if self.fns != None : # 测试集验证机训练集已经分在了txt文件中 好处是不用对图片数据集挑挑拣拣
            with open(self.fns, 'r') as f:
                for line in f :
                    if line.strip():
                        file = line.strip()
                        self.I_paths.append(os.path.join(self.datadir, file))


        if size is not None and size<=len(self.I_paths): # 如果有size控制，那截取size个元素,而且是随机截取
            zipped = self.I_paths
            sampled_tuples = random.sample(zipped, size)
            self.I_paths_s = sampled_tuples
        else:
            self.I_paths_s=self.I_paths

    def align(self, x1, x2, x3):
        h, w = self.HW[0], self.HW[1]
        h, w = h // 32 * 32, w // 32 * 32
        # h_new, w = h + (32 - h % 32), w + 
        x1 = x1.resize((w, h))
        x2 = x2.resize((w, h))
        x3 = x3.resize((w, h))
        return x1, x2, x3

    def __getitem__(self, index):
        filename = os.path.basename(self.I_paths_s[index]).replace('[', '').replace(']', '').replace("'", '').replace('.png', '')

        m_img = Image.open(self.I_paths_s[index]).convert('RGB')
        t_img = m_img.copy()  # 如果没有目标图像，则使用混合图像作为目标
        r_img = Image.fromarray(np.clip(np.array(m_img, dtype=np.float32) - np.array(t_img, dtype=np.float32), 0, 255).astype(np.uint8))

        if self.enable_transforms:
            t_img, m_img, r_img = paired_data_transforms(t_img, m_img, r_img, self.unaligned_transforms)

        if self.if_align:
            t_img, m_img, r_img = self.align(t_img, m_img, r_img)

        B = TF.to_tensor(t_img)
        M = TF.to_tensor(m_img)
        R = TF.to_tensor(r_img)

        dic = {'input': M, 'target_t': B, 'fn': filename, 'real': self.real, 'target_r': R}
        if self.flag is not None:
            dic.update(self.flag) # 用于将一个字典（或键值对序列）的内容合并到当前字典中
        return dic

    # 返回数据集的实际长度，受 size 参数限制
    def __len__(self):
        if self.size is not None:
            return min(len(self.I_paths), self.size)
        else:
            return len(self.I_paths)









# ​​数据集混合​​：将多个数据集（datasets）合并为一个虚拟数据集，按指定比例（fusion_ratios）随机选择样本。
# ​​动态采样​​：每次调用 __getitem__ 时，根据比例随机选择一个子数据集，并返回其样本。
# datasets: 子数据集列表（[dataset1, dataset2, ...]）。
# fusion_ratios: 每个数据集的采样比例（如 [0.7, 0.3] 表示70%来自dataset1，30%来自dataset2）。
class FusionDataset(BaseDataset):
    def __init__(self, datasets, fusion_ratios=None):
        self.datasets = datasets

        # datasets = [ds1, ds2, ds3]
        # print(len(datasets))  # 输出: 3（3个子数据集）
        self.fusion_ratios = fusion_ratios or [1. / len(datasets)] * len(datasets) # 否则 均分 对列表进行乘法操作 结果是生成一个包含 ​​3个相同元素​​ 的新列表
        self.size = int(sum([len(dataset)*self.fusion_ratios[i] for i,dataset in enumerate(datasets)]))
        print('[i] using a fusion dataset: %d %s imgs fused with ratio %s' % (
            self.size, [len(dataset) for dataset in datasets], self.fusion_ratios))

    def reset(self):
        for dataset in self.datasets:
            dataset.reset()

    # 采样过程​​：
    # ​​初始化​​：
    # residual = 1
    # ​​第一轮（ds1）​​：
    # 计算 0.5 / 1 = 0.5，随机数若 <0.5 则选择 ds1（50%概率）。
    # 若未选中，residual = 1 - 0.5 = 0.5。
    # ​​第二轮（ds2）​​：
    # 计算 0.3 / 0.5 = 0.6，随机数若 <0.6 则选择 ds2（实际60%概率，但总概率仍为30%）。
    # 若未选中，residual = 0.5 - 0.3 = 0.2。
    # ​​第三轮（ds3）​​：
    # 计算 0.2 / 0.2 = 1，随机数必 <1，因此100%选择 ds3（总概率20%）。
    # ​​或​​：直接因 i == 2 强制选中。
    
    # 按概率分配并随机选取的实现原理：
    # 每次取数据时 都调用了getitem 然后触发三循环 三个都是有概率取到 如果排在前的先取了 就会return 下一个就得等下一次getitem
    def __getitem__(self, index):
        residual = 1
        for i, ratio in enumerate(self.fusion_ratios):
            if random.random() < ratio / residual or i == len(self.fusion_ratios) - 1:
                dataset = self.datasets[i]
                # len(dataset)​​：当前子数据集的样本总数 将 index ​​取模​​处理，确保索引不超过子数据集的实际大小（防止越界）
                return dataset[index % len(dataset)] # 取子数据集dataset是通过子数据集实现的__getitem方法
            residual -= ratio


    def __len__(self):
        return self.size









import json
from torch.utils.data import Dataset


class VOCJsonDataset(Dataset):
    def __init__(self, datadir, json_file, size=None, enable_transforms=False, 
                 unaligned_transforms=False, round_factor=32, flag=None, 
                 if_align=True, real=False, HW=[256, 256]):
        """
        datadir: 图像根目录 (/home/gzm/gzm-RDNet1/dataset/VOC2012)
        json_file: VOC_results_list.json
        size: 采样的样本数 (None 表示全部)
        enable_transforms: 是否做数据增强
        unaligned_transforms: 是否做不对齐的增强 (例如随机裁剪)
        round_factor: 调整尺寸时按多少取整 (默认32)
        flag: 额外的标记字典
        if_align: 是否强制对齐图像大小
        real: 是否为真实数据 (调试用)
        HW: 输出图像的目标高宽
        """
        super().__init__()
        self.datadir = datadir
        self.size = size
        self.enable_transforms = enable_transforms
        self.unaligned_transforms = unaligned_transforms
        self.round_factor = round_factor
        self.flag = flag
        self.if_align = if_align
        self.real = real
        self.HW = HW

        # 读取 JSON
        with open(json_file, "r") as f:
            self.samples = json.load(f)

        # 采样
        if size == 0:
            self.samples = []
        elif size is not None and size > 0 and size <= len(self.samples):
            self.samples = random.sample(self.samples, size)

    def align(self, *imgs):
        """统一图像大小"""
        h, w = self.HW
        # 保证尺寸是 round_factor 的倍数
        h = h // self.round_factor * self.round_factor
        w = w // self.round_factor * self.round_factor
        return [img.resize((w, h)) for img in imgs]

    def __getitem__(self, index):
        sample = self.samples[index]
        filename = os.path.basename(sample["blended"]).replace(".png", "")

        # 路径拼接
        transmission_path = os.path.join(self.datadir, "transmission_layer", sample["transmission_layer"])
        reflection_path = os.path.join(self.datadir, "reflection_layer", sample["reflection_layer"])
        blended_path = os.path.join(self.datadir, "blended", sample["blended"])

        # 打开图片
        t_img = Image.open(transmission_path).convert("RGB")
        r_img = Image.open(reflection_path).convert("RGB")
        b_img = Image.open(blended_path).convert("RGB")

        # 数据增强 (这里留接口给你)
        if self.enable_transforms:
            t_img, b_img, r_img = paired_data_transforms(t_img, b_img, r_img, self.unaligned_transforms)

        # 尺寸对齐
        if self.if_align:
            t_img, r_img, b_img = self.align(t_img, r_img, b_img)

        # 转 Tensor
        T_tensor = TF.to_tensor(t_img)
        R_tensor = TF.to_tensor(r_img)
        B_tensor = TF.to_tensor(b_img)

        dic = {'input': B_tensor, 'target_t': T_tensor, 'fn': filename, 'real': self.real, 'target_r': R_tensor}

        return dic
    

    def __len__(self):
        return len(self.samples)





class HyperKDataset(Dataset):
    def __init__(self, root="./EndoData", json_path=None, start=343, end=372, size=None,
                 enable_transforms=False, unaligned_transforms=False,
                 if_align=True, HW=[256,256], flag=None, color_jitter=False):
        super(HyperKDataset, self).__init__()
        self.root = root
        self.start = start
        self.end = end
        self.size = size
        self.enable_transforms = enable_transforms
        self.unaligned_transforms = unaligned_transforms
        self.if_align = if_align
        self.HW = HW
        self.flag = flag
        self.real = True
        self.color_jitter = color_jitter

        self.I_paths = []  # 输入
        self.T_paths = []  # 标签

        # 读取 test.json
        json_path = json_path
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"❌ 找不到 {json_path}")
        with open(json_path, "r") as f:
            limits = json.load(f)  # dict, e.g. {"hyperK_343": 1982, ...}

        # 遍历编号
        for idx in range(start, end + 1):
            folder_name = f"hyperK_{idx:03d}"
            input_dir = os.path.join(root, folder_name, "input")
            label_dir = os.path.join(root, folder_name, "label")

            if not os.path.exists(input_dir) or not os.path.exists(label_dir):
                print(input_dir)
                print(f"⚠️ 跳过 {folder_name}, input 或 label 不存在")
                continue

            # 限制数量
            limit = limits.get(folder_name, None)
            input_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(".png")])
            if limit is not None:
                input_files = input_files[:limit]

            for fname in input_files:
                input_path = os.path.join(input_dir, fname)
                label_path = os.path.join(label_dir, fname)
                if os.path.exists(label_path):
                    self.I_paths.append(input_path)
                    self.T_paths.append(label_path)
                else:
                    print(f"⚠️ 没找到匹配的标签: {label_path}")

        self.SampleNewItems()

    def align(self, x1, x2, x3):
        h, w = self.HW
        h, w = h // 32 * 32, w // 32 * 32
        x1 = x1.resize((w, h))
        x2 = x2.resize((w, h))
        x3 = x3.resize((w, h))
        return x1, x2, x3

    def __getitem__(self, index):
        filename = os.path.basename(self.I_paths_S[index]).replace(".png", "")

        m_img = Image.open(self.I_paths_S[index]).convert("RGB")
        t_img = Image.open(self.T_paths_S[index]).convert("RGB")
        r_img = Image.fromarray(np.clip(np.array(m_img, dtype=np.float32) - np.array(t_img, dtype=np.float32), 0, 255).astype(np.uint8))

        if self.enable_transforms:
            t_img, m_img, r_img = paired_data_transforms(t_img, m_img, r_img, self.unaligned_transforms)

        if self.if_align:
            t_img, m_img, r_img = self.align(t_img, m_img, r_img)

        if self.color_jitter:
            # 颜色扰动：默认只对 m_img 生效，并重算 r_img
            m_img, t_img, r_img = _maybe_color_jitter(
                m_img, t_img, r_img,
                p=0.8, brightness=0.2, contrast=0.2, saturation=0.3, hue=0.05,
                apply_to_label=True, recompute_residual=False
            )        

        T = TF.to_tensor(t_img)
        M = TF.to_tensor(m_img)
        R = TF.to_tensor(r_img)

        dic = {'input': M, 'target_t': T, 'fn': filename, 'real': self.real, 'target_r': R, 'Endo':True}

        return dic

    def __len__(self):
        return len(self.I_paths_S)
    
    def SampleNewItems(self):
        '''
            每几个epoches从全部数据库重新采样指定数量的数据 为了数据多样性
        '''
        # 随机抽样
        if self.size == 0:
            self.I_paths_S, self.T_paths_S = [], []
            pass

        elif self.size is not None and self.size > 0 and self.size <= len(self.I_paths):
            total_length = len(self.I_paths)
            step = total_length // self.size
            
            # 在0到step-1之间随机选择起始点
            start_index = np.random.randint(0, step)
            
            # 生成等距采样索引
            indices = [start_index + i * step for i in range(self.size)]
            
            # 确保索引不越界
            indices = [i if i < total_length else i - total_length for i in indices]
            
            # 应用采样到两个路径列表
            self.I_paths_S = [self.I_paths[i] for i in indices]
            self.T_paths_S = [self.T_paths[i] for i in indices]

        else:
            self.I_paths_S= self.I_paths
            self.T_paths_S= self.T_paths














class HyperKDataset_Test(Dataset):
    def __init__(self, root="./EndoData", json_path=None, start=343, end=372, size=None,
                 enable_transforms=False, unaligned_transforms=False,
                 if_align=True, HW=[256,256], flag=None, color_jitter=False):
        # 修正父类初始化，避免 super(type, obj) 传入不相关类型导致的 TypeError
        super(HyperKDataset_Test, self).__init__()
        self.root = root
        self.start = start
        self.end = end
        self.size = size
        self.enable_transforms = enable_transforms
        self.unaligned_transforms = unaligned_transforms
        self.if_align = if_align
        self.HW = HW
        self.flag = flag
        self.real = True
        self.color_jitter = color_jitter

        self.I_paths = []  # 输入

        # 读取 test.json
        json_path = json_path
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"❌ 找不到 {json_path}")
        with open(json_path, "r") as f:
            limits = json.load(f)  # dict, e.g. {"hyperK_343": 1982, ...}

        # 遍历编号
        for idx in range(start, end + 1):
            folder_name = f"hyperK_{idx:03d}"
            input_dir = os.path.join(root, folder_name)

            if not os.path.exists(input_dir):
                print(f"⚠️ 跳过 {folder_name} 不存在")
                continue

            # 限制数量
            limit = limits.get(folder_name, None)
            input_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(".jpg")])
            if limit is not None:
                input_files = input_files[:limit]

            for fname in input_files:
                input_path = os.path.join(input_dir, fname)
                if os.path.exists(input_path):
                    self.I_paths.append(input_path)
                else:
                    print(f"⚠️ input未找到: {input_path}")

        self.SampleNewItems()


    def align(self, x1, x2, x3):
        h, w = self.HW
        h, w = h // 32 * 32, w // 32 * 32
        x1 = x1.resize((w, h))
        x2 = x2.resize((w, h))
        x3 = x3.resize((w, h))
        return x1, x2, x3

    def __getitem__(self, index):
        filename = os.path.basename(self.self.I_paths_S[index]).replace(".png", "")

        m_img = Image.open(self.self.I_paths_S[index]).convert("RGB")   

        m_img = m_img.resize((256,256))   

        M = TF.to_tensor(m_img)

        dic = {'input': M, 'fn': filename, 'real': self.real}

        return dic

    def __len__(self):
        return len(self.self.I_paths_S)
    
    
    def SampleNewItems(self):
        '''
            每几个epoches从全部数据库重新采样指定数量的数据 为了数据多样性
        '''
        # 随机抽样
        if self.size == 0:
            self.I_paths_S = []
            pass

        elif self.size is not None and self.size > 0 and self.size <= len(self.I_paths):
            # 随机开头的等距抽样 保证整体覆盖性
            total_length = len(self.I_paths)
            step = total_length // self.size
            start_index = np.random.randint(0, step)
            indices = [(start_index + i * step) % total_length for i in range(self.size)]
            self.I_paths_S = [self.I_paths[i] for i in indices]

        else:
            self.I_paths_S= self.I_paths



def _maybe_color_jitter(m_img, t_img, r_img,
                        p=0.8,
                        brightness=0.2, contrast=0.2, saturation=0.3, hue=0.05,
                        apply_to_label=False,   # 一般 False：不改 label
                        recompute_residual=True # True：扰动后重算 r=m-t
                        ):
    """对输入图像做颜色扰动：亮度/对比度/饱和度/色调。
    默认只扰动 m_img；若 apply_to_label=True，则 t_img 同步扰动。
    """
    if random.random() >= p:
        return m_img, t_img, r_img

    # 采样一次固定的扰动参数
    jitter = T.ColorJitter(brightness=brightness, contrast=contrast,
                        saturation=saturation, hue=hue)
    fn_idx, b, c, s, h = T.ColorJitter.get_params(
        jitter.brightness, jitter.contrast, jitter.saturation, jitter.hue
    )

    def apply_one(img):
        # 按随机顺序依次应用 4 个操作
        for fn_id in fn_idx:
            if fn_id == 0 and b is not None:
                img = TF.adjust_brightness(img, b)
            elif fn_id == 1 and c is not None:
                img = TF.adjust_contrast(img, c)
            elif fn_id == 2 and s is not None:
                img = TF.adjust_saturation(img, s)
            elif fn_id == 3 and h is not None:
                img = TF.adjust_hue(img, h)
        return img

    # 只扰动 input
    m_img = apply_one(m_img)

    # （可选）同步扰动 label
    if apply_to_label:
        t_img = apply_one(t_img)

    # （可选）重算 residual，保持 r=m-t 一致
    if recompute_residual:
        m_np = np.array(m_img, dtype=np.float32)
        t_np = np.array(t_img, dtype=np.float32)
        r_np = np.clip(m_np - t_np, 0, 255).astype(np.uint8)
        r_img = Image.fromarray(r_np)

    return m_img, t_img, r_img


    