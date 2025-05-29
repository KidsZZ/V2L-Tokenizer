"""
V2L-Tokenizer 训练脚本
====================

这个脚本用于训练 Vision-to-Language (V2L) Tokenizer 模型。
V2L-Tokenizer 是一个将视觉信息转换为语言token的模型，主要用于视觉-语言多模态任务。

主要功能：
1. 加载和预处理 ImageNet 数据集
2. 训练 VQModel_LLaMA 模型（结合了 VQGAN 和 LLaMA 的架构）
3. 使用对抗训练（GAN Loss）优化模型
4. 支持分布式训练
5. 定期保存检查点和训练日志

模型架构：
- 编码器：将图像编码为潜在特征
- 量化层：将连续特征离散化为token
- 解码器：从token重建图像
- 判别器：用于对抗训练，提高生成质量
"""

import argparse
import datetime
import json
import os
import time
from pathlib import Path
import albumentations  # 图像增强库
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter  # 训练日志记录
from PIL import Image
import yaml
import torch
import clip  # OpenAI CLIP 模型
from omegaconf import OmegaConf  # 配置文件管理

from llama_inference.llama import Tokenizer
from models.models_v2l import VQModel_LLaMA 
from training.engine_training import train_one_epoch
import util.misc as misc

from util.misc import NativeScalerWithGradNormCount as NativeScaler

# 设备配置：优先使用GPU，否则使用CPU
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_config(config_path, display=False):
    """
    加载YAML配置文件
    
    Args:
        config_path (str): 配置文件路径
        display (bool): 是否打印配置内容
    
    Returns:
        OmegaConf: 配置对象
    """
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config

def load_vqgan(config, ckpt_path=None, is_gumbel=False):
    """
    加载预训练的VQGAN模型
    
    Args:
        config: 模型配置
        ckpt_path (str, optional): 检查点路径
        is_gumbel (bool): 是否使用Gumbel量化
    
    Returns:
        torch.nn.Module: 加载的VQGAN模型
    """
    model = VQModel(**config.model.params)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
    return model.eval()

def preprocess_vqgan(x):
    """
    VQGAN预处理：将像素值从[0,1]范围转换到[-1,1]范围
    
    Args:
        x (torch.Tensor): 输入图像张量
    
    Returns:
        torch.Tensor: 预处理后的图像张量
    """
    x = 2.*x - 1.
    return x


class ImageNetDataset(Dataset):
    """
    ImageNet数据集类
    
    用于加载和预处理ImageNet图像数据，支持CLIP预处理和多尺度处理
    """
    
    def __init__(self, data_root, image_size, max_words=30, n_class=1000, partition="train", device="cpu"):
        """
        初始化ImageNet数据集
        
        Args:
            data_root (str): 数据根目录路径
            image_size (int): 目标图像尺寸
            max_words (int): 最大词数限制
            n_class (int): 类别数量限制
            partition (str): 数据集分割（train/val）
            device (str): 计算设备
        """
        self.max_words = max_words
        self.device = device
        self.image_size = image_size
        self.data_root = data_root
        
        # 加载CLIP模型的预处理管道
        _, _, self.preprocess = clip.create_model_and_transforms(
            'ViT-L-14', 
            pretrained='laion2b_s32b_b82k', 
            device=device,
            cache_dir="/root/autodl-tmp/downloads"
        )

        # 图像预处理管道：缩放到最小边128像素，然后随机裁剪128x128
        self.rescaler = albumentations.SmallestMaxSize(max_size=128)
        self.cropper = albumentations.RandomCrop(height=128, width=128)
        self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])

        # 不同层级的token数量配置
        self.token_nums = [1, 4, 16, 64, 256, 256]
        self.partition = partition

        # CLIP模型的标准化参数（均值和标准差）
        self.clip_mean = torch.from_numpy(np.array([0.48145466, 0.4578275, 0.40821073])).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).float()
        self.clip_std = torch.from_numpy(np.array([0.26862954, 0.26130258, 0.27577711])).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).float()

        # 加载图像ID和类别标签
        self.image_ids = []
        self.class_labels = []
        with open("imagenet_split/" + partition + "/class_labels.txt") as f:
            for line in f.readlines():
                image_id, class_label = line.strip('\n').split(",")
                if int(class_label) < n_class:  # 只选择指定数量的类别
                    if partition == "train":
                        self.image_ids.append(image_id)
                    elif partition == "val":
                        self.image_ids.append(image_id)
                    self.class_labels.append(int(class_label))

    def __len__(self):
        """返回数据集大小"""
        return len(self.image_ids)

    def __getitem__(self, index):
        """
        获取单个数据样本
        
        Args:
            index (int): 样本索引
        
        Returns:
            list: [图像ID, 处理后的图像, CLIP处理的图像, 类别标签]
        """
        image_ids = self.image_ids[index]
        
        # 加载图像
        image = Image.open(os.path.join(self.data_root, image_ids))
        
        # CLIP预处理
        clip_image = self.clip_preprocessing(image)
        label = self.class_labels[index]

        # 将图像调整到128x128尺寸，并进行标准化处理
        input = torch.nn.functional.interpolate(
            clip_image.unsqueeze(0), 
            size=(128, 128), 
            mode='bilinear', 
            align_corners=False
        ).contiguous()
        
        # 反标准化：从CLIP标准化空间转回原始像素空间
        input = self.clip_std * input + self.clip_mean
        # 转换到[-1, 1]范围（VQGAN的输入要求）
        input = 2 * input - 1
        input = input.squeeze(0)

        return [image_ids, input, clip_image, label]

def get_args_parser():
    """
    创建命令行参数解析器
    
    Returns:
        argparse.ArgumentParser: 配置好的参数解析器
    """
    parser = argparse.ArgumentParser("MAE pre-training", add_help=False)
    
    # 训练相关参数
    parser.add_argument("--batch_size", default=64, type=int,
                       help="每个GPU的批次大小（有效批次大小 = batch_size * accum_iter * GPU数量）")
    parser.add_argument("--epochs", default=400, type=int, help="训练轮数")
    parser.add_argument("--accum_iter", default=1, type=int,
                       help="梯度累积迭代次数（用于在内存限制下增加有效批次大小）")

    # 模型参数
    parser.add_argument("--max_seq_len", type=int, default=512, metavar="LENGTH", 
                       help="最大序列长度")

    # 优化器参数
    parser.add_argument("--weight_decay", type=float, default=0.05, 
                       help="权重衰减系数（默认：0.05）")
    parser.add_argument("--lr", type=float, default=None, metavar="LR", 
                       help="学习率（绝对学习率）")
    parser.add_argument("--blr", type=float, default=1e-3, metavar="LR",
                       help="基础学习率：绝对学习率 = 基础学习率 * 总批次大小 / 256")
    parser.add_argument("--min_lr", type=float, default=0.0, metavar="LR", 
                       help="循环调度器的最低学习率")
    parser.add_argument("--warmup_epochs", type=int, default=40, metavar="N", 
                       help="学习率预热轮数")

    # 数据集和输出参数
    parser.add_argument("--output_dir", default="./output_dir", 
                       help="模型保存路径，空字符串表示不保存")
    parser.add_argument("--log_dir", default="./output_dir", 
                       help="TensorBoard日志保存路径")
    parser.add_argument("--device", default="cuda", 
                       help="训练/测试使用的设备")
    parser.add_argument("--seed", default=0, type=int, help="随机种子")
    parser.add_argument("--resume", default="", help="从检查点恢复训练")
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", 
                       help="开始训练的轮数")
    parser.add_argument("--num_workers", default=0, type=int, 
                       help="数据加载器的工作进程数")
    parser.add_argument("--pin_mem", action="store_true",
                       help="在DataLoader中固定CPU内存，提高GPU传输效率")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # 分布式训练参数
    parser.add_argument("--world_size", default=1, type=int, 
                       help="分布式进程总数")
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://", 
                       help="分布式训练初始化URL")

    # 模型特定参数
    parser.add_argument("--imagenet_path", default="", type=str, 
                       help="ImageNet数据集路径")
    parser.add_argument("--vqgan_path", default="vqgan_weight/vqgan_imagenet_f16_16384", 
                       type=str, help="VQGAN模型权重路径")
    parser.add_argument("--n_vision_words", default=32000, type=int, 
                       help="视觉词汇表大小")
    parser.add_argument("--output_type", default="next_token_prediction", type=str, 
                       help="输出类型：next_token_prediction/classification")
    parser.add_argument("--decode_rate", type=float, default=0, 
                       help="解码损失权重")
    parser.add_argument("--n_class", default=1000, type=int, 
                       help="分类类别数")
    parser.add_argument("--vq_config_path", type=str, default="vqgan_configs/v2l.yaml", 
                       help="VQ模型配置文件路径")
    parser.add_argument("--image_size", type=int, default=256, 
                       help="输入图像尺寸")
    parser.add_argument("--stage", type=int, default=1, 
                       help="训练阶段")
    parser.add_argument("--quantizer_type", type=str, default="org", 
                       help="量化器类型")

    # 模型架构参数
    parser.add_argument("--embed_dim", type=int, default=768, 
                       help="嵌入维度")
    parser.add_argument("--tuning_codebook", type=int, default=1, 
                       help="是否微调codebook")
    parser.add_argument("--use_cblinear", type=int, default=0, 
                       help="是否使用codebook线性投影")
    parser.add_argument("--use_crossatt_dec", type=int, default=0, 
                       help="是否使用交叉注意力解码器")

    # 预训练权重路径
    parser.add_argument("--local_embedding_path", default="local_codebook_embedding.pth", 
                       type=str, help="局部codebook嵌入路径")
    parser.add_argument("--global_embedding_path", default="global_codebook_embedding.pth", 
                       type=str, help="全局codebook嵌入路径")
    
    # GAN训练参数
    parser.add_argument("--disc_start", default=10000, type=int, 
                       help="判别器开始训练的步数")
    parser.add_argument("--rate_q", type=float, default=1, 
                       help="量化损失权重")
    parser.add_argument("--rate_p", type=float, default=1, 
                       help="感知损失权重")
    parser.add_argument("--rate_d", type=float, default=0.75, 
                       help="对抗损失权重")

    return parser


def main(args):
    """
    主训练函数
    
    Args:
        args: 命令行参数对象
    """
    # 创建输出和日志目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # 初始化分布式训练
    misc.init_distributed_mode(args)

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    # 设备和随机种子配置
    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()  # 确保不同进程使用不同种子
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True  # 优化cuDNN性能

    # 创建训练和验证数据集
    dataset_train = ImageNetDataset(
        data_root=args.imagenet_path, 
        image_size=args.image_size, 
        max_words=args.max_seq_len, 
        n_class=args.n_class, 
        partition="train", 
        device=device
    )
    dataset_val = ImageNetDataset(
        data_root=args.imagenet_path, 
        image_size=args.image_size, 
        max_words=args.max_seq_len, 
        n_class=args.n_class, 
        partition="val", 
        device=device
    )

    # 配置分布式数据采样器
    if True:  # args.distributed:
        num_tasks = misc.get_world_size()  # 总进程数
        global_rank = misc.get_rank()      # 当前进程排名
        
        # 训练集采样器：随机采样，确保每个进程获得不同的数据子集
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, drop_last=True
        )

        # 验证集采样器：同样需要分布式采样
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True, drop_last=False
        )

        print("Sampler_train = %s" % str(sampler_train))
    else:
        # 非分布式训练使用随机采样器
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    # 配置TensorBoard日志记录器（仅在主进程中创建）
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # 创建数据加载器
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,  # 丢弃最后一个不完整的批次
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # 加载模型配置并创建模型
    config = load_config(args.vq_config_path, display=True)

    # 创建V2L模型（结合VQGAN和LLaMA的架构）
    model = VQModel_LLaMA(args=args, **config.model.params)
    model.to(device)
    model_without_ddp = model
    
    # 冻结全局编码器参数（不参与训练）
    for param in model.global_encoder.parameters():
        param.requires_grad = False

    # 计算有效批次大小
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # 配置分布式并行训练
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # 配置优化器
    # 自编码器优化器：包含编码器、解码器、token嵌入等组件
    if args.use_cblinear == 1:
        # 如果使用codebook线性投影，将其参数也加入优化
        opt_ae = torch.optim.Adam(
            list(model_without_ddp.encoder.parameters()) +
            list(model_without_ddp.decoder.parameters()) +
            list(model_without_ddp.tok_embeddings.parameters()) +
            list(model_without_ddp.quant_conv.parameters()) +
            list(model_without_ddp.codebook_projection.parameters()) + 
            list(model_without_ddp.post_quant_conv.parameters()), 
            lr=args.lr, betas=(0.5, 0.9), eps=1e-7
        )
    else:
        # 标准自编码器优化器配置
        opt_ae = torch.optim.Adam(
            list(model_without_ddp.encoder.parameters()) +
            list(model_without_ddp.decoder.parameters()) +
            list(model_without_ddp.tok_embeddings.parameters()) +
            list(model_without_ddp.quant_conv.parameters()) +
            list(model_without_ddp.post_quant_conv.parameters()), 
            lr=args.lr, betas=(0.5, 0.9), eps=1e-7
        )
    
    # 判别器优化器：用于对抗训练
    opt_dist = torch.optim.Adam(
        model_without_ddp.discriminator.parameters(), 
        lr=args.lr, betas=(0.5, 0.9), eps=1e-7
    )

    # 梯度缩放器：用于混合精度训练，防止梯度下溢
    loss_scaler_ae = NativeScaler()      # 自编码器梯度缩放器
    loss_scaler_disc = NativeScaler()    # 判别器梯度缩放器

    # 自动恢复训练（如果存在最新检查点）
    if os.path.exists(os.path.join(args.output_dir, 'vqgan_checkpoint-last.pth')):
        ckpt = torch.load(os.path.join(args.output_dir, 'vqgan_checkpoint-last.pth'), map_location="cpu")
        model_without_ddp.load_state_dict(ckpt["model"], strict=True)
        opt_ae.load_state_dict(ckpt["opt_ae"])
        opt_dist.load_state_dict(ckpt["opt_dist"])
        loss_scaler_ae.load_state_dict(ckpt["scaler_ae"])
        loss_scaler_disc.load_state_dict(ckpt["scaler_dist"])
        args = ckpt["args"]
        args.start_epoch = ckpt["epoch"] + 1
        print(args)
        print("*********Resuming From Epoch %d********"%(args.start_epoch))

    # 将优化器和缩放器组织成列表，便于传递给训练函数
    optimizer = [opt_ae, opt_dist]
    loss_scaler = [loss_scaler_ae, loss_scaler_disc]

    num_val_images = len(dataset_val.image_ids)

    # 开始训练循环
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    
    for epoch in range(args.start_epoch, args.epochs):
        # 在分布式训练中，每个epoch都需要设置不同的随机种子
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
            data_loader_val.sampler.set_epoch(epoch)

        # 执行一个epoch的训练
        train_stats = train_one_epoch(
            model, data_loader_train, optimizer, device, epoch, loss_scaler, 
            log_writer=log_writer, args=args
        )
        
        # 保存最新的检查点（每个epoch都保存，用于恢复训练）
        misc.save_model_last_vqgan_ganloss(
            args=args,
            model=model,
            model_without_ddp=model_without_ddp,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            epoch=epoch,
        )
        
        # 定期保存检查点（每10个epoch或最后一个epoch）
        if args.output_dir and (epoch % 10 == 0 or epoch + 1 == args.epochs):
            misc.save_model_vqgan(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
            )

        # 记录训练统计信息
        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch
        }

        # 保存训练日志（仅在主进程中执行）
        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()  # 确保日志写入磁盘
            # 将训练统计信息写入日志文件
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    # 训练完成，计算总训练时间
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))

if __name__ == "__main__":
    # 程序入口点
    args = get_args_parser()
    args = args.parse_args()
    
    # 确保输出目录存在
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 开始训练
    main(args)

