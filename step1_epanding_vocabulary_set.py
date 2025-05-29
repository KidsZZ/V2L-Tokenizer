import argparse
import os
from pathlib import Path
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml
import torch
from omegaconf import OmegaConf
import open_clip as clip
import time

##############
from testfold.testllm import HuggingFaceLLM
import util.misc as misc


def load_config(config_path, display=False):
  config = OmegaConf.load(config_path)
  if display:
    print(yaml.dump(OmegaConf.to_container(config)))
  return config

def get_args_parser():
    parser = argparse.ArgumentParser("MAE pre-training", add_help=False)
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    # Model parameters
    parser.add_argument("--llama_model_path", default="./llama", type=str, help="path of llama model")
    parser.add_argument("--model", default="llama7B", type=str, metavar="MODEL", help="Name of model to train")
    parser.add_argument("--max_seq_len", type=int, default=2048, metavar="LENGTH", help="the maximum sequence length")

    # Dataset parameters
    parser.add_argument("--output_dir", default="./output_dir", help="path where to save, empty for no saving")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")

    return parser


def main(args):

    texts = []
    misc.init_distributed_mode(args)

    print("=" * 60)
    print("🚀 开始词汇表扩展程序")
    print("=" * 60)
    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)
    print(f"📱 使用设备: {device}")
    
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"🎲 设置随机种子: {seed}")

    cudnn.benchmark = True
    

    ###Load CLIP
    model, _, preprocess = clip.create_model_and_transforms('ViT-L-14',pretrained='laion2b_s32b_b82k',device=device,cache_dir="/root/autodl-tmp/downloads")
    model.to(device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])#, find_unused_parameters=True)

    ###Load LLM using HuggingFace
    print("\n🤖 正在加载LLM模型...")
    print(f"📂 模型路径: {args.llama_model_path}")
    
    try:
        start_time = time.time()
        generator = HuggingFaceLLM(model_name=args.llama_model_path, device=device)
        load_time = time.time() - start_time
        print(f"✅ LLM模型加载成功! 耗时: {load_time:.2f}秒")
        print(f"📊 词汇表大小: {generator.tokenizer.vocab_size}")
    except Exception as e:
        print(f"❌ LLM模型加载失败: {e}")
        return
    
    texts = []
    local_vocabularies = []

    print(f"\n🔄 开始处理词汇表 (总共 {generator.tokenizer.vocab_size} 个词)")
    print("=" * 60)

    ###Iterating each Subwords
    total_tokens = generator.tokenizer.vocab_size
    start_process_time = time.time()
    
    for i in range(0, total_tokens):
        try:
            # 进度显示
            if i % 100 == 0 or i < 10:
                progress = (i / total_tokens) * 100
                elapsed_time = time.time() - start_process_time
                if i > 0:
                    avg_time_per_token = elapsed_time / i
                    eta = avg_time_per_token * (total_tokens - i)
                    print(f"🔍 处理进度: {i}/{total_tokens} ({progress:.1f}%) | "
                          f"已用时: {elapsed_time:.1f}s | 预计剩余: {eta:.1f}s")
            
            # 解码当前token
            cur_token = generator.decode(i)
            
            # 严格过滤：只保留有意义的显式字符输出
            def should_skip_token(token):
                if not token:
                    return True
                    
                # 跳过特殊token
                if token in generator.tokenizer.all_special_tokens:
                    return True
                
                # 跳过只包含空白字符的token
                if not token.strip():
                    return True
                    
                # 跳过包含控制字符的token
                if any(ord(c) < 32 for c in token if c not in [' ']):
                    return True
                    
                # 跳过只包含非字母数字字符的token（除了一些常见的有意义符号）
                meaningful_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                if not any(c in meaningful_chars for c in token):
                    return True
                    
                return False
            
            if should_skip_token(cur_token):
                if i < 10:
                    print(f"⚠️  Token {i} 被过滤，跳过处理: {repr(cur_token)}")
                continue

            local_vocabularies.append(cur_token)
            
            if i < 5:  # 只打印前5个token的详细信息
                print(f"  📝 Token {i}: '{cur_token}' (长度: {len(cur_token)})")

            ###Generating Bigrams
            prompts = ["a photo of %s" % str(cur_token)]

            if i < 200:
                print(f"  🔮 生成二元组，提示词: {prompts[0]}")
            
            results = generator.text_completion(
                prompts,
                max_gen_len=1,
                temperature=0,
                top_p=1.0,
            )

            # 添加结果检查
            if not results or len(results) == 0 or 'generation' not in results[0]:
                if i < 10:
                    print(f"⚠️  Token {i} 二元组生成失败或结果为空")
                continue
            
            bigram_result = results[0]['generation']
                
            if i < 200:
                print(f"  📈 二元组结果: '{bigram_result}'")

            ###Generating Trigrams
            trigram_prompt = "a photo of %s" % str(cur_token + bigram_result)
            prompts = [trigram_prompt]
            
            if i < 200:
                print(f"  🔮 生成三元组，提示词: {trigram_prompt}")
            
            results_2 = generator.text_completion(
                prompts,
                max_gen_len=1,
                temperature=0,
                top_p=1.0,
            )
            
            if not results_2 or len(results_2) == 0 or 'generation' not in results_2[0]:
                if i < 10:
                    print(f"⚠️  Token {i} 三元组生成失败")
                continue
                
            trigram_result = results_2[0]['generation']
            if i < 200:
                print(f"  📊 三元组结果: '{trigram_result}'")
            
            ###Save Vocabulary
            cur_cell = {"1": cur_token, "2": bigram_result, "3": trigram_result}
            texts.append(cur_cell)
            
            if i < 3:
                print(f"  💾 保存词汇: {cur_cell}")
                print("  " + "-" * 50)
                
        except Exception as e:
            print(f"❌ 处理Token {i} 时发生错误: {e}")
            import traceback
            traceback.print_exc()  # 打印详细错误信息
            return

    total_process_time = time.time() - start_process_time
    print(f"\n✅ 词汇表处理完成!")
    print(f"📊 统计信息:")
    print(f"  - 总处理数量: {len(texts)}")
    print(f"  - 本地词汇数量: {len(local_vocabularies)}")
    print(f"  - 总耗时: {total_process_time:.2f}秒")
    print(f"  - 平均每词耗时: {total_process_time/len(texts):.3f}秒")

    print(f"\n💾 正在保存结果...")
    
    try:
        ##Global Vocabulary
        global_vocab_path = "Subword_Bigram_Trigram_Vocabulary.npy"
        np.save(global_vocab_path, texts)
        print(f"✅ 全局词汇表已保存: {global_vocab_path}")
        print(f"  - 文件大小: {os.path.getsize(global_vocab_path)/1024/1024:.2f} MB")

        ##Local Vocabulary
        local_vocab_path = "local_vocabulary.npy"
        np.save(local_vocab_path, local_vocabularies)
        print(f"✅ 本地词汇表已保存: {local_vocab_path}")
        print(f"  - 文件大小: {os.path.getsize(local_vocab_path)/1024/1024:.2f} MB")
        
    except Exception as e:
        print(f"❌ 保存文件时发生错误: {e}")
        
    print("\n" + "=" * 60)
    print("🎉 程序执行完成!")
    print("=" * 60)


if __name__ == "__main__":

    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
