from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List, Optional

class HuggingFaceLLM:
    def __init__(self, model_name: str, device: str = "cuda"):
        print(f"Loading tokenizer from {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        
        # 添加 pad_token 如果不存在
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"GPU Memory before loading model: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        print(f"Loading model from {model_name}")
        # 修改模型加载配置，避免分布式张量问题
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            local_files_only=True,
            torch_dtype=torch.float16,  # 使用 half precision
            low_cpu_mem_usage=True,
            device_map=device,  # 直接指定设备而不是"auto"
            max_memory=None,  # 移除内存限制
            trust_remote_code=True
        )
        
        print(f"GPU Memory after loading model: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        self.device = device

    def decode(self, token_id: int) -> str:
        """解码单个token ID为字符串"""
        return self.tokenizer.decode([token_id], skip_special_tokens=True)

    def text_completion(
        self,
        prompts: List[str],
        temperature: float,
        top_p: float,
        max_gen_len: Optional[int] = 128,
        logprobs: bool = False,
        echo: bool = False,
    ):
        generations = []
        
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            # 确保输入在正确的设备上
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 生成更多的token以确保有足够的内容来截断到指定单词数
            estimated_tokens = max_gen_len * 3 if max_gen_len else 256
            
            with torch.no_grad():  # 减少内存使用
                # 修复生成参数的条件判断
                if temperature > 0:
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=estimated_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        use_cache=True
                    )
                else:
                    # 当temperature为0时，使用贪心搜索，不需要temperature和top_p参数
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=estimated_tokens,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        use_cache=True
                    )
            
            generated = output[0]
            if not echo:
                # 只取新生成的部分
                generated = generated[inputs["input_ids"].shape[1]:]
            
            decoded = self.tokenizer.decode(generated, skip_special_tokens=True)
            
            # 按单词截断到指定长度
            if max_gen_len:
                words = decoded.split()
                if len(words) > max_gen_len:
                    words = words[:max_gen_len]
                decoded = ' '.join(words)
            
            generations.append({"generation": decoded})
            
            # 清理中间变量
            del inputs, output, generated
            torch.cuda.empty_cache()
        
        return generations
