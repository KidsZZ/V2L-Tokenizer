from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List, Optional

class HuggingFaceLLM:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.device = device

    def text_completion(
        self,
        prompts: List[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = 128,
        logprobs: bool = False,  # transformers 不直接支持 logprobs 输出
        echo: bool = False,
    ):
        generations = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            output = self.model.generate(
                **inputs,
                max_length=inputs["input_ids"].shape[1] + max_gen_len,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=logprobs
            )
            generated = output.sequences[0]
            decoded = self.tokenizer.decode(
                generated if echo else generated[inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            generations.append({"generation": decoded})
        return generations
