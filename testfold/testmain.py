def test_text_completion():
    from transformers import logging
    from testllm import HuggingFaceLLM
    import torch
    logging.set_verbosity_error()  # 关闭不必要的 warning

    # 替换为你想测试的模型名，如：'meta-llama/Llama-2-7b'
    model_name = "/root/autodl-tmp/downloads/llama-2-7b"
    
    llm = HuggingFaceLLM(model_name=model_name, device="cuda" if torch.cuda.is_available() else "cpu")

    prompts = [
        "a photo of "
    ]

    results = llm.text_completion(
        prompts=prompts,
        max_gen_len=10,
        temperature=0,
        top_p=1.0,
    )

    for i, res in enumerate(results):
        print(f"\n--- Prompt {i+1} ---")
        print("Prompt:", prompts[i])
        print("Generation:", res["generation"])

def test_print_tokens():
    """测试函数：打印模型词汇表的前500个token"""
    from transformers import logging
    from testllm import HuggingFaceLLM
    import torch
    logging.set_verbosity_error()  # 关闭不必要的 warning

    # 替换为你想测试的模型名
    model_name = "/root/autodl-tmp/downloads/llama-2-7b"
    
    print("正在加载模型...")
    llm = HuggingFaceLLM(model_name=model_name, device="cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n模型词汇表大小: {llm.tokenizer.vocab_size}")
    print("=" * 50)
    print("前500个token:")
    print("=" * 50)
    
    # 打印前500个token
    for token_id in range(5000, min(5500, llm.tokenizer.vocab_size)):
        try:
            token_str = llm.decode(token_id)
            # 处理特殊字符显示
            if token_str.strip() == "":
                display_str = "[EMPTY/SPACE]"
            elif "\n" in token_str:
                display_str = token_str.replace("\n", "\\n")
            elif "\t" in token_str:
                display_str = token_str.replace("\t", "\\t")
            else:
                display_str = token_str
            
            print(f"Token ID {token_id:3d}: '{display_str}'")
        except Exception as e:
            print(f"Token ID {token_id:3d}: [ERROR: {e}]")

if __name__ == "__main__":
    # test_text_completion()  # 注释掉原来的测试
    test_print_tokens()  # 运行新的测试函数
