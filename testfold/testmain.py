def test_text_completion():
    from transformers import logging
    from testllm import HuggingFaceLLM
    logging.set_verbosity_error()  # 关闭不必要的 warning

    # 替换为你想测试的模型名，如：'meta-llama/Llama-2-7b-hf'
    model_name = "meta-llama/Llama-2-7b-hf"
    
    llm = HuggingFaceLLM(model_name=model_name, device="cuda" if torch.cuda.is_available() else "cpu")

    prompts = [
        "Once upon a time",
        "The meaning of life is",
        "Artificial intelligence will"
    ]

    results = llm.text_completion(
        prompts=prompts,
        temperature=0.7,
        top_p=0.9,
        max_gen_len=50,
        echo=False,
    )

    for i, res in enumerate(results):
        print(f"\n--- Prompt {i+1} ---")
        print("Prompt:", prompts[i])
        print("Generation:", res["generation"])

if __name__ == "__main__":
    test_text_completion()
