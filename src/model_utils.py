# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

def get_tokenizer_and_model(model_name, attn_implementation, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        if tokenizer.bos_token is not None:
            tokenizer.add_special_tokens({'pad_token': tokenizer.bos_token})
        else:
            tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    model_kwargs = {
        "torch_dtype": torch.float16,
        "attn_implementation": attn_implementation,
        "cache_dir": "/dev/shm",
        "pretrained_model_name_or_path": model_name,
    }

    # Check if model size with float16 is greater than single GPU memory
    model_size_gb = 0
    match = re.search(r'(\d+(\.\d+)?)([mMbB])', model_name)
    if match:
        size, _, unit = match.groups()
        size = float(size)
        if unit.lower() == 'b':
            model_size_gb = size
        elif unit.lower() == 'm':
            model_size_gb = size / 1024  # Convert MB to GB
    single_gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if model_size_gb * 2 > single_gpu_mem_gb:  # float16 is 2 bytes per parameter
        model_kwargs["device_map"] = "auto"
    if model_size_gb > 8:
        model_kwargs["torch_dtype"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

    model.eval()

    if "70B" not in model_name:
        model = model.to(device)

    return model, tokenizer

def get_inst_tokens(model_name, use_sys_token = False, enable_thinking = False):
    inst_token_dict = {
        "deepseek": ["<｜{}｜>", "", "<think>\n"],
        "llama": ("<|start_header_id|>{}<|end_header_id|>\n\n", "", "<|eot_id|>"),
        "mistral": ("INST] ", "[", " [/"),
        "qwen": ("<|im_start|>{}\n", "", "<|im_end|>\n"),
    }

    model_key = next((k for k in inst_token_dict if k in model_name.lower()), None)
    if model_key is None:
        raise ValueError(f"Model name '{model_name}' does not match any known token patterns.")
    is_thinking_model = "deepseek" in model_name.lower() or "qwen3" in model_name.lower()
    inst_tokens = inst_token_dict[model_key]
    tokens_to_return = ["","",""]
    if use_sys_token and model_key not in ["mistral", "deepseek"]:
        tokens_to_return[0] = inst_tokens[1]+inst_tokens[0].format("system")
        tokens_to_return[1] = inst_tokens[2]+inst_tokens[0].format("user")
        tokens_to_return[2] = inst_tokens[2]+inst_tokens[0].format("assistant")
    elif model_key == "deepseek":
        tokens_to_return[0] = inst_tokens[1]+inst_tokens[0].format("User")
        tokens_to_return[1] = ""
        tokens_to_return[2] = inst_tokens[0].format("Assistant")+inst_tokens[2]
    else:
        tokens_to_return[0] = inst_tokens[1]+inst_tokens[0].format("user")
        tokens_to_return[1] = ""
        tokens_to_return[2] = inst_tokens[2]+inst_tokens[0].format("assistant")
    if is_thinking_model and not enable_thinking:
        tokens_to_return[2] += "<think>\n\n</think>\n\n"
    return tokens_to_return

def get_eos_token_ids(stop_strings, tokenizer):
    token_ids = []
    for token_str in stop_strings:
        # Check if token is a known token
        if token_str in tokenizer.vocab or token_str in tokenizer.get_vocab():
            token_ids.append(tokenizer.convert_tokens_to_ids(token_str))
        else:
            # Otherwise encode using a dummy prefix
            dummy_text = "dummy" + token_str
            encoded_ids = tokenizer.encode(dummy_text, add_special_tokens=False)
            # Use the id of the newly encoded token
            token_ids.append(encoded_ids[-1])
    return token_ids
