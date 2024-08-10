import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, TrainingArguments

if __name__ == '__main__':
    ckpt_dir = "ARLLM/"
    tokenizer_path = "Meta-Llama-3-8B-Instruct/"
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16
    # )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, add_bos_token=False)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(ckpt_dir, device_map="mps")
    model.resize_token_embeddings(len(tokenizer))

    chat_turns = [
        {
            "role": "system",
            "content": "Your role is an on-site robotics operation engineer who gives technical and practical advice to client who use Lionel robot to draw lines on the ground.\n"
                       "You belong to August Robotics Ltd. Please do not answer any question not relate to our business, you can simply refuse it by saying no.\n"
                       "Let's think through this carefully, step by step.\n"
        },

    ]
    format_user_input = {
            "role": "user", "content": ""
    }
    format_assistant_output = {
        "role": "assistant", "content": ""
    }

    generation_config = GenerationConfig.from_pretrained(ckpt_dir)
    generation_config.max_new_tokens = 150
    generation_config.repetition_penalty = 1.1
    generation_config.temperature = 0.2
    generation_config.top_p = 0.2
    generation_config.top_k = 20

    while True:
        user_input = input("Message ARLLM: ")
        format_user_input["content"] = str(user_input)
        chat_turns.append(format_user_input)
        model_inputs = tokenizer(tokenizer.apply_chat_template(chat_turns, tokenize=False), return_tensors="pt").to("mps")
        if len(model_inputs["input_ids"][0]) >= 1024:
            print("Reached maximum number of tokens, please start a new conversation.")
            break
        output = model.generate(**model_inputs, generation_config=generation_config)
        format_output = tokenizer.decode(output[0][len(model_inputs["input_ids"][0])+2:], skip_special_tokens=True).strip()
        format_assistant_output["content"] = str(format_output)
        chat_turns.append(format_assistant_output)
        print(format_output)