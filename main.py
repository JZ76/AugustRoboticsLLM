from llama3 import Llama
import torch
from pathlib import Path
import json
from llama3.model import ModelArgs, Transformer
from llama3.tokenizer import ChatFormat, Dialog, Message, Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
from peft import LoraConfig, prepare_model_for_kbit_training

# adding new vocabulary : https://nlp.stanford.edu/~johnhew/vocab-expansion.html
# convert original checkpoint into transformers format : https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ckpt_dir = "F:\AugustRoboticsLLM\Meta-Llama-3-8B-Instruct"
    tokenizer_path = "F:\AugustRoboticsLLM\Meta-Llama-3-8B-Instruct"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    peft_config = LoraConfig(
        r=32,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    generation_config = GenerationConfig.from_pretrained(ckpt_dir)
    generation_config.max_new_tokens=100
    generation_config.repetition_penalty = 1.1
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(ckpt_dir, quantization_config=bnb_config, device_map="auto")
    model.resize_token_embeddings(len(tokenizer))

    # model = prepare_model_for_kbit_training(model)

    #
    # checkpoints = sorted(Path("F:\llama3\Meta-Llama-3-8B-Instruct").glob("*.pth"))
    # ckpt_path = checkpoints[0]
    #
    # checkpoint = torch.load(ckpt_path, weights_only=True, map_location="cpu")
    # with open(Path("F:\llama3\Meta-Llama-3-8B-Instruct") / "params.json", "r") as f:
    #     params = json.loads(f.read())
    # model_args: ModelArgs = ModelArgs(
    #     max_seq_len=64,
    #     max_batch_size=1,
    #     **params,
    # )
    # tokenizer = Tokenizer(model_path="F:\llama3\Meta-Llama-3-8B-Instruct\/tokenizer.model")
    # model = Transformer(model_args)
    # model.load_state_dict(checkpoint, strict=False, )
    # model.to("cpu")
    # generator = Llama(model, tokenizer)

    # generator = Llama.build(
    #     ckpt_dir="F:\llama3\Meta-Llama-3-8B-Instruct",
    #     tokenizer_path="F:\llama3\Meta-Llama-3-8B-Instruct\/tokenizer.model",
    #     max_seq_len=128,
    #     max_batch_size=8,
    #     model_parallel_size=1
    # )
    prompts = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is",
        "Simply put, the theory of relativity states that ",
        """A brief message congratulating the team on the launch:

        Hi everyone,

        I just """,
        # Few shot prompt (providing a few examples before asking model to complete more);
        """Translate English to French:

        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese =>""",
    ]
    # result = pipeline("text-generation", model="F:\AugustRoboticsLLM\Meta-Llama-3-8B-Instruct",
    #                   model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
    # result(prompts)

    model_inputs = tokenizer("How are you today?", return_tensors="pt").to("cuda")
    print(model_inputs)
    output = model.generate(**model_inputs, generation_config=generation_config)

    print(tokenizer.decode(output[0], skip_special_tokens=True))
