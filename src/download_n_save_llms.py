import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download

HF_TOKEN = "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"
MODEL_DIR_PATH = "/Users/tusharmungle/Documents/Projects/llm_prompting/models"


def save_model_pretrained(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=HF_TOKEN)

    subfolder_path = os.path.join(MODEL_DIR_PATH, model_name)
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)

    model.save_pretrained(subfolder_path)
    tokenizer.save_pretrained(subfolder_path)

    print(f"Saved model and tokenizer to:", subfolder_path)
    print("Contents of save path: ")
    print(os.listdir(subfolder_path))


def save_model_snapshot(model_name):
    model_str = model_name.replace('/', '-')
    model_folder_name = f"model-{model_str}"
    path_to_model = os.path.join(MODEL_DIR_PATH, model_folder_name)

    if not os.path.exists(path_to_model):
        os.makedirs(path_to_model)

    snapshot_download(repo_id=model_name,
                      repo_type="model",
                      local_dir=path_to_model,
                      token=HF_TOKEN)


def initialize_local_model(model_name):
    model_str = model_name.replace('/', '-')
    model_folder_name = f"model-{model_str}"
    path_to_model = os.path.join(MODEL_DIR_PATH, model_folder_name)

    tokenizer = AutoTokenizer.from_pretrained(path_to_model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(path_to_model, local_files_only=True)

    return tokenizer, model


def main():
    save_model_snapshot("meta-llama/Llama-3.2-1B")
    print("1")
    # save_model_snapshot("google/gemma-2-27b-it")
    # print("11")
    # # save_model_snapshot("Qwen/Qwen2.5-7B-Instruct") # DONE
    # save_model_snapshot("Qwen/Qwen2.5-32B-Instruct")
    # print("12")
    # save_model_snapshot("01-ai/Yi-1.5-9B-Chat-16K")
    # print("13")
    # save_model_snapshot("01-ai/Yi-1.5-34B-Chat-16K")
    # print("14")
    # save_model_snapshot("mistralai/Ministral-8B-Instruct-2410")
    # print("15")
    # save_model_snapshot("mistralai/Mistral-Small-Instruct-2409")
    # print("16")
    # save_model_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    # save_model_snapshot("meta-llama/Meta-Llama-3-8B-Instruct")
    # save_model_snapshot("google/gemma-7b-it")
    # save_model_snapshot("mistralai/Mistral-7B-Instruct-v0.2")
    # save_model_snapshot("mistralai/Mistral-7B-Instruct-v0.2")
    # save_model_snapshot("google/gemma-2-9b-it")
    # save_model_snapshot("aaditya/Llama3-OpenBioLLM-8B")
    # save_model_snapshot("m42-health/Llama3-Med42-8B")
    # save_model_snapshot("medalpaca/medalpaca-7b")
    # save_model_snapshot("instruction-pretrain/medicine-Llama3-8B")
    # save_model_snapshot("AdaptLLM/medicine-chat")
    # save_model_snapshot("medicalai/ClinicalGPT-base-zh")
    # print("1")
    # save_model_snapshot("medicalai/ClinicalBERT")
    # print("2")
    # save_model_snapshot("mradermacher/Bio-Medical-Llama-3.1-8B-i1-GGUF")
    # print("3")
    # save_model_snapshot("ContactDoctor/Bio-Medical-MultiModal-Llama-3-8B-V1")
    # print("4")
    # save_model_snapshot("instruction-pretrain/medicine-Llama3-8B")
    # print("5")
    # tokenizer, model = initialize_local_model("meta-llama/Llama-3.3-70B-Instruct")
    # print(model)
    # tokenizer, model = initialize_local_model("google/gemma-2-9b-it")
    # print(model)
    # tokenizer, model = initialize_local_model("mistralai/Mistral-7B-Instruct-v0.2")
    # print(model)


if __name__ == "__main__":
    main()
