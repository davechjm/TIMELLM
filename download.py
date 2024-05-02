# %%
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer

def download_llama_resources(model_name='huggyllama/llama-7b'):
    """
    Download LLama model and tokenizer from Hugging Face and save them locally.

    Parameters:
    - model_name (str): The Hugging Face model repository name.
    """
    # Download and save the LLama model configuration
    try:
        llama_config = LlamaConfig.from_pretrained(model_name)
        llama_config.num_hidden_layers = 32
        llama_config.output_attentions = True
        llama_config.output_hidden_states = True
        llama_config.save_pretrained('./llama_model')
        print('Model Downloaded 1')
    except Exception as e:
        print(f"Failed to download or save LLama model configuration: {e}")

    # Download and save the LLama model
    try:
        llama_model = LlamaModel.from_pretrained(model_name, trust_remote_code = True, local_files_only = True, config=llama_config)
        llama_model.save_pretrained('./llama_model')
        print("Model Downloaded 2")
    except Exception as e:
        print(f"Failed to download or save LLama model: {e}")

    # Download and save the tokenizer
    try:
        tokenizer = LlamaTokenizer.from_pretrained(model_name, trust_remote_code = True, local_files_only = True)
        tokenizer.save_pretrained('./llama_model')
        print('Model Downloaded3')
    except Exception as e:
        print(f"Failed to download or save tokenizer: {e}")

if __name__ == '__main__':
    download_llama_resources()
    print("Model Downloads Completed.")


