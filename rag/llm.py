from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

import os
from dotenv import load_dotenv

load_dotenv()

def model_load():
    token = os.getenv('HF_TOKEN')
    try:
        # Load the GPT-2 model
        hf = HuggingFacePipeline.from_model_id(
            model_id="gpt2", 
            task="text-generation",
            pipeline_kwargs={"temperature": 0.5, "max_new_tokens": 300}
        )
        llm = hf
        print("Successfully loaded  model.")
    except Exception as e:
        print(f"An error occurred: {e}")
        llm = None
    
    return llm
