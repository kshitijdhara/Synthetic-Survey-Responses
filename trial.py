import torch
from transformers import pipeline

model_id = "NousResearch/Hermes-3-Llama-3.1-70B-FP8"

pipe = pipeline(
    "text-generation", 
    model=model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)

pipe("The key to life is")