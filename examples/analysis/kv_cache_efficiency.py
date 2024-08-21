import numpy as np  
import time  
import torch  
from transformers import AutoModelForCausalLM, AutoTokenizer  
  
device = "cuda" if torch.cuda.is_available() else "cpu"  
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf")  
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-chat-hf").to(device)  

for _ in range(10): # measuring 10 generations   
    
    times = {"with_cache":[], "without_cache":[]}
    for use_cache in (True, False):  
        start = time.time() 
        input_ids = tokenizer("To understand the tradeoffs between the two methods, we evaluate their end-to-end performance and microbenchmark their overheads", return_tensors="pt")
        print(len(input_ids['input_ids'][0]))
        model.generate(**input_ids.to(device), use_cache=use_cache, max_new_tokens=len(input_ids['input_ids'][0])+1)  
        times["with_cache" if use_cache else "without_cache"].append(time.time() - start)  
        print(f"{'with' if use_cache else 'without'} KV caching: {round(np.mean(times['with_cache' if use_cache else 'without_cache']), 3)} +- {round(np.std(times['with_cache' if use_cache else 'without_cache']), 3)} seconds")
