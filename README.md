# Open Source Mentor AI

An assistant that helps developers understand GitHub issues, interpret unfamiliar codebases, and plan their first contributions.


## Sourcing Data

3 criteria matters most in choosing the right data for fine-tuning:
- Difficult 
- Quality
- Diversity
 
Key lesson - Fine-tuning quality is bounded by data quality far more than by clever trainng optimizations.

Based on the 3 criteria defined above, below are the hard filters applied in selection of the issues:
- A descriptive issue body
- 1-8 files bounded PRs
- Not a bot, dependabot, docs-typo, or release PR
- Issues from repo with permissive licence
- Closed issues


I am going to use a retrieval and judge style meaning a model (Claude family) in my case will retrieve potential candidates of 200 such issues, and another model (Codex family) will judge.

Judge's job:
- Per-issue scoring: Is this a good example? Helps avoid position bias
- Set-level selection: Does the chosen set cover the space? 


Per-issue criteria is yet to be decided but must be sound - Everything else downstream is anchored on it.

Finally, I, the human in the loop will review these datasets to see if they deserve my approval.

The dataset is in a ChatML Format which is the format expected by the SFTTrainer used in this training.


Limitation Note: The steps outlines above in sourcing these dataset are not in any way indicative of the most optimal approach in sourcing dataset in production pipelines, however this is an experimental project and I am tinkering with approaches to see where it lands. 



## Building QLORA Adapter

It is worth mentioning that most tasks do not require fine tuning, and can be easily solved with RAG or prompt engineering techniques hence fine-tuning should only be considered when it is the most efficable choice. 

Model chosen: Qwen3-4B (unsloth/Qwen3-4B-unsloth-bnb-4bit)
Why? It's small enough to train on a consumer GPU with LoRA/QLoRA, has long context, and is strong enough for small coding tasks. I started at 4B to iterate quickly and keep memory low. The setup isn't tied to this size, so I can move to 8B or larger by changing the model name if results call for it. With more time and resources I'd also compare different model families.
I'm fine-tuning from the post-trained (chat) checkpoint rather than the base model, because it already follows instructions and I have limited data.

LORA is when instead of training all model parameters, we train a small set of added low-rank adapter weights (in 16-bit precision). QLORA is just a quantized version of that achieved through precision reduction to 4 bits trading small accuracy for huge memory optim reducing base-model weight memory by roughly 75% (Unsloth's dynamic 4-bit quants have been known to recover this minimal accuracy loss).

r (rank) - higher means more more capacity and target_modules (what layers are targeted) are the most important hyperparameters when doing a LoRA fine-tuning and must be tuned to see what perform best. I am gonna start with the recommended default r=16 and all-linear targeting. There are also other important hyperparameters that must be tuned as needed such as lora_alpha whose default is typically recommended to be same as r or r*2. Parameters like bias, use_gradient_checkpointing etc allows us to decide how memory efficient or optimized the training run will be. This is especially important if you are running it on a local device with limited resources.




## Training

It is worth mentioning that there are multiple ways a supervised fine-tuning can be carried out. One such way is using trl which would be more hands on, but an easier route is using Unsloth which wraps TRL's SFTTrainer with optimized kernels and simpler model loading, so training is faster and more memory-efficient while the hyperparameters stay exposed.


During training on my local device Mac, It could not use the dynamic 4-bit quant that was used in the training script because bitsandbytes does not support Apple Silicon hence it defaulted to MLX 4-bit quantization instead.