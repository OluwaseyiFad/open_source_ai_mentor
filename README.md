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

The dataset is in a ChatML Format which is the format expected by the SFTTrainer used in this training. Ensure the format is properly checked as malformed ChatML templates causes the model to train on garbage tokens


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


Fine-tuning on a few hundreds data points is generally not recommended as it is prone to overfitting, however there are hyperparamters choices that could help reduce that possibility. For example, eval_stratgey and eval_steps matters because that determines how many validation points with which we choose the best checkpoint that gets loaded at the end. Learning rate is also important as we don't want it to be too low due to model not learning esp given limited datasets nor too high due to wild oscillations and swings in loss curve. Many other hyperparamters chosen in this code also matters, but the idea is they have to be intentionally set (or tuned) based on the what we are working with.



### Things to watch out for 
- Overfitting: when training loss improves while validation climbs
- Catastophic forgetting: Model performing well on the narrow tasks but degraging on general tasks
- Loss curve flatness or huge swings: Flatness in loss curve can arise due to too low LR, and huge swings can occur due to too high LR.
- Cuda OOM errors (especially on consumer GPUs): tune hyperparameters to be more memory efficient based on the device limitation.


## Evaluation

Evaluation is just as important (if not more) than the training or fine-tuning itself. So it is very important to carefully choose the metrics that would be used in evaluation. 

The metric to be choosen has to be evaluated itself (evaluate the evaluation), and the metric has to be task-specific. Different kind of metrics are suitable to different kind of tasks, for example F1 Score and exact accuracy is good for classification task, LLM-as-a-judge and human evaluation for generation task when optimizing for human preferences, and BLUE and ROGUE when not.