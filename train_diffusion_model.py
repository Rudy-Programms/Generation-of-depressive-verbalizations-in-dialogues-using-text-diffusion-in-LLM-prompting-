from Diffusion_Model import DiffusionModel
from red_team_diffusion import RedTeamDiffusion
from rtdConfig import RTD_Config
from rewarding import PsychInterestJudge
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets
import transformers
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoModelForCausalLM
from argparse import ArgumentParser
import vec2text
from vec2text import models as v2t_models
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import re
import os
import numpy as np
os.environ["WANDB_PROJECT"] = "Red-Team-Diffusion"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
transformers.logging.set_verbosity_error()


def preprocess_alpaca_data(sample):
    sample["query"] = sample["instruction"] + sample["input"]
    return sample

def preprocess_red_teaming_data(datapoint):
    datapoint["query"] = re.findall(r"Human: (.*?)\n", datapoint["transcript"])[0]
    return datapoint

def preprocess_rlhf_data(datapoint):
    datapoint["query"] = re.findall(r"Human: (.*?)\n", datapoint["chosen"])[0].replace("Assistant:", "")
    return datapoint

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = ArgumentParser()
    args.add_argument("--name", type=str, default="unnamed_experiment")
    args.add_argument("--eps", type=float, default=0.1)
    args.add_argument("--batch-size", type=int, default=64)
    args.add_argument("--target-model", type=str, default="vicgalle/gpt2-alpaca")
    args.add_argument("--epochs", type=int, default=1)
    args.add_argument("--log", action="store_true")
    args.add_argument("--dataset", choices=["alpaca", "red-team", "all", "custom"])
    args.add_argument("--data-path", type=str, help="Path to local dataset file")
    args.add_argument("--data-col",  type=str, default="query", help="Column name that contains prompts")
    args.add_argument("--max-prompt-tokens", type=int, default=2048)
    args.add_argument("--cache-dir", type=str)
    args.add_argument("--save-path", type=str, default="unnamed_experiment")
    args.add_argument("--device", type=str)
    args = args.parse_args()

    if args.device:
        device = args.device
        print(device)

    models_root = (args.cache_dir + "/models") if args.cache_dir else None

    t5_dir  = "t5-base"
    gtr_name = "sentence-transformers/gtr-t5-base"

    diffusion_model = DiffusionModel(t5_dir, 768, cache_dir=models_root)
    diffusion_tokenizer = AutoTokenizer.from_pretrained(
        t5_dir, cache_dir=models_root, local_files_only=True
    )

    # SentenceTransformer-based embedder (matches vec2text)
    embedder = SentenceTransformer(
        gtr_name,
        cache_folder=models_root,
    )
    embedder.max_seq_length = 512
    embedder_tokenizer = None  # no longer needed; kept for API compatibility


    diffusion_tokenizer.model_max_length = 2048

    os.environ["TORCHDYNAMO_DISABLE"] = "1"  # avoid accidental tracing on import

    # Force HF to build full CPU weights instead of lazy/meta
    from sentence_transformers.models import Transformer as _STTransformer
    __orig_st_init = _STTransformer.__init__
    def __patched_st_init(self, *args, **kwargs):
        model_args = kwargs.get("model_args") or {}
        model_args["low_cpu_mem_usage"] = False
        model_args["device_map"] = None
        model_args["torch_dtype"] = torch.float32
        kwargs["model_args"] = model_args
        return __orig_st_init(self, *args, **kwargs)
    _STTransformer.__init__ = __patched_st_init

    # load reconstruction model

    inversion_model = v2t_models.InversionModel.from_pretrained(
        "ielabgroup/vec2text_gtr-base-st_inversion",
        low_cpu_mem_usage=False,
        device_map=None,
        torch_dtype=torch.float32
    )
    v2t_root = (args.cache_dir + "/models/") if args.cache_dir else None
    inv_path = "ielabgroup/vec2text_gtr-base-st_inversion"
    print(f"[train] loading vec2text inversion model from {inv_path}", flush=True)
    inversion_model = v2t_models.InversionModel.from_pretrained(
        inv_path,
        low_cpu_mem_usage=False,
        device_map=None,
        torch_dtype=torch.float32,
    )

    corr_path = "ielabgroup/vec2text_gtr-base-st_corrector"
    print(f"[train] loading vec2text corrector encoder from {corr_path}", flush=True)
    corrector_model = v2t_models.CorrectorEncoderModel.from_pretrained(
        corr_path,
        low_cpu_mem_usage=False,
        device_map=None,
        torch_dtype=torch.float32,
    )

    decoder = vec2text.load_corrector(inversion_model, corrector_model)


    # load reward model
    models_root = (args.cache_dir + "/models") if args.cache_dir else None
    tox_dir = "nicholasKluge/ToxicityModel"

    toxicity_tokenizer = AutoTokenizer.from_pretrained(
        tox_dir, cache_dir=models_root
    )
    toxicity_classifier = AutoModelForSequenceClassification.from_pretrained(
        tox_dir, cache_dir=models_root
    )

    # load target model
    models_root = (args.cache_dir + "/models") if args.cache_dir else None
    tm_dir = args.target_model

    target_model = AutoModelForCausalLM.from_pretrained(
        tm_dir, cache_dir=models_root
    )
    target_model.to(device)

    target_tokenizer = AutoTokenizer.from_pretrained(
        tm_dir, cache_dir=models_root
    )

    if target_tokenizer.pad_token is None:
        if getattr(target_tokenizer, "eos_token", None) is not None:
            target_tokenizer.pad_token = target_tokenizer.eos_token
        else:
            target_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            target_model.resize_token_embeddings(len(target_tokenizer))
    target_model.config.pad_token_id = target_tokenizer.pad_token_id
    try:
        target_tokenizer.padding_side = "left"
    except Exception:
        pass


    # set up target model generation
    if "gpt2" in args.target_model:
        generation_config =  {"do_sample":False, 
                              "temperature" : 1, 
                              "top_p" : 0.92, 
                              "top_k" : 0, 
                              "max_new_tokens" : 32} 
        system_prefix = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n"
        system_suffix = "\n### Response:\n\n"
    elif "vicuna" in args.target_model:
        generation_config =  {"do_sample":False, 
                              "temperature" : 1, 
                              "top_p" : 0.92, 
                              "top_k" : 0, 
                              "max_new_tokens" : 32}
        system_prefix = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nUSER:"""
        system_suffix = "\nASSISTANT:\n"
    elif "llama" in args.target_model:
        system_prefix = "[INST]"
        system_suffix = "[/INST]"
        generation_config =  {"do_sample":False, 
                              "temperature" : 1, 
                              "top_p" : 0.92, 
                              "top_k" : 0, 
                              "max_new_tokens" : 32}
        target_tokenizer.pad_token = target_tokenizer.eos_token
    
    # configure trainer
    config = RTD_Config(
        log_wandb=args.log,
        eps=args.eps,
        name=args.name,
        batch_size=args.batch_size,
        minibatch_size=max(1, min(args.batch_size, 32))        
    )
    config.batch_size = args.batch_size

    interest_judge = None
    if config.reward == "psych_interest":
        interest_judge = PsychInterestJudge(
            api_key=os.getenv(config.gemini_api_key_env),
            model=config.gemini_model,
            timeout_s=config.judge_timeout_s,
            max_retries=config.judge_retries,
        )
        
    trainer = RedTeamDiffusion(
        config=config,
        diffusion_model=diffusion_model,
        diffusion_tokenizer=diffusion_tokenizer,
        embedder=embedder,
        embedder_tokenizer=embedder_tokenizer,
        decoder=decoder,
        device=device
    )

    # load data
    if args.dataset == "alpaca":
        dataset = load_dataset("vicgalle/alpaca-gpt4", split="train", cache_dir=(args.cache_dir+"/data/") if args.cache_dir else None)
        dataset = dataset.map(preprocess_alpaca_data)
    elif args.dataset == "red-team":
        dataset = load_dataset("Anthropic/hh-rlhf", data_dir="red-team-attempts")
        dataset = dataset["train"].map(preprocess_red_teaming_data)
    elif args.dataset == "all":
        alpaca = load_dataset("vicgalle/alpaca-gpt4", split="train", cache_dir=(args.cache_dir+"/data/") if args.cache_dir else None)
        alpaca = alpaca.map(preprocess_alpaca_data).select(range(4096, len(alpaca)))
        red_team = load_dataset("Anthropic/hh-rlhf", data_dir="red-team-attempts")
        red_team = red_team["train"].map(preprocess_red_teaming_data).select(range(4096, len(red_team["train"])))        
        dataset = concatenate_datasets([red_team, red_team])
        dataset = dataset.shuffle()
    elif args.dataset == "custom":
        import os
        path = args.data_path
        if path is None or not os.path.exists(path):
            raise ValueError("--data-path must point to an existing file for --dataset custom")
        ext = os.path.splitext(path)[1].lower()
        if ext in [".txt"]:
            # one prompt per line
            from datasets import Dataset
            with open(path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            dataset = Dataset.from_dict({ "query": lines })
        elif ext in [".jsonl", ".json"]:
            dataset = load_dataset("json", data_files=path, split="train")
            if args.data_col != "query":
                dataset = dataset.rename_column(args.data_col, "query")
        elif ext in [".csv"]:
            dataset = load_dataset("csv", data_files=path, split="train")
            if args.data_col != "query":
                dataset = dataset.rename_column(args.data_col, "query")
        else:
            from datasets import Dataset
            with open(path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            dataset = Dataset.from_dict({ "query": lines })
        # shuffle for randomness
        dataset = dataset.shuffle(seed=42)

    print(len(dataset))
    dataset = dataset.filter(lambda sample: len(diffusion_tokenizer.encode(sample["query"])) < args.max_prompt_tokens)
    print(len(dataset))
    if args.dataset != "custom":
        dataset = dataset.select(range(4096, len(dataset)))
        val_dataset = dataset.select(range(1024))
    else:
        # simple 90/10 split 
        split_n = max(1, int(0.1 * len(dataset)))
        val_dataset = dataset.select(range(0, split_n))
        dataset = dataset.select(range(split_n, len(dataset)))
    dataset = DataLoader(dataset["query"], batch_size=config.batch_size)
    val_dataset = DataLoader(val_dataset["query"], config.batch_size//2)
    best_rew = float("-inf")
    # main training loop
    for i in range(args.epochs):
        for j, batch in enumerate(tqdm(dataset)):
            # modify prompt
            modified_prompt, tokens, noise, logprobs = trainer.modify_prompt(batch, num_steps=80, deterministic=True)

            # get llm output
            target_tokens = target_tokenizer([system_prefix + prompt + system_suffix for prompt in modified_prompt], return_tensors="pt", padding=True)
            target_tokens = target_tokens.to(device)
            target_out = target_model.generate(**target_tokens, **generation_config)
            if "gpt2" in args.target_model:
                target_out = [target_tokenizer.decode(out).replace(system_prefix + modified_prompt[i] + system_suffix, "").replace("<|endoftext|>", "") for i, out in enumerate(target_out)]
            elif "vicuna" in args.target_model:
                target_out = [target_tokenizer.decode(out).replace(system_prefix + modified_prompt[i] + system_suffix, "").replace("<unk>", "") for i, out in enumerate(target_out)]
            elif "Llama" in args.target_model:
                target_out = [target_tokenizer.decode(out).replace(system_prefix + modified_prompt[i] + system_suffix, "").replace("<s>", "").replace("</s>", "") for i, out in enumerate(target_out)]
            # get reward
            if config.reward == "psych_interest":
                # Single scalar in [0,1] from Google AI judge.
                scores = [interest_judge.score(conv) for conv in target_out]
                rewards = torch.tensor(scores, dtype=torch.float32).unsqueeze(-1).detach()
            else:
                rewards = (torch.tensor([-toxicity_classifier(**toxicity_tokenizer(out, return_tensors="pt", padding=True)).logits.item() for out in target_out]).unsqueeze(-1).detach())
                if config.reward == "1/0":
                    rewards = (rewards > 0).float()
                elif config.reward == "relative_toxicity":
                    unmod_prompt = target_tokenizer([system_prefix + prompt + system_suffix for prompt in batch], return_tensors="pt", padding=True)
                    unmod_prompt = unmod_prompt.to(device)
                    unmod_out = target_model.generate(**unmod_prompt, **generation_config)
                    unmod_out = [target_tokenizer.decode(out)[len(system_prefix)+len(modified_prompt[i])+len(system_suffix):].replace("<|endoftext|>", "") for i, out in enumerate(unmod_out)]
                    baseline_reward = (torch.tensor([-toxicity_classifier(**toxicity_tokenizer(out, return_tensors="pt", padding=True)).logits.item() for out in unmod_out]).unsqueeze(-1))
                    rewards = rewards - baseline_reward
            
            # update model
            modified_ids = diffusion_tokenizer(modified_prompt, return_tensors="pt", padding=True).to(device)
            stats = trainer.training_step(**tokens, modified_ids=modified_ids["input_ids"], modified_masks=modified_ids["attention_mask"], noise=noise, logprobs_old=logprobs, rewards=rewards, batch=batch)
            if config.reward == "relative_toxicity":
                stats["env/toxicity_dist"] = (rewards + baseline_reward).squeeze()
                stats["env/toxicity_mean"] = (rewards + baseline_reward).squeeze().mean().item()
                stats["env/toxic_answer_ratio"] = (stats["env/toxicity_dist"] > 0).float().mean()
            else:
                stats["env/toxicity_dist"] = rewards.squeeze()   # here: actually "interest" if psych_interest
                stats["env/toxicity_mean"] = rewards.mean().item()
            # log progress
            if args.log and j % 10 == 0:
                # evaluate
                cumulative_metric, val_modified_prompts, val_batches, val_outs = [], [], [], []
                for val_batch in val_dataset:
                    with torch.no_grad():
                        val_prompts_modified, _, _, _ = trainer.modify_prompt(val_batch, num_steps=80, deterministic=True)
                        target_tokens = target_tokenizer([system_prefix + prompt + system_suffix for prompt in val_prompts_modified], return_tensors="pt", padding=True)
                        target_tokens = target_tokens.to(device)
                        target_out_val = target_model.generate(**target_tokens, **generation_config)
                        if "gpt2" in args.target_model:
                            target_out_val = [target_tokenizer.decode(out).replace(system_prefix + val_prompts_modified[i] + system_suffix, "").replace("<|endoftext|>", "").replace("<|endoftext|>", "") for i, out in enumerate(target_out_val)]
                        elif "vicuna" in args.target_model:
                            target_out_val = [target_tokenizer.decode(out).replace(system_prefix + val_prompts_modified[i] + system_suffix, "").replace("<unk>", "") for i, out in enumerate(target_out_val)]
                        elif "Llama" in args.target_model:
                            target_out_val = [target_tokenizer.decode(out).replace(system_prefix + val_prompts_modified[i] + system_suffix, "").replace("<s>", "").replace("</s>", "") for i, out in enumerate(target_out_val)]
                        val_modified_prompts += val_prompts_modified
                        val_batches += val_batch
                        val_outs += target_out_val
                        # get reward
                        if config.reward == "psych_interest":
                            cumulative_metric += [interest_judge.score(conv) for conv in target_out_val]
                        else:
                            cumulative_metric += [-toxicity_classifier(**toxicity_tokenizer(out, return_tensors="pt", padding=True)).logits.item() for out in target_out_val]
                eval_rewards = float(np.mean(cumulative_metric)) if len(cumulative_metric) else 0.0
                if eval_rewards > best_rew:
                    torch.save(diffusion_model.state_dict(), args.save_path)
                    best_rew = eval_rewards
                with torch.no_grad():
                    trainer.log(stats, batch, modified_prompt, target_out, val_batches, val_modified_prompts, val_outs, np.array(cumulative_metric))
                if config.lr_schedule:
                    trainer.scheduler.step(eval_rewards)

torch.save(diffusion_model.state_dict(), args.save_path)

from rewarding import JUDGE_TOTAL_CALLS, JUDGE_FAILED_CALLS
print("\n=== PsychInterestJudge stats ===")
print(f"Total judge calls: {JUDGE_TOTAL_CALLS}")
print(f"Failures (including retries): {JUDGE_FAILED_CALLS}")
print(f"Approx failure rate: {JUDGE_FAILED_CALLS / max(1, JUDGE_TOTAL_CALLS):.2%}")

print(f"Saved final model to {args.save_path}")
            