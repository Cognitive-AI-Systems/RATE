import torch
import wandb
from tqdm import tqdm
import numpy as np

from RATE_GTrXL import mem_transformer_v2_GTrXL

from TMaze_new.TMaze_new_src.inference.val_tmaze import get_returns_TMaze
from TMaze_new.TMaze_new_src.utils import seeds_list
from TMaze_new.TMaze_new_src.train import FactorScheduler
import torch.nn as nn

def train(model, optimizer, scheduler, raw_model, new_segment, epochs_counter, segments_count, wandb_step, ckpt_path, config, train_dataloader, val_dataloader, max_n_final):
    
    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model is None:
        model = mem_transformer_v2_GTrXL.MemTransformerLM(**config["model_config"])

        model.loss_last_coef = config["training_config"]["coef"]
        torch.nn.init.xavier_uniform_(model.r_w_bias);
        torch.nn.init.xavier_uniform_(model.r_r_bias);
        wandb_step  = 0
        epochs_counter = 0
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["training_config"]["learning_rate"], 
                                      weight_decay=config["training_config"]["weight_decay"], 
                                      betas=(config["training_config"]["beta_1"], config["training_config"]["beta_2"]))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda steps: min((steps+1)/config["training_config"]["warmup_steps"], 1))
        # scheduler = FactorScheduler(optimizer, factor=1.0, stop_factor_lr=config["training_config"]["lr_end_factor"], 
        #                             base_lr=config["training_config"]["learning_rate"], total_iterations = config["training_config"]["epochs"] * config["training_config"]["max_segments"],
        #                             max_segments = config["training_config"]['max_segments'], warmup_steps=config["training_config"]["warmup_steps"], max_epochs=config["training_config"]["epochs"])
        raw_model = model.module if hasattr(model, "module") else model

        print(config)
        
    model.to(device)
    model.train()
    
    wwandb = config["wandb_config"]["wwandb"]
    print(f"model parameters: {sum(p.numel() for p in list(model.parameters()))}")
    it_counter = 0
    best_val_loss = np.inf
    epochs_without_improvement = 0
    max_epochs_without_improvement = 50

    EFFECTIVE_SIZE_BLOCKS = config["training_config"]["context_length"] * config["training_config"]["sections"]
    BLOCKS_CONTEXT = config["training_config"]["context_length"]

    switch = False
    val_loss_all = torch.tensor([float('inf')])
    
    pbar = tqdm(range(config["training_config"]["epochs"]))

    tokens_dict = {}
    tokens_dict[0] = None

    ckpt_dict = {}
    ckpt_dict[0] = None

    for epoch in pbar:
        is_train = True
        model.train()
        for it, batch in enumerate(train_dataloader):
            s, a, rtg, d, timesteps, masks = batch
            memory = None
            mem_tokens = None
            model.cache = None
            
            block_part_range = range(EFFECTIVE_SIZE_BLOCKS//BLOCKS_CONTEXT)
            if tokens_dict[0] is None:
                for block_part in block_part_range:
                    tokens_dict[block_part] = None
            if ckpt_dict[0] is None:
                for block_part in block_part_range:
                    ckpt_dict[block_part] = None
                
            for block_part in block_part_range:
                from_idx = block_part*(BLOCKS_CONTEXT)
                to_idx = (block_part+1)*(BLOCKS_CONTEXT)

                x1 = s[:, from_idx:to_idx, :].to(device)
                y1 = a[:, from_idx:to_idx, :].to(device).float()
                r1 = rtg[:,:,:][:, from_idx:to_idx, :].to(device).float() 
                t1 = timesteps[:, from_idx:to_idx].to(device)
                masks1 = masks[:, from_idx:to_idx].to(device)

                model.flag = 1 if block_part == max(block_part_range) else 0
                if mem_tokens is not None:
                    mem_tokens = mem_tokens.detach()
                    model.cache = model.cache.detach()
                elif raw_model.mem_tokens is not None:
                    mem_tokens = raw_model.mem_tokens.repeat(1, r1.shape[0], 1)
                    model.cache = mem_tokens.permute(1,0,2).detach()
                
                if config["model_mode"] not in  ["DT", "DTXL", "RATE_wo_nmt"]:
                    mean_tokens = torch.mean(mem_tokens, dim=1)[0]

                    if tokens_dict[block_part] is not None:
                        cos_sim = cos(mean_tokens, tokens_dict[block_part]).item()
                    else:
                        cos_sim = None
                with torch.set_grad_enabled(is_train):
                    optimizer.zero_grad()
                    res = model(x1, y1, r1, y1, t1, *memory, mem_tokens=mem_tokens, masks=masks1) if memory is not None else model(x1, y1, r1, y1, t1, mem_tokens=mem_tokens, masks=masks1)
                    memory = res[0][2:]
                    logits, loss = res[0][0], res[0][1]
                    mem_tokens = res[1]
        
                    train_loss_all = model.loss_all
                    if model.flag == 1:
                        train_loss_last = model.loss_last

                    if wwandb and model.flag == 1:
                        wandb.log({"train_last_loss": train_loss_last.item(), 
                                   "train_loss": train_loss_all.item(), 
                                   "train_accuracy": model.accuracy, 
                                   "train_last_acc": model.last_acc})
                    
                    if config["model_mode"] not in  ["DT", "DTXL", "RATE_wo_nmt"]:
                        if wwandb:
                            wandb.log({f"cos_1st_token_block_{block_part}": cos_sim})
                            tokens_dict[block_part] = mean_tokens

                        if ((epoch + 1) % int(config["training_config"]["ckpt_epoch"])) == 0 or epoch == config["training_config"]["epochs"] - 1 or (epoch + 1) == 1:
                            if it == len(train_dataloader)-1:
                                if ckpt_dict[block_part] is not None:
                                    cos_sim_ckpt= cos(mean_tokens, ckpt_dict[block_part]).item()
                                else:
                                    cos_sim_ckpt = None

                                if wwandb:
                                    wandb.log({f"ckpt_cos_1st_token_block_{block_part}": cos_sim_ckpt})

                                ckpt_dict[block_part] = mean_tokens

                if is_train:
                    model.zero_grad()
                    optimizer.zero_grad()
                    train_loss_all.backward(retain_graph=True)
                    if config["training_config"]["grad_norm_clip"] is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config["training_config"]["grad_norm_clip"])
                    optimizer.step()
                    scheduler.step()
                    lr = optimizer.state_dict()['param_groups'][0]['lr']

                    pbar.set_description(f"[train] ep {epoch+1} it {it} tTotal {train_loss_all.item():.2f} vTotal {val_loss_all.item():.2f} lr {lr:e}")
            it_counter += 1 
            epochs_counter += 1
        
        # Val
        model.eval()
        is_train = False
        with torch.no_grad():
            for it, batch in enumerate(val_dataloader):        
                s, a, rtg, d, timesteps, masks = batch
                memory = None
                mem_tokens = None
                model.cache = None
                block_part_range = range(EFFECTIVE_SIZE_BLOCKS//BLOCKS_CONTEXT)
                    
                for block_part in block_part_range:
                    from_idx = block_part*(BLOCKS_CONTEXT)
                    to_idx = (block_part+1)*(BLOCKS_CONTEXT)
                    x1 = s[:, from_idx:to_idx, :].to(device)
                    y1 = a[:, from_idx:to_idx, :].to(device).float()
                    r1 = rtg[:,:,:][:, from_idx:to_idx, :].to(device).float() 
                    t1 = timesteps[:, from_idx:to_idx].to(device)
                    masks1 = masks[:, from_idx:to_idx].to(device)
                        
                    model.flag = 1 if block_part == max(block_part_range) else 0
                    if mem_tokens is not None:
                        mem_tokens = mem_tokens.detach()
                        model.cache = model.cache.detach()
                    elif raw_model.mem_tokens is not None:
                        mem_tokens = raw_model.mem_tokens.repeat(1, r1.shape[0], 1)
                        model.cache = mem_tokens.permute(1,0,2).detach()
                    with torch.set_grad_enabled(is_train):
                        optimizer.zero_grad()
                        res = model(x1, y1, r1, y1, t1, *memory, mem_tokens=mem_tokens, masks=masks1) if memory is not None else model(x1, y1, r1, y1, t1, mem_tokens=mem_tokens, masks=masks1)
                        memory = res[0][2:]
                        logits, loss = res[0][0], res[0][1]
                        mem_tokens = res[1]
                        
                        val_loss_all = model.loss_all
                        if model.flag == 1:
                            val_loss_last = model.loss_last

                        if wwandb and model.flag == 1:
                            wandb.log({"val_last_loss": val_loss_last.item(), 
                                       "val_loss": val_loss_all.item(), 
                                       "val_accuracy": model.accuracy, 
                                       "val_last_acc": model.last_acc,
                                       "learning_rate": lr,
                                       "epoch": epoch})
                    if model.flag == 1:
                        pbar.set_description(f"[val] ep {epoch+1} it {it} tTotal {train_loss_all.item():.2f} vTotal {val_loss_all.item():.2f} lr {lr:e}")

        # Early stopping
        if val_loss_all < best_val_loss:
            best_val_loss = val_loss_all
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= max_epochs_without_improvement and config["training_config"]["use_erl_stop"] == True:
            print("Early stopping!")
            break        

        # Scheduler changer
        if it_counter >= config["training_config"]["warmup_steps"] and switch == False: # !
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, 
                                                          end_factor=config['training_config']['lr_end_factor'], 
                                                          total_iters=len(train_dataloader)*config["training_config"]["epochs"]*config["training_config"]["max_segments"])
            switch = True
        
        if wwandb:
            wandb.log({"segments_count": segments_count})
        
        if ((epoch + 1) % int(config["training_config"]["ckpt_epoch"])) == 0 or epoch == config["training_config"]["epochs"] - 1 or epoch == 0:
            if config["training_config"]["online_inference"]:
                model.eval()
                with torch.no_grad():
                    goods, bads = 0, 0
                    timers = []
                    rewards = []
                    seeds = seeds_list
                    pbar2 = range(len(seeds))
                    for indx, iii in enumerate(pbar2):
                        episode_return, act_list, t, _ , delta_t, attn_map = get_returns_TMaze(model=model, ret=config["data_config"]["desired_reward"], 
                                                                                               seed=seeds[iii], 
                                                                                               episode_timeout=config["online_inference_config"]["episode_timeout"],
                                                                                               corridor_length=config["online_inference_config"]["corridor_length"], 
                                                                                               context_length=config["training_config"]["context_length"],
                                                                                               device=device, act_dim=config["model_config"]["ACTION_DIM"],
                                                                                               config=config, create_video=False)
                        if episode_return == config["data_config"]["desired_reward"]:
                            goods += 1
                        else:
                            bads += 1
                        timers.append(delta_t)
                        rewards.append(episode_return)
                        
                        pbar.set_description(f"[inference | {indx+1}/{len(seeds)}] ep {epoch+1} it {it} tTotal {train_loss_all.item():.2f} vTotal {val_loss_all.item():.2f} lr {lr:e}")
                            
                    suc_rate = goods / (goods + bads)
                    ep_time = np.mean(timers)

                    if wwandb:
                        wandb.log({"Success_rate": suc_rate, "Mean_D[time]": ep_time})

        # ! INFERENCE AT ALL LENGHTS AT LAST EPOCH !!!
        if epoch == config["training_config"]["epochs"] - 1 and segments_count == max_n_final:
            for _segments in [1, 2, 3, 5, 7, 9, 12, 16, 20, 25, 30]:
                _episode_timeout = 30*_segments
                _corridor_length = 30*_segments - 2
                if config["training_config"]["last_inference"]:
                    model.eval()
                    with torch.no_grad():
                        goods, bads = 0, 0
                        timers = []
                        rewards = []
                        seeds = seeds_list
                        pbar2 = range(len(seeds))
                        for indx, iii in enumerate(pbar2):
                            episode_return, act_list, t, _ , delta_t, attn_map = get_returns_TMaze(model=model, ret=config["data_config"]["desired_reward"], 
                                                                                                seed=seeds[iii], 
                                                                                                episode_timeout=_episode_timeout,
                                                                                                corridor_length=_corridor_length, 
                                                                                                context_length=config["training_config"]["context_length"],
                                                                                                device=device, act_dim=config["model_config"]["ACTION_DIM"],
                                                                                                config=config, create_video=False)
                            if episode_return == config["data_config"]["desired_reward"]:
                                goods += 1
                            else:
                                bads += 1
                            timers.append(delta_t)
                            rewards.append(episode_return)
                            
                            pbar.set_description(f"[final inference| S: {_segments} | {indx+1}/{len(seeds)}]")
                                
                        suc_rate = goods / (goods + bads)
                        ep_time = np.mean(timers)

                        if wwandb:
                            wandb.log({f"Success_rate_S_{_segments}": suc_rate, f"Mean_D[time]_S_{_segments}": ep_time})




        
            model.train()
            wandb_step += 1 
            if wwandb:
                wandb.log({"checkpoint_step": wandb_step})
            torch.save(model.state_dict(), ckpt_path + '_save' + '_KTD.pth')

    return model, wandb_step, optimizer, scheduler, raw_model, epochs_counter
