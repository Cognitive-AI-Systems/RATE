import datetime
import wandb
from torch.utils.data import random_split, DataLoader
import argparse
import yaml
import sys

import os
import sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from TMaze_new.TMaze_new_src.utils import set_seed, get_intro, TMaze_data_generator, CombinedDataLoader
from TMaze_new.TMaze_new_src.train import train

# python3 TMaze_new/TMaze_new_src/train_tmaze.py --model_mode 'RATE' --arch_mode 'TrXL' --curr 'true' --ckpt_folder 'RATE_max_3' --max_n_final 3 --text 'RATE_max_3'

os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1"  
os.environ["OMP_NUM_THREADS"] = "1" 

with open("wandb_config.yaml") as f:
    wandb_config = yaml.load(f, Loader=yaml.FullLoader)
os.environ['WANDB_API_KEY'] = wandb_config['wandb_api']

with open("TMaze_new/TMaze_new_src/config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    
def create_args():
    parser = argparse.ArgumentParser(description='Description of your program')

    parser.add_argument('--model_mode',     type=str, default='RATE',  help='Model training mode. Available variants: "DT, DTXL, RATE (Ours), RATEM (RMT)"')    
    parser.add_argument('--arch_mode',      type=str, default='TrXL',  help='Model architecture mode. Available variants: "TrXL", "TrXL-I", "GTrXL"')
    parser.add_argument('--min_n_final',    type=int, default=1,       help='Start number of considered segments during training')
    parser.add_argument('--max_n_final',    type=int, default=3,       help='End number of considered segments during training')
    parser.add_argument('--start_seed',     type=int, default=1,       help='Start seed')
    parser.add_argument('--end_seed',       type=int, default=10,       help='End seed')
    parser.add_argument('--curr',           type=str, default='true',  help='Curriculum mode. If "true", then curriculum will be used during training')
    parser.add_argument('--ckpt_folder',    type=str, default='ckpt',  help='Checkpoints directory')
    parser.add_argument('--text',           type=str, default='',      help='Short text description of rouns group')

    parser.add_argument('--nmt',       type=int, default=5,       help='')
    parser.add_argument('--mem_len',       type=int, default=2,       help='')
    parser.add_argument('--n_head_ca',       type=int, default=2,       help='')
    parser.add_argument('--mrv_act',       type=str, default='relu',       help='["no_act", "relu", "leaky_relu", "elu", "tanh"]')

    return parser

if __name__ == '__main__':
    get_intro()
    
    args = create_args().parse_args()

    model_mode = args.model_mode
    start_seed = args.start_seed
    end_seed = args.end_seed
    arch_mode = args.arch_mode
    curr = args.curr
    min_n_final = args.min_n_final
    max_n_final = args.max_n_final
    ckpt_folder = args.ckpt_folder
    TEXT_DESCRIPTION = args.text
    mem_len = args.mem_len
    nmt = args.nmt
    n_head_ca = args.n_head_ca
    mrv_act = args.mrv_act

    SEGMENT_LENGTH = config["training_config"]["context_length"]

    config["training_config"]["max_segments"] = max_n_final
    config["online_inference_config"]["episode_timeout"] = max_n_final*SEGMENT_LENGTH
    config["online_inference_config"]["corridor_length"] = max_n_final*SEGMENT_LENGTH-2
    config["training_config"]["sections"] = max_n_final
    config["model_mode"] = model_mode
    config["training_config"]["curriculum"] = curr
    config["arctitecture_mode"] = arch_mode

    for RUN in range(start_seed, end_seed+1):
        set_seed(RUN)

        """ ARCHITECTURE MODE """
        if config["arctitecture_mode"] == "TrXL":
            config["model_config"]["use_gate"] = False
            config["model_config"]["use_stable_version"] = False
        elif config["arctitecture_mode"] == "TrXL-I":
            config["model_config"]["use_gate"] = False
            config["model_config"]["use_stable_version"] = True
        elif config["arctitecture_mode"] == "GTrXL":
            config["model_config"]["use_gate"] = True
            config["model_config"]["use_stable_version"] = True     

        print(f"Selected Architecture: {config['arctitecture_mode']}")  

        """ MODEL MODE """
        if config["model_mode"] == "RATE": 
            config["model_config"]["mem_len"] = mem_len
            config["model_config"]["mem_at_end"] = True
            config["model_config"]["num_mem_tokens"] = nmt
            config["model_config"]["n_head_ca"] = n_head_ca
            config["model_config"]["mrv_act"] = mrv_act

        elif config["model_mode"] == "DT":
            config["model_config"]["mem_len"] = 0
            config["model_config"]["mem_at_end"] = False
            config["model_config"]["num_mem_tokens"] = 0
            config["model_config"]["n_head_ca"] = 0
            config["training_config"]["context_length"] = config["training_config"]["context_length"] * config["training_config"]["sections"]
            config["training_config"]["sections"] = 1
            max_length = config["training_config"]["context_length"]

        elif config["model_mode"] == "DTXL":
            config["model_config"]["mem_len"] = mem_len
            config["model_config"]["mem_at_end"] = False
            config["model_config"]["num_mem_tokens"] = 0
            config["model_config"]["n_head_ca"] = 0
            config["training_config"]["context_length"] = config["training_config"]["context_length"] * config["training_config"]["sections"]
            config["training_config"]["sections"] = 1
            max_length = config["training_config"]["context_length"]

        elif config["model_mode"] == "RATEM":
            config["model_config"]["mem_len"] = 0
            config["model_config"]["mem_at_end"] = True
            config["model_config"]["num_mem_tokens"] = nmt
            config["model_config"]["n_head_ca"] = n_head_ca
            config["model_config"]["mrv_act"] = mrv_act
            max_length = config["training_config"]["sections"]*config["training_config"]["context_length"]

        elif config["model_mode"] == "RATE_wo_nmt":
            print("Custom Mode!!! RATE wo nmt")
            config["model_config"]["mem_len"] = mem_len
            config["model_config"]["mem_at_end"] = False
            config["model_config"]["num_mem_tokens"] = 0
            config["model_config"]["n_head_ca"] = 0
            max_length = config["training_config"]["sections"]*config["training_config"]["context_length"]
        
        if nmt == 0:
            config["model_config"]["mem_at_end"] = False

        print(f"Selected Model: {config['model_mode']}")  

        mini_text = f"arch_mode_{config['arctitecture_mode']}"
        now = datetime.datetime.now()
        date_time = now.strftime("%Y-%m-%d_%H-%M-%S").replace('-', '_')
        group = f'{TEXT_DESCRIPTION}_{mini_text}_{config["model_mode"]}_min_{min_n_final}_max_{max_n_final}'
        name = f'{TEXT_DESCRIPTION}_{mini_text}_{config["model_mode"]}_min_{min_n_final}_max_{max_n_final}_RUN_{RUN}_{date_time}'
        current_dir = os.getcwd()
        current_folder = os.path.basename(current_dir)
        ckpt_path = f'../{current_folder}/TMaze_new/TMaze_new_checkpoints/{ckpt_folder}/{name}/'
        isExist = os.path.exists(ckpt_path)
        if not isExist:
            os.makedirs(ckpt_path)

        if config["wandb_config"]["wwandb"]:
            run = wandb.init(project=config['wandb_config']['project_name'], name=name, group=group, config=config, save_code=True, reinit=True) #entity="RATE"

        TMaze_data_generator(max_segments=config["training_config"]["max_segments"], multiplier=config["data_config"]["multiplier"], 
                             hint_steps=config["data_config"]["hint_steps"], desired_reward=config["data_config"]["desired_reward"], 
                             win_only=config["data_config"]["win_only"], segment_length=SEGMENT_LENGTH)

        wandb_step = 0
        epochs_counter = 0
        model, optimizer, scheduler, raw_model = None, None, None, None
        prev_ep = None
        
        if config["training_config"]["curriculum"].lower() == 'true':
            print("MODE: CURRICULUM")
            for n_final in range(min_n_final, max_n_final+1):

                n_fin = n_final
                
                if config["model_mode"] != "DT" and config["model_mode"] != "DTXL":
                    config["training_config"]["sections"] = n_final
                else:
                    config["training_config"]["sections"] = 1

                combined_dataloader = CombinedDataLoader(n_init=min_n_final, 
                                                         n_final=n_fin, 
                                                         multiplier=config["data_config"]["multiplier"], 
                                                         hint_steps=config["data_config"]["hint_steps"], 
                                                         batch_size=config["training_config"]["batch_size"],
                                                         mode="", 
                                                         cut_dataset=config["data_config"]["cut_dataset"], 
                                                         desired_reward=config["data_config"]["desired_reward"], 
                                                         win_only=config["data_config"]["win_only"],
                                                         segment_length=SEGMENT_LENGTH)

                # Split dataset into train and validation sets
                full_dataset = combined_dataloader.dataset
                train_size = int(0.8 * len(full_dataset))
                val_size = len(full_dataset) - train_size
                train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

                # Use DataLoader to load the datasets in parallel
                train_dataloader = DataLoader(train_dataset, batch_size=config["training_config"]["batch_size"], shuffle=True, num_workers=4)
                val_dataloader = DataLoader(val_dataset, batch_size=config["training_config"]["batch_size"], shuffle=True, num_workers=4)
                print(f"Number of considered segments: {n_final}, dataset length: {len(combined_dataloader.dataset)}, Train: {len(train_dataset)}, Val: {len(val_dataset)}")
                del full_dataset
                del train_dataset
                del val_dataset
                new_segment = True
                model, wandb_step, optimizer, scheduler, raw_model, epochs_counter = train(model, optimizer, scheduler, 
                                                                        raw_model, new_segment, epochs_counter, n_final, wandb_step, ckpt_path, config,
                                                                        train_dataloader, val_dataloader, max_n_final)
                del train_dataloader
                del val_dataloader
                
        elif config["training_config"]["curriculum"].lower() == 'false':
            print("MODE: CLASSIC")
            
            n_fin = max_n_final
            
            if config["model_mode"] != "DT" and config["model_mode"] != "DTXL":
                config["training_config"]["sections"] = max_n_final
            else:
                config["training_config"]["sections"] = 1

            combined_dataloader = CombinedDataLoader(n_init=min_n_final, 
                                                     n_final=n_fin, 
                                                     multiplier=config["data_config"]["multiplier"], 
                                                     hint_steps=config["data_config"]["hint_steps"], 
                                                     batch_size=config["training_config"]["batch_size"], 
                                                     mode="", 
                                                     cut_dataset=config["data_config"]["cut_dataset"], 
                                                     one_mixed_dataset=True, 
                                                     desired_reward=config["data_config"]["desired_reward"], 
                                                     win_only=config["data_config"]["win_only"],
                                                     segment_length=SEGMENT_LENGTH)
            
            # Split dataset into train and validation sets
            full_dataset = combined_dataloader.dataset
            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

            # Use DataLoader to load the datasets in parallel
            train_dataloader = DataLoader(train_dataset, batch_size=config["training_config"]["batch_size"], shuffle=True, num_workers=4)
            val_dataloader = DataLoader(val_dataset, batch_size=config["training_config"]["batch_size"], shuffle=True, num_workers=4)
            print(f"Number of considered segments: {max_n_final}, dataset length: {len(combined_dataloader.dataset)}, Train: {len(train_dataset)}, Val: {len(val_dataset)}")
            del full_dataset
            del train_dataset
            del val_dataset
            new_segment = True
            model, wandb_step, optimizer, scheduler, raw_model, epochs_counter = train(model, optimizer, scheduler, 
                                                                    raw_model, new_segment, epochs_counter, max_n_final, wandb_step, ckpt_path, config,
                                                                    train_dataloader, val_dataloader, max_n_final)
            del train_dataloader
            del val_dataloader
        
        if config["wandb_config"]["wwandb"]:
            run.finish()







