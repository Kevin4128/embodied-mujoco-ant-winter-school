import os
import random
import time
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. CONFIGURATION (EDIT THESE PATHS)
# ==============================================================================
# Path to the 'agent.pt' file inside your 'runs' folder. 
# Example: "runs/sim_embodied_ant__123456/agent.pt"
CHECKPOINT_PATH = "runs/YOUR_EXP_FOLDER_NAME/agent.pt" 

# Your XML Model Path
MODEL_PATH = "../../sim/assets/embodied_mujoco_ant_higher_fidelity.xml"
ENV_ID = "SimEmbodiedAnt"

# Set to True to save a video, False to watch a live window
SAVE_VIDEO = True
VIDEO_FOLDER = "playback_videos"

# ==============================================================================
# 2. PASTE YOUR ACTOR CLASS HERE
# ==============================================================================
# You must copy the "class Actor(nn.Module):" section from your 
# sac_cleanrl.py file and paste it here. It usually looks like this:

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# --- PASTE ACTOR CLASS BELOW THIS LINE ---
class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        # NOTE: Make sure these array sizes match what you trained with!
        # If your training script calculated these dynamically, you might need
        # to hardcode them or load the env first.
        # usually: np.array(env.single_observation_space.shape).prod()
        # and: np.array(env.single_action_space.shape).prod()
        
        # EXAMPLE STRUCTURE (Adjust to match your sac_cleanrl.py):
        # self.fc1 = layer_init(nn.Linear(np.array(env.single_observation_space.shape).prod(), 256))
        # self.fc2 = layer_init(nn.Linear(256, 256))
        # self.fc_mean = layer_init(nn.Linear(256, np.array(env.single_action_space.shape).prod()), std=0.01)
        # self.fc_logstd = layer_init(nn.Linear(256, np.array(env.single_action_space.shape).prod()), std=0.01)
        pass 

    def get_action(self, x):
        # PASTE THE METHOD FROM YOUR SCRIPT
        pass
# -----------------------------------------


# ==============================================================================
# 3. PLAYBACK LOOP
# ==============================================================================
if __name__ == "__main__":
    # Import your custom environment registration if needed
    # from my_custom_env_file import SimEmbodiedAnt 
    
    # Create the environment
    env = gym.make(
        ENV_ID, 
        model_path=MODEL_PATH, 
        render_mode="rgb_array" if SAVE_VIDEO else "human",
        task_type="go_to_target", 
        target_x=0.0, 
        target_y=0.0
    )

    if SAVE_VIDEO:
        env = gym.wrappers.RecordVideo(env, VIDEO_FOLDER)

    # Initialize Agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = Actor(env).to(device)
    
    # Load the trained weights
    print(f"Loading model from {CHECKPOINT_PATH}...")
    actor.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    actor.eval()

    obs, _ = env.reset()
    done = False
    
    print("Starting playback...")
    while not done:
        # Get action from model
        with torch.no_grad():
            obs = torch.Tensor(obs).to(device)
            # Depending on your Actor definition, it might return (mean, logstd) or just action
            # Usually in CleanRL evaluation we just want the mean (deterministic)
            # If your get_action returns sampled action, try to find the 'get_mean' or similar if it exists
            # Otherwise, standard get_action is fine for a quick check.
            action, _, _ = actor.get_action(obs) 
            action = action.cpu().numpy()

        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        obs = next_obs
        
        if not SAVE_VIDEO:
            env.render()
            time.sleep(0.05) # Slow down slightly for human viewing

    env.close()
    print("Done! Video saved." if SAVE_VIDEO else "Done!")