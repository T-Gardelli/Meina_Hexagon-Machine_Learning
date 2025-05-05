import os
import time
import numpy as np
import threading
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor

# --- AMP Import ---
from torch.cuda.amp import autocast

# --- Game env and Configs ---
from game_env import SuperHexagonEnv
import config

# --- Setup Directories ---
# Use paths from config module
if hasattr(config, 'LOG_DIR') and hasattr(config, 'MODEL_SAVE_DIR'):
    LOG_DIR = config.LOG_DIR
    MODEL_SAVE_DIR = config.MODEL_SAVE_DIR
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
else:
    print("Error: config.py must define LOG_DIR and MODEL_SAVE_DIR")
    exit()

# --- Configuration ---
USE_FEATURE_EXTRACTOR_AMP = True # Set to False to disable AMP in the extractor

# --- Setup Directories ---
os.makedirs(config.LOG_DIR, exist_ok=True)
os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)


# --- SimpleCNN with BatchNorm and basic AMP support ---
class SimpleCNN(BaseFeaturesExtractor):
    """
    Custom CNN with BatchNorm and optional AMP autocast in forward pass.
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
    :param use_amp: (bool) Whether to enable autocast for the forward pass (requires CUDA).
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 128, use_amp: bool = False):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        print(f"CustomCNN: Input channels={n_input_channels}")

        # Check if AMP is requested and CUDA is available
        self.enable_amp = use_amp and torch.cuda.is_available()
        if use_amp and not torch.cuda.is_available():
            print("Warning: AMP requested but CUDA is not available. AMP disabled.")
        elif self.enable_amp:
            print("CustomCNN: Automatic Mixed Precision (autocast) enabled for forward pass.")

        # Define the convolutional layers with BatchNorm
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=8, stride=4, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # Maybe remove the third Conv layer compared to Nature DQN
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            dummy_input = torch.as_tensor(observation_space.sample()[None]).float()
            # Use autocast during shape calculation if enabled, matching forward pass
            with autocast(enabled=self.enable_amp):
                 n_flatten = self.cnn(dummy_input).shape[1]
            print(f"CustomCNN: Flattened features before final linear layer={n_flatten}")

        # Define the final linear layer before outputting features
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Use autocast context manager for AMP if enabled
        # This applies mixed precision only to this feature extractor's operations
        with autocast(enabled=self.enable_amp):
            features = self.cnn(observations)
            output = self.linear(features)
        # Ensure output is float32 for compatibility with subsequent layers in SB3
        # unless full AMP is implemented throughout the model.
        return output.float() # Cast back to float32 for safety with standard SB3 DQN head


# Asynchronous Checkpoint Callback (Threading - Generally OK for I/O)
class AsyncCheckpointCallback(CheckpointCallback):
    """ CheckpointCallback that saves the model in a background thread """
    def _save_model_async(self, save_path: str):
        try:
            # Ensure the path exists before saving
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            self.model.save(save_path)
            print(f"Asynchronously saved model checkpoint to {save_path}.zip")
        except Exception as e:
            print(f"Error saving model asynchronously: {e}")

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            # Determine save paths
            model_path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps")
            buffer_path = os.path.join(self.save_path, f"{self.name_prefix}_replay_buffer_{self.num_timesteps}_steps.pkl") if self.save_replay_buffer else None
            vecnormalize_path = os.path.join(self.save_path, f"{self.name_prefix}_vecnormalize_{self.num_timesteps}_steps.pkl") if self.save_vecnormalize else None

            print(f"Triggering asynchronous checkpoint save (Timestep {self.num_timesteps})...")
            thread = threading.Thread(
                target=self._save_model_async,
                args=(model_path,)
            )
            thread.daemon = True
            thread.start()

            # Synchronous saving for buffer/vecnormalize
            if self.save_replay_buffer and hasattr(self.model, 'save_replay_buffer') and self.model.replay_buffer is not None and buffer_path:
                try:
                    print(f"Saving replay buffer synchronously to {buffer_path}...")
                    self.model.save_replay_buffer(buffer_path)
                except Exception as e:
                    print(f"Error saving replay buffer synchronously: {e}")
            if self.save_vecnormalize and hasattr(self.training_env, "save") and vecnormalize_path:
                try:
                    print(f"Saving VecNormalize stats synchronously to {vecnormalize_path}...")
                    self.training_env.save(vecnormalize_path)
                except Exception as e:
                    print(f"Error saving VecNormalize stats synchronously: {e}")
        return True

#
# Asynchronous Evaluation Callback (Threading - May contend for resources)
class AsyncEvalCallback(EvalCallback):
    """ EvalCallback that runs evaluation in a background thread """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_evaluating = False

    def _evaluate_async(self):
        try:
            mean_reward, std_reward = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(mean_reward)
                ep_length = -1
                if hasattr(self.eval_env, 'get_episode_lengths'):
                    lengths = self.eval_env.get_episode_lengths()
                    if len(lengths) > 0:
                        ep_length = np.mean(lengths[-self.n_eval_episodes:]) if len(lengths) >= self.n_eval_episodes else np.mean(lengths)
                self.evaluations_length.append(ep_length)

                kwargs = {}
                if hasattr(self, 'evaluations_successes') and self.evaluations_successes:
                    kwargs = {"successes": self.evaluations_successes}

                print(f"Saving evaluation results to {self.log_path}.npz")
                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            print(f"Async Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")

            if self.model.logger:
                self.model.logger.record("eval/mean_reward", float(mean_reward))
                if hasattr(self, 'evaluations_successes') and self.evaluations_successes:
                    self.model.logger.record("eval/success_rate", np.mean(self.evaluations_successes))
                self.model.logger.record("eval/std_reward", float(std_reward))
                if ep_length != -1:
                    self.model.logger.record("eval/mean_ep_length", float(ep_length))
                self.model.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                print(f"Async Eval: New best mean reward! {mean_reward:.2f} > {self.best_mean_reward:.2f}")
                self.best_mean_reward = mean_reward
                if self.best_model_save_path is not None:
                    os.makedirs(os.path.dirname(self.best_model_save_path), exist_ok=True)
                    print(f"Saving best model asynchronously to {self.best_model_save_path}.zip")
                    thread = threading.Thread(target=self.model.save, args=(self.best_model_save_path,))
                    thread.daemon = True
                    thread.start()

            self.is_evaluating = False

        except Exception as e:
            print(f"Async evaluation encountered an error: {e}")
            self.is_evaluating = False # Ensure lock is released on error

    def _on_step(self) -> bool:
        continue_training = True
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0 and not self.is_evaluating:
            self.is_evaluating = True
            print(f"\nTriggering asynchronous evaluation (Timestep {self.num_timesteps})...")
            eval_thread = threading.Thread(target=self._evaluate_async)
            eval_thread.daemon = True
            eval_thread.start()

        return continue_training
    
    # --- Environment Creation Function (MODIFIED FOR MONITOR) ---
    def create_env(render_mode_env=None, is_eval=False):
        """Creates and wraps the SuperHexagon environment."""
        env = SuperHexagonEnv(render_mode=render_mode_env)

        # --- Apply Monitor Wrapper ---
        # Apply Monitor always for accurate episode statistics tracking by evaluate_policy.
        # Disable CSV file logging for evaluation environments by setting filename=None.
        monitor_filename = None # Default: no file logging
        if not is_eval:
            # Construct unique path for training monitor log file
            monitor_log_path = os.path.join(LOG_DIR, f"monitor_train_{time.time_ns()}.csv")
            # Ensure the directory for the monitor file exists
            os.makedirs(os.path.dirname(monitor_log_path), exist_ok=True)
            monitor_filename = monitor_log_path # Enable file logging for training env

        # Wrap with Monitor.
        env = Monitor(env, filename=monitor_filename, info_keywords=("is_success",)) # Add other keywords if needed

        return env

#
# Main Training Script
#

if __name__ == "__main__":
    # --- Create Environment Function ---
    def create_env(render_mode_env=None, is_eval=False):
        env = SuperHexagonEnv(render_mode=render_mode_env)
        if not is_eval:
            # Wrap training env with Monitor
            log_dir = os.path.join(config.LOG_DIR, f"monitor_{time.time()}") # Unique monitor log dir
            os.makedirs(log_dir, exist_ok=True)
            env = Monitor(env, log_dir)
        return env

    N_ENVS = 1 # Keep as 1 based on original script's constraint
    print(f"Creating {N_ENVS} vectorized training environment(s)...")
    env_fns = [lambda: create_env(render_mode_env=None, is_eval=False) for _ in range(N_ENVS)]
    # Consider SubprocVecEnv if N_ENVS > 1 becomes possible
    env = DummyVecEnv(env_fns)

    # --- Apply VecTransposeImage Wrapper ---
    print("Applying VecTransposeImage wrapper to training env...")
    env = VecTransposeImage(env)
    print(f"Final Training Env observation space: {env.observation_space.shape} {env.observation_space.dtype}")

    # --- Create Evaluation Environment (Correctly Wrapped) ---
    print("Creating evaluation environment...")
    eval_env_fns = [lambda: create_env(render_mode_env=None, is_eval=True) for _ in range(N_ENVS)] # Use same N_ENVS for consistency? Often 1 is enough for eval.
    eval_env = DummyVecEnv(eval_env_fns) # Typically use DummyVecEnv for evaluation
    print("Applying VecTransposeImage wrapper to evaluation env...")
    eval_env = VecTransposeImage(eval_env) # Must wrap eval env identically regarding observations
    print(f"Final Eval Env observation space: {eval_env.observation_space.shape} {eval_env.observation_space.dtype}")


    # --- Define Asynchronous Callbacks ---
    checkpoint_callback = AsyncCheckpointCallback(
        save_freq=max(100000 // N_ENVS, 1),
        save_path=config.MODEL_SAVE_DIR,
        name_prefix=config.MODEL_FILENAME_BASE + "_ckpt",
        save_replay_buffer=False,
        save_vecnormalize=False,
        verbose=1
    )

    eval_callback = AsyncEvalCallback(
        eval_env, # The wrapped evaluation environment
        best_model_save_path=os.path.join(config.MODEL_SAVE_DIR, 'best_model'),
        log_path=os.path.join(config.MODEL_SAVE_DIR, 'eval_logs'),
        eval_freq=max(50000 // N_ENVS, 1),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        warn=False,
        verbose=1
    )

    callback_list = [checkpoint_callback, eval_callback]

    # --- Define the DQN Model ---
    CUSTOM_FEATURES_DIM = 128

    # --- policy_kwargs to enable AMP in extractor ---
    policy_kwargs = dict(
        features_extractor_class=SimpleCNN,
        features_extractor_kwargs=dict(
            features_dim=CUSTOM_FEATURES_DIM,
            use_amp=USE_FEATURE_EXTRACTOR_AMP # Pass the flag here
        ),
        # net_arch=[64] # Example customization
    )

    # Determine device explicitly for potential later use (e.g., full AMP scaler)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if USE_FEATURE_EXTRACTOR_AMP and device.type != 'cuda':
         print("Warning: AMP requested for feature extractor, but device is CPU. AMP will not be active.")

    model = DQN(
        'CnnPolicy',
        env,
        verbose=1,
        tensorboard_log=config.LOG_DIR,
        buffer_size=150000,
        learning_rate=1e-4,
        batch_size=64,
        learning_starts=20000,
        target_update_interval=5000,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.1,
        train_freq=(4, "step"),
        gradient_steps=4,
        gamma=0.99,
        device=device,
        optimize_memory_usage=True,
        replay_buffer_kwargs=dict(handle_timeout_termination=False),
        policy_kwargs=policy_kwargs # Pass the custom arguments
    )
    print("DQN Model Defined using Custom Policy Kwargs (BatchNorm, optional Extractor AMP).")
    # print(model.policy) # Print the structure to verify

    # --- Load Existing Model (Optional) ---
    load_model_path = None # Example: os.path.join(config.MODEL_SAVE_DIR, "best_model.zip")
    reset_num_timesteps = True
    if load_model_path and os.path.exists(load_model_path):
        print(f"Loading existing model from: {load_model_path}")
        # Pass custom_objects if the loaded model used the modified SimpleCNN
        # If the saved model definition *exactly* matches the current SimpleCNN definition,
        # SB3 might load it automatically. However, explicitly passing it is safer
        # when custom classes are involved, especially if they changed.
        custom_objects = {
            "policy_kwargs": policy_kwargs,
        }
        model = DQN.load(
            load_model_path,
            env=env,
            tensorboard_log=config.LOG_DIR,
            device=device, # Ensure device consistency
            custom_objects=custom_objects # Pass custom policy/extractor info
         )
        reset_num_timesteps = False
        buffer_load_path = load_model_path.replace(".zip", "_replay_buffer.pkl")
        if os.path.exists(buffer_load_path):
            try:
                print(f"Loading replay buffer from {buffer_load_path}")
                model.load_replay_buffer(buffer_load_path)
            except Exception as e:
                print(f"Warning: Could not load replay buffer: {e}")
        print("Model and potentially buffer loaded.")

    # --- Start Training ---
    TOTAL_TIMESTEPS = 15_000_000

    print("\n" + "="*40)
    print("!!! Ensure Super Hexagon is running and the window is ACTIVE !!!")
    print("!!! Ensure NumLock is OFF (if using numpad keys) !!!")
    print(f"Starting/Resuming training for DQN method, {TOTAL_TIMESTEPS} total timesteps...")
    print(f"Logging to TensorBoard: {config.LOG_DIR}")
    print(f"Saving models to: {config.MODEL_SAVE_DIR}")
    if USE_FEATURE_EXTRACTOR_AMP and device.type == 'cuda':
         print("Feature Extractor AMP (autocast) is ENABLED.")
    print("Training will begin in 5 seconds...")
    print("="*40 + "\n")
    time.sleep(5)

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callback_list,
            log_interval=10,
            tb_log_name=config.MODEL_FILENAME_BASE,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            final_model_path = os.path.join(config.MODEL_SAVE_DIR, f"{config.MODEL_FILENAME_BASE}_final")
            print(f"\nSaving final model to: {final_model_path}.zip")
            model.save(final_model_path)
        except Exception as e:
            print(f"Error saving final model: {e}")

        print(f"\nBest performing model during evaluation potentially saved in: {os.path.join(config.MODEL_SAVE_DIR, 'best_model.zip')}")
        print("Closing environments...")
        try:
            if 'env' in locals() and env is not None:
                env.close()
        except Exception as e:
            print(f"Error closing training environment: {e}")
        try:
             if 'eval_env' in locals() and eval_env is not None:
                 eval_env.close()
        except Exception as e:
             print(f"Error closing evaluation environment: {e}")
        print("Environments closed.")

    print("\n--- DQN Training Complete or Stopped ---")
    print(f"To view logs, run: tensorboard --logdir={config.LOG_DIR}")