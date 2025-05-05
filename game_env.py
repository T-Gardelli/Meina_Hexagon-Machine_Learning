import gymnasium as gym
from gymnasium import spaces
from collections import deque
import numpy as np
import cv2
import threading
import math
import time
import config
from typing import Tuple, Dict, Any, Optional, List # Added for type hinting
# import traceback # for debugging deep issues

from screen_capture import grab_screen
from image_processor import process_frame
from action_handler import press_left, press_right, press_space, release_keys

class SuperHexagonEnv(gym.Env):
    """ Custom Gymnasium Environment for Super Hexagon optimized for DQN.

    Observation Space: (H, W, C) NumPy array where C = FRAME_STACK (Edges Only).
                     VecTransposeImage wrapper should be used in training for PyTorch (channels first).
    Action Space: Discrete(3) - 0: Left, 1: Right, 2: No-op (implicitly handled by releasing keys)
    Reward: Based on survival time and penalty for termination.
    Render Mode ('human'): Displays latest Edges view with stats.
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    # --- Configuration Attributes ---
    GAMEOVER_PATCH_HEIGHT = 50
    GAMEOVER_BLACK_THRESHOLD = 30 # Pixels < 30 = 'black'
    GAMEOVER_WHITE_THRESHOLD = 225 # Pixels > 225 = 'white'
    GAMEOVER_COLOR_RATIO = 0.075

    # Reward parameters
    REWARD_SURVIVAL = 0.2
    PENALTY_TERMINATION = -10.0

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        self.render_mode = render_mode
        self._render_window_name = "Super Hexagon RL View (Edges Only)" # Updated window name
        self.action_space = spaces.Discrete(3)
        self._frame_h, self._frame_w = config.RESIZE_DIM[1], config.RESIZE_DIM[0]

        # --- MODIFICATION: Update number of channels and observation space ---
        self._num_channels = config.FRAME_STACK # Only edge frames
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(self._frame_h, self._frame_w, self._num_channels), # Shape uses updated channel count
            dtype=np.uint8
        )
        print(f"Initialized Env - Observation Space Shape: {self.observation_space.shape}")
        # --------------------------------------------------------------------
        # Internal State
        self._running = True
        self.frame_stack = deque(maxlen=config.FRAME_STACK)
        self._last_raw_frame: Optional[np.ndarray] = None
        self._step_count: int = 0
        self._episode_reward: float = 0.0
        self.current_key: Optional[str] = None

        self._last_edges_render: Optional[np.ndarray] = None # Keep for rendering
        # --------------------------------------------------

        # Background thread for non-blocking key presses
        self._key_thread = threading.Thread(target=self._dispatch_key_events, daemon=True)
        self._key_thread.start()

    def _dispatch_key_events(self) -> None:
        """Background thread to manage key presses/releases asynchronously."""
        while self._running:
            key_to_action = self.current_key
            if key_to_action == 'left':
                press_left()
            elif key_to_action == 'right':
                press_right()
            else:
                release_keys()
            time.sleep(0.01)

    # --- Gather and Process the image ---
    def _get_obs_and_process(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Grabs a screen, processes it for edges, updates frame stack, returns stacked obs.
        Also returns the latest edge channel for rendering.
        Returns (None, None) on failure.
        """
        try:
            raw_frame = grab_screen()
            if raw_frame is None:
                print("Warning: grab_screen returned None.")
                return None, self._last_edges_render # Return potentially stale edge frame

            self._last_raw_frame = raw_frame # Store for game over detection

            # Process frame (now returns H, W, 1 edge image)
            processed_frame = process_frame(raw_frame)

            # Check shape (expecting H, W, 1)
            expected_shape = (self._frame_h, self._frame_w, 1)
            if processed_frame.shape != expected_shape:
                print(f"Warning: process_frame returned unexpected shape {processed_frame.shape}. Expected {expected_shape}.")
                return None, self._last_edges_render # Return potentially stale edge frame

            # Store latest edges for rendering (squeeze to make it 2D H,W for OpenCV functions)
            self._last_edges_render = np.squeeze(processed_frame, axis=-1).copy()

            self.frame_stack.append(processed_frame) # Append (H, W, 1) frame

            # Pad stack if needed
            while len(self.frame_stack) < config.FRAME_STACK:
                 self.frame_stack.append(processed_frame)

            if not self.frame_stack:
                print("Error: Frame stack is unexpectedly empty after processing.")
                return None, self._last_edges_render

            obs_list = list(self.frame_stack)
            # Check shape of frames in deque (expecting H, W, 1)
            if any(f.shape != expected_shape for f in obs_list):
                print("Error: Inconsistent frame shapes in deque.")
                return None, self._last_edges_render

            # Concatenate along the channel axis (axis=-1)
            # Stacks (H, W, 1) frames into (H, W, FRAME_STACK)
            stacked_obs = np.concatenate(obs_list, axis=-1)

            # Ensure final observation shape matches the defined space
            if stacked_obs.shape != self.observation_space.shape:
                print(f"Error: Final stacked observation shape {stacked_obs.shape} != expected {self.observation_space.shape}")
                return None, self._last_edges_render

            # Return observation and the latest 2D edge frame for rendering
            return stacked_obs, self._last_edges_render

        except Exception as e:
            print(f"Error in _get_obs_and_process: {e}")
            # traceback.print_exc()
            # Return None obs, potentially stale edge frame
            return None, self._last_edges_render
    # --- MODIFICATION END ---

    def _get_zero_observation(self) -> np.ndarray:
        """Returns a zero observation matching the observation space."""
        # This automatically uses the updated self.observation_space.shape
        return np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)

    def _detect_game_over(self) -> bool:
        """
        Detects game over by checking for a large black OR white area
        at the bottom of the raw frame.
        """
        if self._last_raw_frame is None:
            # print("Warning: Cannot detect game over, last raw frame is None.")
            return False # Assume not game over if we don't have a frame

        try:
            h, w, _ = self._last_raw_frame.shape
            if h <= 0 or w <= 0: return False # Invalid frame dimensions

            patch_h = min(self.GAMEOVER_PATCH_HEIGHT, h)
            # Ensure patch coordinates are valid
            start_y = max(0, h - patch_h)
            bottom_patch = self._last_raw_frame[start_y : h, :, :]

            if bottom_patch.size == 0: return False

            # Check for mostly black pixels
            black_mask = np.all(bottom_patch < self.GAMEOVER_BLACK_THRESHOLD, axis=-1)
            black_ratio = np.mean(black_mask) if black_mask.size > 0 else 0

            # Check for mostly white pixels
            white_mask = np.all(bottom_patch > self.GAMEOVER_WHITE_THRESHOLD, axis=-1)
            white_ratio = np.mean(white_mask) if white_mask.size > 0 else 0

            # Game is over if either ratio exceeds the threshold
            is_game_over = (black_ratio > self.GAMEOVER_COLOR_RATIO and
                            white_ratio > self.GAMEOVER_COLOR_RATIO)

            # Optional: Print ratios for debugging threshold values
            # if is_game_over:
            #print(f"Game Over Detected: Black Ratio={black_ratio:.3f}, White Ratio={white_ratio:.3f}")

            return is_game_over

        except Exception as e:
             print(f"Error during game over detection: {e}")
             # import traceback
             # traceback.print_exc()
             return False # Assume not game over on error

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Resets the environment for a new episode. Ensures game restarts
        if a game over screen is detected.
        """
        super().reset(seed=seed)

        # Reset internal state
        release_keys() # Release any held keys first
        self.current_key = None
        self._step_count = 0
        self._episode_reward = 0.0
        self.frame_stack.clear()
        self._last_raw_frame = None
        self._last_edges_render = None
        print("--- Env Reset Triggered ---")

        # --- MODIFIED: Robust Restart Game Logic ---
        RESET_RETRY_DELAY = 0.15 # Seconds to wait after pressing space
        RESET_MAX_RETRIES = 30   # Max attempts to restart (~4.5 seconds)
        retries = 0
        initial_game_over_detected = False

        while retries < RESET_MAX_RETRIES:
            # Grab a fresh frame
            self._last_raw_frame = grab_screen()

            if self._last_raw_frame is None:
                print("Warning: grab_screen failed during reset loop. Retrying...")
                time.sleep(0.1) # Wait a bit longer if screen grab fails
                retries += 1
                continue

            # Check if game is over using the updated detection logic
            if self._detect_game_over():
                if not initial_game_over_detected:
                     print("Game over detected on reset. Attempting restart loop...")
                     initial_game_over_detected = True # Only print the first time

                #print(f"Restart attempt {retries + 1}/{RESET_MAX_RETRIES}...")
                press_space() # Press space to restart
                time.sleep(RESET_RETRY_DELAY) # Wait for game to react/screen update
                release_keys() # Release space briefly (might help some games)
                time.sleep(0.05) # Short pause after release
                retries += 1
            else:
                # Game over screen is gone (or wasn't there initially)
                if initial_game_over_detected:
                    print("Game restart successful.")
                else:
                    print("No game over detected, proceeding.")
                break # Exit the restart loop
        else:
            # Loop finished without breaking (max retries reached)
            print(f"Error: Failed to restart game after {RESET_MAX_RETRIES} attempts.")
            # Handle failure: return a zero observation, log error, potentially raise exception
            # For now, return zero observation so training doesn't crash immediately
            # but logs should show this episode start failed.
            initial_obs = self._get_zero_observation()
             # Ensure frame stack is populated with zeros if reset failed early
            zero_processed = np.zeros((self._frame_h, self._frame_w, 1), dtype=np.uint8)
            while len(self.frame_stack) < config.FRAME_STACK:
                self.frame_stack.append(zero_processed)
            if self._last_edges_render is None:
                self._last_edges_render = np.squeeze(zero_processed, axis=-1)
            return initial_obs, {"reset_failed": True} # Add info about failure

        # --- Initialize Observation Stack (Only if restart loop succeeded) ---
        print("Initializing observation stack...")
        initial_obs = None
        init_attempts = 0
        # Need to grab frames until the stack is full
        while len(self.frame_stack) < config.FRAME_STACK and init_attempts < config.FRAME_STACK * 3 : # Limit attempts
            obs_fill, edges_fill = self._get_obs_and_process() # _g_o_a_p appends to stack
            if obs_fill is not None:
                 initial_obs = obs_fill # Keep the latest valid stacked observation
                 # Check stack size inside _g_o_a_p ensures it fills, rely on obs_fill not being None
            else:
                 print("Warning: Failed to get observation during stack initialization.")
                 time.sleep(0.02) # Small delay if grab/process fails
            init_attempts +=1


        # If loop finished without getting any valid observation OR stack not full
        if initial_obs is None or len(self.frame_stack) < config.FRAME_STACK:
            print("Error: Could not initialize observation stack after reset. Returning zero observation.")
            initial_obs = self._get_zero_observation()
            zero_processed = np.zeros((self._frame_h, self._frame_w, 1), dtype=np.uint8)
            # Ensure stack is full even on failure
            while len(self.frame_stack) < config.FRAME_STACK:
                self.frame_stack.append(zero_processed)
            if self._last_edges_render is None:
                 self._last_edges_render = np.squeeze(zero_processed, axis=-1)
            # Consider adding info={"init_failed": True} here too
            return initial_obs, {"init_failed": True}


        print("--- Env Reset Complete ---")
        return initial_obs, {} # Return observation and empty info dict on success


    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one time step within the environment."""
        self._step_count += 1

        # --- Action Handling ---
        action_map = {0: 'left', 1: 'right', 2: None}
        target_key_state = action_map.get(action)
        if self.current_key != target_key_state:
             self.current_key = target_key_state

        # --- Get Observation & Update Render Cache ---
        observation, last_edges_for_render = self._get_obs_and_process() # last_edges_for_render is now 2D H,W
        # ------------------------------------

        # --- Handle observation failure ---
        if observation is None:
            print(f"Warning: Failed to get observation in step {self._step_count}. Ending episode.")
            terminated = True
            truncated = False
            reward = self.PENALTY_TERMINATION
            # Use the zero observation matching the *new* observation space shape
            observation = self._get_zero_observation()
            info = {'error': 'Observation failed'}
            info['episode'] = {"r": self._episode_reward + reward, "l": self._step_count}
            self._episode_reward += reward
            # Render potentially stale edge frame if human mode active
            if self.render_mode == 'human':
                self._render_frame() # _render_frame uses self._last_edges_render
            return observation, reward, terminated, truncated, info
        # ---------------------------------------

        # --- Game Over Check ---
        terminated = self._detect_game_over() # Uses self._last_raw_frame

        # --- Calculate Reward ---
        reward = self.PENALTY_TERMINATION if terminated else self.REWARD_SURVIVAL
        self._episode_reward += reward

        # --- Prepare Info Dictionary ---
        info = {}
        truncated = False
        if terminated or truncated:
            info['episode'] = {"r": self._episode_reward, "l": self._step_count}

        # --- Render (if requested) ---
        if self.render_mode == 'human':
            # Render uses the cached self._last_edges_render
            self._render_frame()

        return observation, reward, terminated, truncated, info

    # --- Rendering Method (For Edges Only View) ---
    def _get_render_frame(self) -> Optional[np.ndarray]:
        """
        Creates the simplified debug view frame (Edges Only, Scaled).
        Uses the cached _last_edges_render (which is 2D).
        Returns None if cached frame is unavailable.
        """
        if self._last_edges_render is None:
            # Return a placeholder (half the previous width)
            placeholder_h = self._frame_h * 2 # Scaled height
            placeholder_w = self._frame_w * 2 # Scaled width for one frame
            placeholder = np.zeros((placeholder_h, placeholder_w, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Waiting for frames...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            return placeholder

        try:
            # Convert edges (already 2D) to BGR for display
            edges_bgr = cv2.cvtColor(self._last_edges_render, cv2.COLOR_GRAY2BGR)

            # Resize for better visibility (optional)
            scale_factor = 2
            h, w = edges_bgr.shape[:2]
            if h > 0 and w > 0:
                 resized_frame = cv2.resize(
                     edges_bgr,
                     (w * scale_factor, h * scale_factor),
                     interpolation=cv2.INTER_NEAREST
                 )
                 return resized_frame
            else:
                 print(f"Warning: Invalid dimensions for edge frame ({h}x{w}). Cannot resize.")
                 return edges_bgr # Return unresized if dimensions weird

        except cv2.error as e:
            print(f"Error during OpenCV operations in _get_render_frame: {e}")
            placeholder_h = self._frame_h * 2
            placeholder_w = self._frame_w * 2
            placeholder = np.zeros((placeholder_h, placeholder_w, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Render Error", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            return placeholder
        except Exception as e:
            print(f"Unexpected error in _get_render_frame: {e}")
            placeholder_h = self._frame_h * 2
            placeholder_w = self._frame_w * 2
            placeholder = np.zeros((placeholder_h, placeholder_w, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Render Error", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            return placeholder
    # ------

    def _render_frame(self) -> None:
        """Gets the render frame (edges only) and adds text statistics."""
        frame = self._get_render_frame() # Gets the scaled edge frame

        if frame is None:
            print("Error: _get_render_frame returned None. Displaying black screen.")
            placeholder_h = self._frame_h * 2
            placeholder_w = self._frame_w * 2
            frame = np.zeros((placeholder_h, placeholder_w, 3), dtype=np.uint8)
            cv2.putText(frame, "Render Failed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        elif frame.shape[0] <= 40 or frame.shape[1] <= 100:
             print("Warning: Render frame is very small, skipping text overlays.")
        else:
            # Add text overlays (position might need adjustment)
            y_offset = 20
            font_scale = 0.6
            font_color = (50, 200, 50)
            font_thickness = 1
            try:
                cv2.putText(frame, f"Step: {self._step_count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness)
                y_offset += 25
                cv2.putText(frame, f"Episode Reward: {self._episode_reward:.3f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness)
                y_offset += 25
                action_display = "N/A"
                if self.current_key == 'left': action_display = "Left (0)"
                elif self.current_key == 'right': action_display = "Right (1)"
                elif self.current_key is None: action_display = "None (2)"
                cv2.putText(frame, f"Action: {action_display}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness)
            except Exception as e:
                 print(f"Error drawing text: {e}")

        # Display the frame
        try:
            cv2.imshow(self._render_window_name, frame)
            cv2.waitKey(1)
        except Exception as e:
            print(f"Error displaying frame with cv2.imshow: {e}")


    def render(self) -> Optional[np.ndarray]:
        """Renders the environment based on render_mode."""
        if self.render_mode == 'human':
            self._render_frame() # Shows edges + stats
            return None

        elif self.render_mode == 'rgb_array':
            # Returns the scaled edge frame (without stats overlays)
            frame = self._get_render_frame()
            if frame is None:
                 placeholder_h = self._frame_h * 2
                 placeholder_w = self._frame_w * 2
                 return np.zeros((placeholder_h, placeholder_w, 3), dtype=np.uint8)
            return frame

        return None

    def close(self) -> None:
        """Cleans up resources."""
        print("Closing SuperHexagonEnv...")
        self._running = False
        release_keys()
        if hasattr(self, '_key_thread') and self._key_thread.is_alive():
            self._key_thread.join(timeout=0.5)
        if cv2.getWindowProperty(self._render_window_name, cv2.WND_PROP_VISIBLE) >= 1:
             try:
                  cv2.destroyWindow(self._render_window_name)
             except cv2.error as e:
                  print(f"Error destroying window '{self._render_window_name}': {e}")
        cv2.waitKey(1)
        print("SuperHexagonEnv closed.")