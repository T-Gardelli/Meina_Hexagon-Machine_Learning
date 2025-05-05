# --- Screen Capture Settings ---
import pygetwindow as gw

# Name of the game window (adjust if different)
GAME_WINDOW_TITLE = "Super Hexagon"

def get_game_window_region():
    try:
        # Fetch all windows with the specified title
        target_windows = gw.getWindowsWithTitle(GAME_WINDOW_TITLE)
        
        if not target_windows:
            raise Exception(f"Window '{GAME_WINDOW_TITLE}' not found. Is the game running?")

        game_window = target_windows[0]  # Take the first matching window
        
        # Ensure the window is not minimized
        if game_window.isMinimized:
            game_window.restore()

        return {
            "top": game_window.top,
            "left": game_window.left,
            "width": game_window.width,
            "height": game_window.height
        }
    
    except Exception as e:
        raise RuntimeError(f"Failed to detect game window: {str(e)}")

# Auto-detect game window region
GAME_WINDOW_REGION = get_game_window_region()

# --- Image Processing Settings ---
# Size to resize the game image to (smaller is faster for the AI)
# --- Image Processing ---
RESIZE_DIM = (84, 84)           # Target size (Width, Height) for processed frames
BLUR_KERNEL_SIZE = (3,3)       # Gaussian blur kernel size (Use odd numbers, e.g., (3, 3), (5, 5))
CANNY_THRESHOLD1 = 0           # Lower threshold for Canny edge detection (tune this)
CANNY_THRESHOLD2 = 45          # Upper threshold for Canny edge detection (tune this)

# --- Training Settings ---
LOG_DIR = "logs/" # Folder to save training progress (for TensorBoard)
MODEL_SAVE_DIR = "models/" # Folder to save the trained AI model
MODEL_FILENAME_BASE = "meinahexagon_dqn_beta" # Base name for saved models

# --- Environment Settings ---
FRAME_STACK = 6 # How many consecutive frames to stack together (gives sense of motion)

# --- Action Settings ---
# Delay after pressing a key (in seconds) - might need tuning
#ACTION_DELAY = 0.000005