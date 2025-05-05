import pydirectinput
import time
import config

# Make sure NumLock is OFF for pydirectinput arrow keys to work reliably!
pydirectinput.PAUSE = 0.005 # Small delay between key down/up if needed

def press_left():
    """Presses and releases the Left arrow key."""
    pydirectinput.keyDown('left')
    time.sleep(0.03) # Optional tiny hold
    pydirectinput.keyUp('left')
    # print("Action: Left") # Optional debug print

def press_right():
    """Presses and releases the Right arrow key."""
    pydirectinput.keyDown('right')
    time.sleep(0.03) # Optional tiny hold
    pydirectinput.keyUp('right')
    # print("Action: Right") # Optional debug print

def press_space():
    """Presses and releases the Space arrow key."""
    pydirectinput.keyDown('space')
    time.sleep(0.5) # Optional tiny hold
    pydirectinput.keyUp('space')
    # print("Action: Space") # Optional debug print

def release_keys():
    """Ensures no keys are stuck down."""
    pydirectinput.keyUp('left')
    pydirectinput.keyUp('right')
    pydirectinput.keyUp('space')

# --- Optional: Test Function ---
if __name__ == "__main__":
    print("Testing key presses.")
    print("Ensure Super Hexagon window is active/focused!")
    print("Make sure NumLock is OFF.")
    print("Starting in 5 seconds...")
    time.sleep(5)

    print("Pressing Left...")
    press_left()
    time.sleep(1)

    print("Pressing Right...")
    press_right()
    time.sleep(1)

    release_keys()
    print("Key press test finished.")