import mss
import numpy as np
import cv2
from PIL import Image
import config

def grab_screen():
    """Captures the game window region specified in config.py."""
    with mss.mss() as sct:
        monitor = config.GAME_WINDOW_REGION
        sct_img = sct.grab(monitor)

        # Convert the raw image data to an OpenCV image (BGR format)
        img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        img_np = np.array(img)
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        return img_cv

# --- Optional: Test Function ---
if __name__ == "__main__":
    import time
    print("Testing screen capture. Press 'q' in the window to quit.")
    while True:
        frame = grab_screen()
        cv2.imshow("Screen Capture Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.01) # Small delay
    cv2.destroyAllWindows()
    print("Capture test finished.")