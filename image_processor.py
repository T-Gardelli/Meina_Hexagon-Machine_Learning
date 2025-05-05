import cv2
import numpy as np
import config

def process_frame(frame):
    """
    Process a raw frame and return a 1-channel image:
    - Channel 1: Grayscale resized frame. ( if needed)
    - Channel 0: Canny edge detection result using thresholds from config. Default
    """
    # Resize the frame (config.RESIZE_DIM is (width, height))
    resized = cv2.resize(frame, config.RESIZE_DIM, interpolation=cv2.INTER_AREA) # Use INTER_AREA for shrinking

    # Convert to grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur using kernel size from config
    # Ensure kernel dimensions are positive and odd
    kernel_h = max(1, int(config.BLUR_KERNEL_SIZE[1]))
    kernel_w = max(1, int(config.BLUR_KERNEL_SIZE[0]))
    if kernel_h % 2 == 0: kernel_h += 1 # Make odd
    if kernel_w % 2 == 0: kernel_w += 1 # Make odd
    blur = cv2.GaussianBlur(gray, (kernel_w, kernel_h), 0)

    # Perform Canny edge detection using thresholds from config
    # Ensure thresholds are valid integers
    thresh1 = max(0, int(config.CANNY_THRESHOLD1))
    thresh2 = max(0, int(config.CANNY_THRESHOLD2))
    if thresh1 >= thresh2: # Ensure threshold1 < threshold2
        thresh1 = max(0, thresh2 - 1)

    edges = cv2.Canny(blur, threshold1=thresh1, threshold2=thresh2)

    # Return only the edges, but add a channel dimension (H, W, 1)
    processed = np.expand_dims(edges, axis=-1)

    # Ensure the data is in uint8 to match the observation space
    return processed.astype(np.uint8)

# Example usage (for testing standalone)
# --- Testing section ---
if __name__ == '__main__':
    # Create a dummy config *or* modify the imported config object directly
    class DummyConfig:
        RESIZE_DIM = (168, 168) # Larger for visualization
        BLUR_KERNEL_SIZE = (5, 5)
        CANNY_THRESHOLD1 = 50 # Will be overridden in the loop
        CANNY_THRESHOLD2 = 150 # Will be overridden in the loop
    config_test = DummyConfig() # Use a separate object for testing

    # List of threshold pairs to test [(thresh1, thresh2), ...]
    threshold_pairs_to_test = [
        (30, 90),
        (50, 150),
        (70, 210),
        (100, 200),
        (150, 250),
        (20, 50)
    ]

    # Create a dummy black frame with a white circle and rectangle
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(dummy_frame, (100, 100), (300, 300), (255, 255, 255), 2)
    cv2.circle(dummy_frame, (450, 240), 80, (255, 255, 255), -1) # Filled circle

    print("Starting Canny threshold test sequence.")
    print("Press 'n' for next threshold pair, 'q' or ESC to quit.")

    cv2.imshow("Original Dummy Frame", cv2.resize(dummy_frame, (320, 240)))

    current_index = 0
    original_config = config # Store original config if modifying directly
    config = config_test # Point 'config' to our test config

    while 0 <= current_index < len(threshold_pairs_to_test):
        t1, t2 = threshold_pairs_to_test[current_index]

        print(f"\nTesting Threshold Pair {current_index + 1}/{len(threshold_pairs_to_test)}: ({t1}, {t2})")

        # --- Crucial Step: Update the config object before calling process_frame ---
        config.CANNY_THRESHOLD1 = t1
        config.CANNY_THRESHOLD2 = t2
        # ---

        # Process the frame - it will now use the updated thresholds from 'config'
        processed_output = process_frame(dummy_frame)

        # Display the results for this pair
        #cv2.imshow("Processed Grayscale", processed_output[:, :, 0]) # Grayscale is constant if resize/blur don't change

        window_title_edges = f"Edges (T1={t1}, T2={t2})"
        cv2.imshow(window_title_edges, processed_output[:, :, 0])

        # Wait for user input
        key = cv2.waitKey(0) & 0xFF # Use mask for 64-bit compatibility

        # Clean up the specific edge window before showing the next one
        cv2.destroyWindow(window_title_edges)

        if key == ord('q') or key == 27: # 'q' or ESC key
            print("Quit signal received.")
            break
        elif key == ord('n'):
             print("Moving to next...")
             current_index += 1
        else:
            # Default behavior: treat any other key like 'n' (next)
            print("Moving to next (default)...")
            current_index += 1

    config = original_config # Restore original config if needed
    print("Testing finished.")
    cv2.destroyAllWindows() # Close all remaining OpenCV windows