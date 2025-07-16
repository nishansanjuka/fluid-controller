import cv2
import mediapipe as mp
import numpy as np
import time
import ctypes
import json
import websocket
import pyautogui
import threading
from collections import deque

# Configure pyautogui for maximum speed
pyautogui.FAILSAFE = True  # Move mouse to top-left corner to abort
pyautogui.PAUSE = 0  # No pause between pyautogui calls for maximum speed

class SmoothCursorController:
    def __init__(self, smoothing_factor=0.15, debounce_time=0.016, max_speed=2000):
        """
        Initialize smooth cursor controller
        
        Args:
            smoothing_factor: How much to smooth movement (0.1 = very smooth, 0.5 = less smooth)
            debounce_time: Minimum time between movements in seconds (0.016 = ~60fps)
            max_speed: Maximum pixels per second the cursor can move
        """
        self.smoothing_factor = smoothing_factor
        self.debounce_time = debounce_time
        self.max_speed = max_speed
        
        self.current_pos = None
        self.target_pos = None
        self.last_move_time = 0
        self.position_history = deque(maxlen=5)  # Keep last 5 positions for smoothing
        
        # Threading for smooth movement
        self.movement_thread = None
        self.should_stop = False
        self.movement_lock = threading.Lock()
        
        # Start movement thread
        self.start_movement_thread()
    
    def start_movement_thread(self):
        """Start the background thread for smooth cursor movement"""
        self.should_stop = False
        self.movement_thread = threading.Thread(target=self._movement_loop, daemon=True)
        self.movement_thread.start()
    
    def stop(self):
        """Stop the movement thread"""
        self.should_stop = True
        if self.movement_thread:
            self.movement_thread.join()
    
    def _movement_loop(self):
        """Background thread loop for smooth cursor movement"""
        while not self.should_stop:
            current_time = time.time()
            
            with self.movement_lock:
                if (self.target_pos is not None and 
                    current_time - self.last_move_time >= self.debounce_time):
                    
                    if self.current_pos is None:
                        # First movement - get current cursor position
                        try:
                            self.current_pos = pyautogui.position()
                        except:
                            self.current_pos = (0, 0)
                    
                    # Calculate smooth movement
                    target_x, target_y = self.target_pos
                    current_x, current_y = self.current_pos
                    
                    # Calculate distance and apply speed limiting
                    dx = target_x - current_x
                    dy = target_y - current_y
                    distance = np.sqrt(dx*dx + dy*dy)
                    
                    if distance > 0:
                        # Apply smoothing
                        new_x = current_x + dx * self.smoothing_factor
                        new_y = current_y + dy * self.smoothing_factor
                        
                        # Speed limiting
                        time_delta = current_time - self.last_move_time
                        if time_delta > 0:
                            max_distance = self.max_speed * time_delta
                            if distance > max_distance:
                                # Scale down movement to respect max speed
                                scale = max_distance / distance
                                new_x = current_x + dx * scale
                                new_y = current_y + dy * scale
                        
                        # Apply movement
                        try:
                            pyautogui.moveTo(int(new_x), int(new_y))
                            self.current_pos = (new_x, new_y)
                            self.last_move_time = current_time
                        except Exception as e:
                            print(f"Mouse movement error: {e}")
            
            # Sleep for smooth 60fps movement
            time.sleep(0.016)  # ~60fps
    
    def update_target(self, x, y):
        """Update target position for smooth movement"""
        if x is None or y is None:
            return
            
        with self.movement_lock:
            self.target_pos = (float(x), float(y))
            
            # Add to position history for additional smoothing
            self.position_history.append((x, y))
            
            # Use average of recent positions for even smoother movement
            if len(self.position_history) >= 3:
                avg_x = sum(pos[0] for pos in self.position_history) / len(self.position_history)
                avg_y = sum(pos[1] for pos in self.position_history) / len(self.position_history)
                self.target_pos = (avg_x, avg_y)

class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,  # Slightly higher for better accuracy
            min_tracking_confidence=0.5   # Slightly higher for better tracking
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_draw_styles = mp.solutions.drawing_styles
        
    def detect_hands(self, frame):
        """Detect hands in the frame and return hand positions"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        hand_positions = {'left': None, 'right': None}
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Get hand label (left or right)
                hand_label = handedness.classification[0].label.lower()
                
                # Get wrist position (landmark 0)
                wrist = hand_landmarks.landmark[0]
                h, w, _ = frame.shape
                x = int(wrist.x * w)
                y = int(wrist.y * h)
                
                hand_positions[hand_label] = (x, y)
                
                # Draw hand landmarks
                color = (0, 0, 255) if hand_label == 'left' else (255, 255, 0)  # Red for left, Yellow for right
                self.mp_draw.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw_styles.get_default_hand_landmarks_style(),
                    self.mp_draw_styles.get_default_hand_connections_style()
                )
                
                # Draw wrist circle
                cv2.circle(frame, (x, y), 10, color, -1)
                
                # Add hand label
                cv2.putText(frame, f"{hand_label.upper()}", (x-30, y-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return frame, hand_positions

def create_overlay(frame, hand_positions, screen_size):
    """Create position overlay on the frame"""
    overlay = frame.copy()
    
    # Create semi-transparent background for overlay
    cv2.rectangle(overlay, (screen_size[0] - 350, 10), (screen_size[0] - 10, 180), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Add title
    cv2.putText(frame, "Hand Positions", (screen_size[0] - 340, 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add screen size
    cv2.putText(frame, f"Screen: {screen_size[0]}x{screen_size[1]}", 
               (screen_size[0] - 340, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Add left hand position
    left_text = f"Left: {hand_positions['left']}" if hand_positions['left'] else "Left: Not detected"
    left_color = (0, 0, 255) if hand_positions['left'] else (100, 100, 100)
    cv2.putText(frame, left_text, (screen_size[0] - 340, 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, left_color, 1)
    
    # Add right hand position
    right_text = f"Right: {hand_positions['right']}" if hand_positions['right'] else "Right: Not detected"
    right_color = (255, 255, 0) if hand_positions['right'] else (100, 100, 100)
    cv2.putText(frame, right_text, (screen_size[0] - 340, 115), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, right_color, 1)
    
    # Add instructions
    cv2.putText(frame, "Press 'q' to quit", (screen_size[0] - 340, 150), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(frame, "Press 'f' for fullscreen", (screen_size[0] - 340, 170), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    return frame

def get_screen_size():
    user32 = ctypes.windll.user32
    screensize = (user32.GetSystemMetrics(0), user32.GetSystemMetrics(1))
    return screensize

def convert_to_screen_coords(hand_pos, cam_size, screen_size):
    if hand_pos is None:
        return None
    cam_w, cam_h = cam_size
    screen_w, screen_h = screen_size
    x, y = hand_pos
    # Scale coordinates
    screen_x = int(x * screen_w / cam_w)
    screen_y = int(y * screen_h / cam_h)
    return (screen_x, screen_y)

def send_hand_positions(ws, hand_positions):
    try:
        ws.send(json.dumps(hand_positions))
    except Exception as e:
        print(f"WebSocket send error: {e}")

def click_with_hand(hand_positions_screen, hand='left'):
    """Perform mouse click when specific hand is detected"""
    if hand_positions_screen[hand] is not None:
        pos = hand_positions_screen[hand]
        try:
            # Click at hand position
            pyautogui.click(pos['x'], pos['y'])
        except Exception as e:
            print(f"Mouse click error: {e}")

def main():
    # Initialize camera with optimized settings for 60fps
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    # Initialize hand detector
    detector = HandDetector()
    
    # Initialize smooth cursor controller
    cursor_controller = SmoothCursorController(
        smoothing_factor=0.15,  # Adjust for more/less smoothing
        debounce_time=0.016,    # 60fps debounce
        max_speed=2000          # Max pixels per second
    )

    # Optimize camera settings for 60fps
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)  # Request 60fps
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize latency
    
    # Additional camera optimizations
    cap.set(cv2.CAP_PROP_EXPOSURE, -6)  # Lower exposure for faster capture
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus for consistent performance
    cap.set(cv2.CAP_PROP_AUTO_WB, 1)    # Enable auto white balance
    
    # Get actual camera settings
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    cam_size = (width, height)

    # Get screen size
    screen_size = get_screen_size()
    print(f"Camera resolution: {width}x{height}")
    print(f"Camera FPS: {actual_fps}")
    print(f"Screen size: {screen_size[0]}x{screen_size[1]}")
    print("Controls:")
    print("- Press 'q' to quit")
    print("- Press 'f' to toggle fullscreen")
    print("- Press 's' to save screenshot")
    print("- Press 'r' to use right hand for mouse control")
    print("- Press 'l' to use left hand for mouse control")
    print("- Press 'c' to click with left hand")
    print("- Press 'space' to disable/enable mouse movement")
    print("- Press '+' to increase smoothing")
    print("- Press '-' to decrease smoothing")

    # Create window
    cv2.namedWindow('Hand Detection', cv2.WINDOW_NORMAL)
    fullscreen = False
    
    # Mouse control settings
    active_hand = 'right'  # Default to right hand
    mouse_enabled = True

    # Connect to WebSocket server
    ws = None
    try:
        ws = websocket.create_connection("ws://localhost:5000")
        print("Connected to WebSocket server.")
    except Exception as e:
        print(f"WebSocket connection error: {e}")

    # FPS calculation
    fps_counter = 0
    start_time = time.time()
    display_fps = 0

    # Track previous hand positions for WebSocket
    prev_positions = {'left': None, 'right': None}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)

            # Get current frame size
            h, w = frame.shape[:2]

            # Detect hands
            frame, hand_positions = detector.detect_hands(frame)
            if hand_positions is None:
                hand_positions = {'left': None, 'right': None}

            # Convert hand positions to screen coordinates and format as {left: {x, y}, right: {x, y}}
            def format_hand(pos):
                if pos is None:
                    return None
                return {'x': pos[0], 'y': pos[1]}

            hand_positions_screen = {
                'left': format_hand(convert_to_screen_coords(hand_positions['left'], cam_size, screen_size)),
                'right': format_hand(convert_to_screen_coords(hand_positions['right'], cam_size, screen_size))
            }

            # Check for movement
            movement = False
            for hand in ['left', 'right']:
                curr = hand_positions_screen[hand]
                prev = prev_positions[hand]
                if curr != prev:
                    movement = True
                    break

            # Send to WebSocket server only if movement detected
            if ws and movement:
                send_hand_positions(ws, hand_positions_screen)
                prev_positions = hand_positions_screen.copy()

            # Update smooth cursor controller with hand position
            if mouse_enabled and hand_positions_screen[active_hand] is not None:
                pos = hand_positions_screen[active_hand]
                if pos is not None:  # Additional type check
                    cursor_controller.update_target(pos['x'], pos['y'])

            # Add overlay with position information
            frame = create_overlay(frame, hand_positions, (w, h))
            
            # Add status information
            smoothing = cursor_controller.smoothing_factor
            status_text = f"Active hand: {active_hand.upper()} | Mouse: {'ON' if mouse_enabled else 'OFF'} | Smoothing: {smoothing:.2f}"
            cv2.putText(frame, status_text, (10, h - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Calculate and display FPS
            fps_counter += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:
                display_fps = fps_counter / elapsed_time
                fps_counter = 0
                start_time = time.time()

            cv2.putText(frame, f"FPS: {display_fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Display frame
            cv2.imshow('Hand Detection', frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('f'):
                # Toggle fullscreen
                if fullscreen:
                    cv2.setWindowProperty('Hand Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    fullscreen = False
                else:
                    cv2.setWindowProperty('Hand Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    fullscreen = True
            elif key == ord('s'):
                # Save screenshot
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"hand_detection_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved as {filename}")
            elif key == ord('r'):
                # Use right hand for mouse control
                active_hand = 'right'
                print("Mouse control switched to RIGHT hand")
            elif key == ord('l'):
                # Use left hand for mouse control
                active_hand = 'left'
                print("Mouse control switched to LEFT hand")
            elif key == ord('c'):
                # Click with left hand
                click_with_hand(hand_positions_screen, 'left')
                print("Left hand click executed")
            elif key == ord(' '):
                # Toggle mouse movement
                mouse_enabled = not mouse_enabled
                print(f"Mouse movement {'ENABLED' if mouse_enabled else 'DISABLED'}")
            elif key == ord('+') or key == ord('='):
                # Increase smoothing (less smooth, more responsive)
                cursor_controller.smoothing_factor = min(0.5, cursor_controller.smoothing_factor + 0.05)
                print(f"Smoothing increased to {cursor_controller.smoothing_factor:.2f}")
            elif key == ord('-'):
                # Decrease smoothing (more smooth, less responsive)
                cursor_controller.smoothing_factor = max(0.05, cursor_controller.smoothing_factor - 0.05)
                print(f"Smoothing decreased to {cursor_controller.smoothing_factor:.2f}")

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Cleanup
        cursor_controller.stop()
        cap.release()
        cv2.destroyAllWindows()
        if ws:
            ws.close()
        print("Cleanup completed")

if __name__ == "__main__":
    main()