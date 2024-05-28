from pynput import keyboard
from cartpole.Communicator import Communicator

communicator = Communicator("/dev/ttyUSB0", 115200)
communicator.send_message('h', 0)
communicator.send_message('s', 12800)
communicator.send_message('v', 0)
communicator.send_message('a', 1000000)

left = False
right = False

def on_press(key):
    global left, right
    try:
        if key == keyboard.Key.left and not left:
            communicator.send_message('v', 60000)
            left = True
            right = False
        elif key == keyboard.Key.right and not right:
            communicator.send_message('v', -60000)
            right = True
            left = False
    except AttributeError:
        pass

def on_release(key):
    # Optional: handle key release events if necessary
    pass

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# Keep the program running
listener.join()
