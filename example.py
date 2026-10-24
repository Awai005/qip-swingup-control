# Description: This is an example of how to use the simulator of the QIP System
from fill_the_code import Controller
import socket

def send_command(command):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(('localhost', 65432))
        s.sendall(command.encode())
        data = s.recv(1024).decode()
    return data.strip()

################################################################################################

your_controller = Controller()                               # Instantiate your controller in fill_the_code.py.
CTRL_MS = 10                                                 # Control period in milliseconds.
MAX_STEPS = int(10 * 1000 / CTRL_MS)                         # Maximum number of steps (e.g., 10 sec total).

################################################################################################

steps = 0  
state_str = send_command('reset')                                                          # Initialize the QIP system and get the initial state.
state = list(map(float, state_str.split()))                                                # Make the state a list.

while True:
    steps += 1
    force = Controller().compute_force(state)                                              # Compute the proper force from the controller.

    next_state_str = send_command(f'step {force}')                                         # Get the next state (changed from the applied force).
    next_state = list(map(float, next_state_str.split()))                                  # Make the next state a list.
    print(f"Steps:{steps}, State: {state}, Force: {force}, Next State: {next_state}")      # Print the state, action, and next state.
   
    state = next_state                                                                     # Update the state.
    if steps >= MAX_STEPS:                                                                 # break the loop.
        break
