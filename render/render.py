import time
import copy
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrow
import math
import socket

def print_formatted(float_list, decimal_places=9):
    formatted_numbers = ', '.join(f"{x:.{decimal_places}f}" for x in float_list)
    return formatted_numbers

def send_command(command):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(('localhost', 65432))
        s.sendall(command.encode())
        data = s.recv(1024).decode()
    return data.strip()

class Renderer:
    def __init__(self, controller, sec):       
        
        ctrl_ms_str = send_command(f'get_ctrl_ms')
        ctrl_ms_str, *_ = ctrl_ms_str.split()
        self.ctrl_ms = int(ctrl_ms_str)

        self.max_steps = int(sec * 1000 / self.ctrl_ms)
        self.controller = controller
        
        self.history_size = 400
        self.history = {
            'x': deque(maxlen=self.history_size),
            'link 1': deque(maxlen=self.history_size),
            'link 2': deque(maxlen=self.history_size),
            'link 3': deque(maxlen=self.history_size),
            'link 4': deque(maxlen=self.history_size)
        }
        self.cur_steps = [0]  # Use a list to maintain state across updates
        
        
        state_str = send_command('reset')
        self.state = list(map(float, state_str.split()))  # Initialize state
        
        length_str = send_command('get_lengths')
        self.L = list(map(float, length_str.split()))  

        self.states = [self.state]
        self.next_states = []
        self.actions = []

        self.finish_flag = False

    def reset(self):
        self.history = {
            'x': deque(maxlen=self.history_size),
            'link 1': deque(maxlen=self.history_size),
            'link 2': deque(maxlen=self.history_size),
            'link 3': deque(maxlen=self.history_size),
            'link 4': deque(maxlen=self.history_size)
        }

    def render(self):
        fig, (ax, ax_cart, ax_state) = plt.subplots(3, 1, gridspec_kw={'height_ratios': [4, 1.2, 1.2]})
        plt.subplots_adjust(hspace=0.4, bottom=0.2)  # Adjust subplot to make room for slider and increase vertical space between plots

        # Main animation plot
        ax.set_xlim(-7.3816, 7.3816)
        ax.set_ylim(-1.8144, 1.8144)

        # Add horizontal line for cart track
        ax.axhline(0, color='gray', linestyle='--', zorder=1)

        cart_width = 0.3
        cart_height = 0.2

        # Cart    
        cart = plt.Rectangle((self.state[0] - cart_width / 2, -cart_height / 2), cart_width, cart_height,
                             facecolor='blue', edgecolor='black', zorder=2)
        ax.add_patch(cart)

        # Pendulums
        pendulum_lines = []
        for i in range(4):
            line = plt.Line2D([], [], markerfacecolor='black', markersize=4, color=(0.4, 0.4, 0.4), marker='o', lw=3.5,
                              zorder=3)
            pendulum_lines.append(line)
            ax.add_line(line)

        # Initial dummy arrow
        action_arrow = FancyArrow(0, 0, 0, 0, width=0, head_width=0, head_length=0)
        ax.add_patch(action_arrow)

        # Cart position graph
        ax_cart.set_xlim(0, self.history_size)
        ax_cart.set_ylim(-4, 4)
        ax_cart.set_xlabel(f'Last {self.history_size} steps')
        ax_cart.set_ylabel('Cart Position')
        line_cart, = ax_cart.plot([], [], label='Cart Position')
        ax_cart.axhline(0, color='grey', linestyle=':')  # y=0 점선 추가
        ax_cart.legend(loc='upper left')

        # State info graph
        ax_state.set_xlim(0, self.history_size)
        ax_state.set_ylim(-np.pi,  np.pi)
        ax_state.set_xlabel(f'Last {self.history_size} steps')
        ax_state.set_ylabel('Angles (rad)')
        line_angle1, = ax_state.plot([], [], label='link 1')
        line_angle2, = ax_state.plot([], [], label='link 2')
        line_angle3, = ax_state.plot([], [], label='link 3')
        line_angle4, = ax_state.plot([], [], label='link 4')
        ax_state.axhline(0, color='grey', linestyle=':')  # y=-pi, pi 점선 추가
        ax_state.legend(loc='upper left')

        # Action and step text
        action_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10, verticalalignment='top', zorder=4)
        step_text = ax.text(0.02, 0.88, '', transform=ax.transAxes, fontsize=10, verticalalignment='top', zorder=4)

        def init():         
            cart.set_xy((self.state[0] - cart_width / 2, -cart_height / 2))
            for line in pendulum_lines:
                line.set_data([], [])
            line_cart.set_data([], [])
            line_angle1.set_data([], [])
            line_angle2.set_data([], [])
            line_angle3.set_data([], [])
            line_angle4.set_data([], [])
            action_text.set_text('')
            step_text.set_text('')
            return cart, *pendulum_lines, action_arrow, line_cart, line_angle1, line_angle2, line_angle3, line_angle4, action_text, step_text

        def update(frame):
            nonlocal action_arrow
            action = self.controller.compute_force(self.state)

            self.cur_steps[0] += 1  # Increment the step count
            if self.cur_steps[0] > self.max_steps:
                self.finish_flag = True
                return

            state_str = send_command(f'step {action}')
            self.state = list(map(float, state_str.split()))


            self.actions.append(action)
            self.next_states.append(copy.deepcopy(self.state))
            self.states.append(copy.deepcopy(self.state))

            # Update cart position
            cart.set_xy((self.state[0] - cart_width / 2, -cart_height / 2))

            # Calculate pendulum positions
            x = self.state[0]
            angles = [self.state[2], self.state[4], self.state[6], self.state[8]]
            lengths = [self.L[0], self.L[1], self.L[2], self.L[3]]
            for i in range(4):
                positions = [x]
                heights = [0]
                for j in range(i + 1):
                    positions.append(positions[-1] + lengths[j] * np.sin(angles[j]))
                    heights.append(heights[-1] + lengths[j] * np.cos(angles[j]))
                pendulum_lines[i].set_data(positions, heights)

            # Remove the old arrow and add a new one
            action_arrow.remove()
            scale = 0.0045   # Scale the size of the arrow for better visualization
            if action > 0:
                action_arrow = FancyArrow(x - 3 * cart_width / 2, 0, scale * action, 0, width=0.02, head_width=0.08,
                                          head_length=0.1, fc='lightcoral', ec='lightcoral', zorder=3)
            else:
                action_arrow = FancyArrow(x + 3 * cart_width / 2, 0, scale * action, 0, width=0.02, head_width=0.08,
                                          head_length=0.1, fc='lightcoral', ec='lightcoral', zorder=3)
            ax.add_patch(action_arrow)

            # Update state history
            self.history['x'].append(x)

            def wrap_angle(angle):
                return math.remainder(angle, 2 * math.pi)

            # 각도 리스트 생성 및 제한
            __angles = [wrap_angle(self.state[2]), wrap_angle(self.state[4]), wrap_angle(self.state[6]),
                        wrap_angle(self.state[8])]

            # 기록 갱신
            self.history['link 1'].append(__angles[0])
            self.history['link 2'].append(__angles[1])
            self.history['link 3'].append(__angles[2])
            self.history['link 4'].append(__angles[3])

            # Update cart position graph
            line_cart.set_data(range(len(self.history['x'])), self.history['x'])

            # Update state info graph
            line_angle1.set_data(range(len(self.history['link 1'])), self.history['link 1'])
            line_angle2.set_data(range(len(self.history['link 2'])), self.history['link 2'])
            line_angle3.set_data(range(len(self.history['link 3'])), self.history['link 3'])
            line_angle4.set_data(range(len(self.history['link 4'])), self.history['link 4'])

            # Update action and step text
            action_text.set_text(f'Action: {action:.2f}')
            step_text.set_text(f'Steps: {self.cur_steps[0]}')

            return cart, *pendulum_lines, action_arrow, line_cart, line_angle1, line_angle2, line_angle3, line_angle4, action_text

        ani = FuncAnimation(fig, update, frames=400, init_func=init, blit=False, interval=1)

        while not self.finish_flag:
            plt.pause(0.01)  # Allow the plot to update and check finish_flag regularly
            if self.finish_flag:
                plt.close(fig)

                assert len(self.states) == len(self.actions) + 1 and len(self.states) == len(self.next_states) + 1, \
                    ("The length of states, actions, and next_states must be the same. But len(states) = {}, len(actions)"
                     " = {}, len(next_states) = {}").format(len(self.states), len(self.actions), len(self.next_states))
                transitions = []
                for i in range(len(self.states) - 1):
                    state = self.states[i]
                    action = self.actions[i]
                    next_state = self.next_states[i]

                    state = "(" + print_formatted(state) + ")"
                    next_state = "(" + print_formatted(next_state) + ")"
                    action = "(" + f"{action:.{9}f}" + ")"

                    transitions.append((state, action, next_state))
                return transitions

                