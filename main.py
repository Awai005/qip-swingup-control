from fill_the_code import Controller
from render.render import Renderer

def main():
    #####################################################################################
    your_controller = Controller()      # Instantiate your controller in fill_the_code.py.
    SEC = 10                            # Rendering period in seconds (e.g., 10).
    #####################################################################################

    renderer = Renderer(controller=your_controller, sec=SEC)
    transitions = renderer.render()

    # Save the transitions to a txt file.
    with open('transitions.txt', 'w') as f:
        for transition in transitions:
            f.write(transition[0])
            f.write(transition[1])
            f.write(transition[2])
            f.write('\n')


if __name__ == '__main__':
    main()

