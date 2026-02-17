import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class GifGenerator:
    def __init__(self, arrow=True):
        # Initialize the figure and axis
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-0.1, 1.1)  # Add margins to ensure visibility of arrows
        self.ax.set_ylim(-0.1, 1.1)
        self.ax.grid()

        self.arrow = arrow

        # Dots for drones
        self.drone1, = self.ax.plot([], [], 'ro')  # red dot for drone 1 (P)
        self.drone2, = self.ax.plot([], [], 'bo')  # blue dot for drone 2 (E)

        if self.arrow:
            # Arrows for heading (quivers)
            self.quiver1 = self.ax.quiver(0, 0, 0, 0, angles='xy', scale_units='xy', scale=1, color='r')  # red arrow for drone 1 (P)
            self.quiver2 = self.ax.quiver(0, 0, 0, 0, angles='xy', scale_units='xy', scale=1, color='b')  # blue arrow for drone 2 (E)

        self.drone_positions = None

    def init(self):
        """Initialize the background of the plot."""
        self.drone1.set_data([], [])
        self.drone2.set_data([], [])

        if self.arrow:
            self.quiver1.set_offsets([[0, 0]])  # Set a default valid position
            self.quiver2.set_offsets([[0, 0]])
            self.quiver1.set_UVC(0, 0)  # Set default U and V components to 0
            self.quiver2.set_UVC(0, 0)
            return self.drone1, self.drone2, self.quiver1, self.quiver2
        else:
            return self.drone1, self.drone2

    def update(self, frame):
        x1, y1, psi1, x2, y2, psi2 = self.drone_positions[frame]
        # Update drone positions
        self.drone1.set_data([x1], [y1])
        self.drone2.set_data([x2], [y2])

        if self.arrow:
            # Compute the direction vectors for the arrows
            arrow1_dx = np.cos(psi1) * 0.05  # Adjust scale factor (0.05) as needed
            arrow1_dy = np.sin(psi1) * 0.05
            arrow2_dx = np.cos(psi2) * 0.05
            arrow2_dy = np.sin(psi2) * 0.05

            # Update arrows
            self.quiver1.set_offsets(np.array([[x1, y1]]))  # Correct 2D array format
            self.quiver1.set_UVC(arrow1_dx, arrow1_dy)
            self.quiver2.set_offsets(np.array([[x2, y2]]))  # Correct 2D array format
            self.quiver2.set_UVC(arrow2_dx, arrow2_dy)
            return self.drone1, self.drone2, self.quiver1, self.quiver2
        
        return self.drone1, self.drone2
    
    def plot_gif(self, df_states, filepath = 'drone_movement.gif', fps = 30):
        self.drone_positions = df_states[['x_p', 'y_p', 'psi_p', 'x_e', 'y_e', 'psi_e']].to_numpy()

        # Create the animation
        ani = animation.FuncAnimation(self.fig, self.update, frames=len(self.drone_positions), init_func=self.init, blit=True)
        
        # Save the animation as a GIF file
        ani.save(filepath, writer='pillow', fps=fps)  # Adjust fps as needed

        plt.close()
        print('Gif generated with success')
        