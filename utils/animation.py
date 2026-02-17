import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Initialize the figure and axis
fig, ax = plt.subplots()
ax.set_xlim(-0.1, 1.1)  # Add margins to ensure visibility of arrows
ax.set_ylim(-0.1, 1.1)
ax.grid()
# Dots for drones
drone1, = ax.plot([], [], 'ro')  # red dot for drone 1 (P)
drone2, = ax.plot([], [], 'bo')  # blue dot for drone 2 (E)
# Arrows for heading (quivers)
quiver1 = ax.quiver(0, 0, 0, 0, angles='xy', scale_units='xy', scale=1, color='r')  # red arrow for drone 1 (P)
quiver2 = ax.quiver(0, 0, 0, 0, angles='xy', scale_units='xy', scale=1, color='b')  # blue arrow for drone 2 (E)

def init():
    """Initialize the background of the plot."""
    drone1.set_data([], [])
    drone2.set_data([], [])
    quiver1.set_offsets([[0, 0]])  # Set a default valid position
    quiver2.set_offsets([[0, 0]])
    quiver1.set_UVC(0, 0)  # Set default U and V components to 0
    quiver2.set_UVC(0, 0)
    return drone1, drone2, quiver1, quiver2

def update(frame, drone_positions):
    """Update the plot for each frame."""
    print(len(drone_positions))
    x1, y1, psi1, x2, y2, psi2 = drone_positions[frame]

    # Update drone positions
    drone1.set_data([x1], [y1])
    drone2.set_data([x2], [y2])

    # Compute the direction vectors for the arrows
    arrow1_dx = np.cos(psi1) * 0.05  # Adjust scale factor (0.05) as needed
    arrow1_dy = np.sin(psi1) * 0.05
    arrow2_dx = np.cos(psi2) * 0.05
    arrow2_dy = np.sin(psi2) * 0.05

    # Update arrows
    quiver1.set_offsets(np.array([[x1, y1]]))  # Correct 2D array format
    quiver1.set_UVC(arrow1_dx, arrow1_dy)
    quiver2.set_offsets(np.array([[x2, y2]]))  # Correct 2D array format
    quiver2.set_UVC(arrow2_dx, arrow2_dy)

    return drone1, drone2, quiver1, quiver2

def create_animation(df, filename = 'drone_movement.gif', fps=30):

    '''
        df: pandas dataframe with the following columns: 'x_p', 'y_p', 'psi_p', 'x_e', 'y_e', 'psi_e'
    '''
    drone_positions = df[['x_p', 'y_p', 'psi_p', 'x_e', 'y_e', 'psi_e']].to_numpy()

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=len(drone_positions), init_func=init, blit=True, fargs=(drone_positions))

    # Save the animation as a GIF file
    ani.save(filename, writer='pillow', fps=fps)  # Adjust fps as needed
