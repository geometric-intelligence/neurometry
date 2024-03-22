import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm

L = 10  # Length of the domain

# Initialize parameters
num_samples = 1000
theta = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
N = 4  # Number of harmonics
activation = 'relu'


def gaussian_on_circle(theta, loc, sigma=0.1):
    """A Gaussian-like function defined on the circle."""
    return np.exp(-(theta-loc)**2 / (2 * sigma**2))

def relu(x):
    return np.maximum(0, x)

# Function to plot a harmonic given amplitude and phase
def plot_harmonic(ax, amplitude, phase, n, label, activation='relu'):

    harmonic_values = amplitude * np.cos(n * theta + phase)
    if activation == 'relu':
        harmonic_values = relu(harmonic_values)
    ax.plot(np.cos(theta), np.sin(theta), zs=0, zdir='z', linestyle='--',linewidth=3, color='black')
    normalized_phase = (phase + np.pi) / (2 * np.pi)  # Normalizing from -π to π to 0 to 1
    color = cm.hsv(normalized_phase)
    ax.plot(np.cos(theta), np.sin(theta), harmonic_values, label=label,linewidth=3,color=color,alpha=1-0.1*n)
    ax.axis('off')

# Prepare figure for plotting
fig, axs = plt.subplots(2, N+1, figsize=(20, 10), subplot_kw={'projection': '3d'})
plt.tight_layout()

def update(loc):
    bump_samples = gaussian_on_circle(theta, loc=loc)

    # Compute FFT
    coefficients_fft = np.fft.fft(bump_samples)
    frequencies = np.fft.fftfreq(num_samples, d=(2*np.pi/num_samples))

    # Clear previous plots
    for ax_row in axs:
        for ax in ax_row:
            ax.cla()
            ax.axis('off')

    # Plot original function
    axs[0, 2].plot(np.cos(theta), np.sin(theta), zs=0, zdir='z', linestyle='--',linewidth=3,color='black')
    axs[0, 2].plot(np.cos(theta), np.sin(theta), bump_samples, label='Original Function',linewidth=3,color='tomato')
    axs[0, 2].set_title('Target place field, position = {:.2f}'.format(loc),fontsize=20)
    axs[0, 2].scatter(np.cos(loc), np.sin(loc), zs=0, zdir='z', s=100, c='red')

    # Plot each harmonic and the reconstructed function
    reconstructed = np.zeros(num_samples)
    for n in range(1, N+1):
        index = n if frequencies[n] >= 0 else num_samples + n
        amplitude = np.abs(coefficients_fft[index])
        phase = np.angle(coefficients_fft[index])
        
        plot_harmonic(axs[1, n-1], amplitude, phase, n, f'GC module {n}, period $\lambda=${L/n:0.1f}', activation=activation)
        axs[1, n-1].set_title(f'GC module {n}, period $\lambda_{n}=${L/n:0.1f}',fontsize=18)
        if activation == 'relu':
            reconstructed += relu(amplitude * np.cos(n * theta + phase))
        else:
            reconstructed += amplitude * np.cos(n * theta + phase)

    # Reconstructed function
    axs[1, N].plot(np.cos(theta), np.sin(theta), zs=0, zdir='z', linestyle='--',linewidth=3,color='black')
    axs[1, N].plot(np.cos(theta), np.sin(theta), reconstructed, label='Reconstructed',linewidth=3,color='limegreen')
    axs[1, N].set_title('Place field readout',fontsize=20)

# Create animation
loc_values = np.linspace(0, 2*np.pi, 100)  
ani = FuncAnimation(fig, update, frames=loc_values, repeat=True)

# Save the animation
ani.save('position_from_grid_cells.gif', writer='imagemagick', fps=10)