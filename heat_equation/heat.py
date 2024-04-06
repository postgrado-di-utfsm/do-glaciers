import numpy as np
import matplotlib.pyplot as plt

def F(T: np.ndarray, alpha: float, dx: float, dy: float) -> np.ndarray:
    """
    Calculate right-hand-side of the heat equation for a given temperature at previous time.
    Computes the laplacian using a five-point stencil.

    Parameters
    ----------
    T : numpy.ndarray
        2D array representing the temperature distribution.
    alpha : float
        Thermal diffusivity constant.
    dx : float
        Grid spacing in the x-direction.
    dy : float
        Grid spacing in the y-direction.

    Returns
    -------
    numpy.ndarray
        2D array representing the heat equation for the given temperature distribution.

    Notes
    -----
    This function assumes periodic boundary conditions.

    """
    T_ij   = np.copy(T)
    T_ip1j = np.roll(T,-1, axis=1) # T_{i+1,j}
    T_im1j = np.roll(T, 1, axis=1) # T_{i-1,j}
    T_ijp1 = np.roll(T,-1, axis=0) # T_{i,j+1}
    T_ijm1 = np.roll(T, 1, axis=0) # T_{i,j-1}
    Txx = (T_ip1j - 2 * T_ij + T_im1j) / dx ** 2
    Tyy = (T_ijp1 - 2 * T_ij + T_ijm1) / dy ** 2 
    return alpha * (Txx + Tyy)


def euler_method(T0: np.ndarray, alpha: float, dx: float, dy: float, dt: float, Nt: int) -> np.ndarray:
    """
    Solve the heat equation using the Euler method.

    Parameters
    ----------
    T0 : numpy.ndarray
        Initial temperature distribution.
    F : function
        Function that calculates the heat flux at each point.
    alpha : float
        Thermal diffusivity.
    dx : float
        Grid spacing in the x-direction.
    dy : float
        Grid spacing in the y-direction.
    dt : float
        Time step size.
    Nt : int
        Number of time steps.

    Returns
    -------
    numpy.ndarray
        Temperature distribution after Nt time steps.
    """
    T = np.copy(T0)
    for n in range(Nt):
        T += dt * F(T, alpha, dx, dy)
    return T

def predict(initial_condition_dir: str, k: float, days: int) -> tuple:
    """
    Predicts the temperature distribution over time using the heat equation.

    Parameters
    ----------
    initial_condition_dir : str
        The directory path to the initial condition file.
    k : float
        The thermal diffusivity constant.
    days : int
        The number of days to simulate.

    Returns
    -------
    x : numpy.ndarray
        The x-coordinates of the grid points.
    y : numpy.ndarray
        The y-coordinates of the grid points.
    T0 : numpy.ndarray
        The initial temperature distribution.
    T : numpy.ndarray
        The temperature distribution over time.
    """
    T0 = np.load(initial_condition_dir)
    # Not sure if it is inverted in y-axis, but
    T0 = np.flipud(T0)
    T0[np.isnan(T0)] = 288.15 # Fill NaN values with ambient temperature
    Ny, Nx = T0.shape
    Nt = 100 * days
    t_min, t_max = 0, 3600 * 24 * days 
    x = np.linspace(0, 30 * (Nx - 1), Nx)
    y = np.linspace(0, 30 * (Ny - 1), Ny)
    t = np.linspace(t_min, t_max, Nt)
    dx, dy, dt = x[1] - x[0], y[1] - y[0], t[1] - t[0]
    T = euler_method(T0, k, dx, dy, dt, Nt)
    return x, y, T0, T


def show(data):
    """
    Display the temperature data.

    Parameters:
    data : np.ndarray containing the following elements:
        x (ndarray): The x-coordinates of the grid points.
        y (ndarray): The y-coordinates of the grid points.
        T0 (ndarray): The initial temperature distribution.
        T (ndarray): The final temperature distribution.

    Returns:
    None
    """
    x, y, T0, T = data
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
    plt.title(r"$T_0$")
    p1 = axes[0].contourf(x, y, T0)
    p2 = axes[1].contourf(x, y, T)
    p3 = axes[2].contourf(x, y, T - T0)
    axes[0].set_title(r"Initial temperature: $T_0$")
    axes[1].set_title(r"Final temperature: $T$")
    axes[2].set_title(r"Temperature difference $T - T_0$")
    fig.colorbar(p1, ax=axes[0], label="Temperature (K)")
    fig.colorbar(p2, ax=axes[1], label="Temperature (K)")
    fig.colorbar(p3, ax=axes[2], label="Temperature (K)")
    plt.tight_layout()
    plt.show()
    return None