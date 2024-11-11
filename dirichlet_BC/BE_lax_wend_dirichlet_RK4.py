import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import matplotlib.animation as animation
from PIL import Image

import seaborn as sns

sns.set_theme(
    context="paper",
    style='ticks',
    palette='bright',
    font='serif',
    font_scale=1,
    color_codes=True,
    rc=None
)
sns.despine()


def IC(x, length, flag=0):
    if flag == 0:
        if x < length / 2:
            return 0.5
        else:
            return 1
    elif flag == 1:
        return np.sin(4 * np.pi * x)


def compute_f(u):
    return 1 / 2 * u * u


def g(u, delta_x, delta_t, viscosity, umatplus, umatminus, fmatplus, fmatminus, space_mat, u_left, u_right, added_vector):
    u_plus_half = dirichlet_flux_half_step(umatplus, fmatplus, delta_t, delta_x, u, 'plus', u_left, u_right)
    u_minus_half = dirichlet_flux_half_step(umatminus, fmatminus, dt, dx, u_n, 'minus', u_left, u_right)
    f_half_plus = compute_f(u_plus_half)
    f_half_minus = compute_f(u_minus_half)

    fluxes = 1 / delta_x * (f_half_minus - f_half_plus)
    ans = fluxes + viscosity / delta_x ** 2 * space_mat @ u + added_vector
    return ans


def u_exact(x, x0, t, u_left, u_right, f_left, f_right):
    s = (f_left - f_right) / (u_left - u_right)
    var = (x0 - x + s * t) * (u_left - u_right) / (4 * nu)
    u = (u_right + u_left) / 2 - (u_left - u_right) / 2 * np.tanh(var)
    return u


def dirichlet_flux_half_step(mat1, mat2, delta_t, delta_x, u, flag, uLeft, uRight):
    u_vec_bc = np.zeros(len(u))
    f_vec = compute_f(u)
    f_vec_bc = np.zeros_like(f_vec)
    flux_coeff = 1 / 2 * delta_t / delta_x
    if flag == 'plus':
        u_vec_bc[-1] = uRight
        f_vec_bc[-1] = -compute_f(uRight)

    elif flag == 'minus':
        u_vec_bc[0] = uLeft
        f_vec_bc[0] = compute_f(uLeft)

    return 1 / 2 * (mat1 @ u + u_vec_bc) + flux_coeff * (mat2 @ f_vec + f_vec_bc)


print('-----------------------------------------------------------------------------------------------------------')
print('Begin code...')
print('-----------------------------------------------------------------------------------------------------------')
plt.rcParams.update({
                'font.size': 16,  # Base font size
                'axes.labelsize': 14,  # X and Y labels
                'axes.titlesize': 14,  # Title size
                'legend.fontsize': 12,  # Legend font size
                'xtick.labelsize': 14,  # X-axis tick labels
                'ytick.labelsize': 14,  # Y-axis tick labels
            })

nu_list = np.array([0.01])
# nu_list = [0.01, 0.1]
print(f'nu vales = {nu_list}\n')
for nu in nu_list:
    # ---------------------------------------------------------------------------------------------------------------
    print('Beginning calculation...')
    # ---------------------------------------------------------------------------------------------------------------
    # define variables
    dt = 0.001
    dx = 5 * dt
    L = 20
    T = 0.5
    # ---------------------------------------------------------------------------------------------------------------
    # derived variables
    mu = dt / dx
    m = int(L / dx) + 1
    n = int(T / dt) + 1

    m2 = m - 2

    print(f'Length={L}, dx={dx}, dt={dt}, mu={mu}\n')

    # ---------------------------------------------------------------------------------------------------------------
    # create initial u vector
    xs = np.arange(0, L + dx / 2, dx)
    ans_array = np.zeros((n, m))
    IC_flag = 0
    ans_array[0, :] = np.array([IC(x, L, IC_flag) for x in xs])

    # ---------------------------------------------------------------------------------------------------------------
    # create image directory
    path = './images_nu=' + str(nu)
    if not os.path.exists(path):
        os.mkdir(path)
        print('images directory created.\n')
    else:
        print('images directory already exists')
        os.chmod(path, 0o666)
        files = glob.glob(f'{path}/*.png')
        print('Removing files...')
        for f in files:
            os.remove(f)

        print('Files removed.\n')

    # ---------------------------------------------------------------------------------------------------------------
    # Create boundary conditions
    print('Implementing boundary conditions...')

    # Dirichlet BCs:
    uL = IC(xs[0], L, IC_flag)
    uR = IC(xs[-1], L, IC_flag)
    fL = uL ** 2 / 2
    fR = uR ** 2 / 2
    ans_array[:, 0] = uL
    ans_array[:, -1] = uR

    # ---------------------------------------------------------------------------------------------------------------
    # Create matrices

    # Create matrices used in flux calculations in Lax-Wendroff
    flux_matrix_plus = np.diag(np.ones(m2)) - np.diag(np.ones(m2 - 1), k=1)
    flux_matrix_minus = -np.diag(np.ones(m2)) + np.diag(np.ones(m2 - 1), k=-1)

    u_half_matrix_plus = np.diag(np.ones(m2)) + np.diag(np.ones(m2 - 1), k=+1)
    u_half_matrix_minus = np.diag(np.ones(m2)) + np.diag(np.ones(m2 - 1), k=-1)

    # Create matrix used for spatial discretization
    spatial_matrix = np.diag(-2 * np.ones(m2)) + np.diag(np.ones(m2 - 1), k=1) + np.diag(np.ones(m2 - 1), k=-1)

    # This next vector is specifically for Dirichlet boundary conditions.
    # See notes for more information.
    u_boundary_vec = np.zeros(m2)
    u_boundary_vec[0] = uL
    u_boundary_vec[-1] = uR

    boundary_add_vector = nu / dx ** 2 * u_boundary_vec

    print('Initial conditions complete.\n')

    # Create true answer:
    true_ans_array = np.zeros_like(ans_array)
    true_ans_array[0, :] = ans_array[0, :]
    # ---------------------------------------------------------------------------------------------------------------
    # Begin time stepping
    print(f'Begin time stepping... dt={dt}')
    ims = []
    for i in range(0, n):
        if i > 0:
            # Generate answer
            u_n = ans_array[i - 1, 1:-1]

            # generate the k values
            # Fluxes must be computed in the g function, so a lot of information will be passed
            k1 = g(u_n, dx, dt, nu, u_half_matrix_plus, u_half_matrix_minus, flux_matrix_plus, flux_matrix_minus, spatial_matrix, uL, uR, boundary_add_vector)
            k2 = g(u_n + k1 * dt / 2, dx, dt, nu, u_half_matrix_plus, u_half_matrix_minus, flux_matrix_plus, flux_matrix_minus, spatial_matrix, uL, uR, boundary_add_vector)
            k3 = g(u_n + k2 * dt / 2, dx, dt, nu, u_half_matrix_plus, u_half_matrix_minus, flux_matrix_plus, flux_matrix_minus, spatial_matrix, uL, uR, boundary_add_vector)
            k4 = g(u_n + k2 * dt, dx, dt, nu, u_half_matrix_plus, u_half_matrix_minus, flux_matrix_plus, flux_matrix_minus, spatial_matrix, uL, uR, boundary_add_vector)

            slope = (k1 + 2 * k2 + 2 * k3 + k4) / 6

            u_next = u_n + slope * dt
            ans_array[i, 1:-1] = u_next

            t_current = i * dt
            center = (xs[0] + xs[-1]) / 2
            true_ans_array[i, :] = np.array([u_exact(x, center, t_current, uL, uR, fL, fR) for x in xs])

        if i % 25 == 0:
            # Create and save plot
            plt.figure()
            sns.lineplot(x=xs, y=ans_array[i, :], label='Numerical Solution',linewidth=2.5)
            sns.lineplot(x=xs, y=true_ans_array[i, :], label='Analytical Solution',linestyle='dashed',linewidth=2.5)
            plt.xlabel('x')
            plt.ylabel('u(x,t)')
            plt.legend(loc='upper left')
            plt.ylim(0, 1.5)
            plt.title(
                f"Burgers' Equation with Dirichlet Boundary Conditions\n" + fr'$\nu={nu}$; $T=[0,{T}]$; $\Delta x={dx}$; $\Delta t = {dt}$')
            img_name = f'{i}_plot.png'
            plt.savefig(f'{path}/{img_name}', dpi=300)
            plt.close()
            ims.append(img_name)

        print(f'{i} of {n - 1} completed.')

    # ---------------------------------------------------------------------------------------------------------------
    # Create animation
    fig = plt.figure()
    frames = []

    filename = os.path.basename(__file__).split('.')[0]
    if not os.path.exists(f'{filename}_mu={round(mu, 2)}_nu={nu}.apng'):
        for im in ims:
            img = Image.open(f'{path}/{im}')
            frames.append([plt.imshow(img, animated=True)])
        ani = animation.ArtistAnimation(fig, frames, interval=150, blit=True)
        ani.save(f'{filename}_mu={round(mu, 2)}_nu={nu}.apng', writer='pillow')

        print(f'Animation saved as {filename}.apng\n')

    print(f'\nnu={nu} calculation complete.\n')

print(f'Code finished.')
