# %%
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Time to run for
max_time = 4

# Dimension of each vector (Roughly the number of tokens)
d = 40

# Number of vectors
N = 30

beta = 1

def diff_eqn_system(t, x_data):
    x_data_dots = np.empty(0)
    for i in range(N):
        x_i = x_data[i*d:(i+1)*d]
        x_i_dot_before_projection = np.zeros_like(x_i)
        Z_i = 0.0
        for j in range(N):
            x_j = x_data[j * d:(j + 1) * d]

            Z_i = Z_i + np.exp(beta * np.dot(x_i, x_j))

            x_i_dot_before_projection = x_i_dot_before_projection + np.exp(beta * np.dot(x_i, x_j)) * x_j

        x_i_dot_before_projection = x_i_dot_before_projection / Z_i

        x_i_dot = x_i_dot_before_projection - x_i * np.dot(x_i_dot_before_projection, x_i) / np.dot(x_i,x_i)

        x_data_dots = np.concatenate([x_data_dots, x_i_dot])

    return x_data_dots

# Generate N vectors of length 1
x_data = np.empty(0)
for i in range(N):
    x_i = np.random.rand(d)
    x_i = x_i / np.linalg.norm(x_i)
    x_data = np.append(x_data, x_i)

t_span = (0, max_time)
t_eval = np.linspace(*t_span, 10000)
print("Solving...")
solution = solve_ivp(diff_eqn_system, t_span, y0=x_data, args=(),
                     t_eval=t_eval,
                     method='LSODA', atol=1e-9, rtol=1e-7)

print("Solved")
print(solution)


def dot_product_histogram(time, state):
    dots = []

    for i in range(N):
        x_i = state[i*d:(i+1)*d]
        for j in range(i+1, N):
            x_j = state[j * d:(j + 1) * d]

            dots.append(np.dot(x_i, x_j))

    plt.hist(dots, bins = 30)
    plt.xlabel("Dot product <i, j>")
    plt.ylabel("Frequency")
    plt.xlim(-1.1, 1.1)
    plt.title(f"Time = {time}")
    plt.show()

time_0 = solution.t[0]
state_0 = solution.y[:, 0]
dot_product_histogram(time_0, state_0)

time_m1 = solution.t[-1]
state_m1 = solution.y[:, -1]
dot_product_histogram(time_m1, state_m1)


# %%
