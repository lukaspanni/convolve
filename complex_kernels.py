import numpy as np
import math
from functools import reduce
from numba import njit, float32
from numba.typed import List
from numba.experimental import jitclass

# These scales bring the size of the below components to roughly the specified radius - I just hard coded these
kernel_scales = [1.4, 1.2, 1.2, 1.2, 1.2, 1.2]

# Kernel parameters a, b, A, B
# These parameters are drawn from <http://yehar.com/blog/?p=1495>
kernel_params = [
    # 1-component
    [[0.862325, 1.624835, 0.767583, 1.862321]],

    # 2-components
    [[0.886528, 5.268909, 0.411259, -0.548794],
     [1.960518, 1.558213, 0.513282, 4.56111]],

    # 3-components
    [[2.17649, 5.043495, 1.621035, -2.105439],
     [1.019306, 9.027613, -0.28086, -0.162882],
     [2.81511, 1.597273, -0.366471, 10.300301]],

    # 4-components
    [[4.338459, 1.553635, -5.767909, 46.164397],
     [3.839993, 4.693183, 9.795391, -15.227561],
     [2.791880, 8.178137, -3.048324, 0.302959],
     [1.342190, 12.328289, 0.010001, 0.244650]],

    # 5-components
    [[4.892608, 1.685979, -22.356787, 85.91246],
     [4.71187, 4.998496, 35.918936, -28.875618],
     [4.052795, 8.244168, -13.212253, -1.578428],
     [2.929212, 11.900859, 0.507991, 1.816328],
     [1.512961, 16.116382, 0.138051, -0.01]],

    # 6-components
    [[5.143778, 2.079813, -82.326596, 111.231024],
     [5.612426, 6.153387, 113.878661, 58.004879],
     [5.982921, 9.802895, 39.479083, -162.028887],
     [6.505167, 11.059237, -71.286026, 95.027069],
     [3.869579, 14.81052, 1.405746, -3.704914],
     [2.201904, 19.032909, -0.152784, -0.107988]]]


@jitclass([
    ('a', float32),
    ('b', float32),
    ('A', float32),
    ('B', float32)
])
class parameter(object):
    def __init__(self, a, b, A, B):
        self.a = a
        self.b = b
        self.A = A
        self.B = B


# Obtain specific parameters and scale for a given component count
def get_parameters(component_count=2):
    parameter_index = max(0, min(component_count - 1, len(kernel_params)))
    parameters = [parameter(*b) for b in kernel_params[parameter_index]]
    return (parameters, kernel_scales[parameter_index])


# Produces a complex kernel of a given radius and scale (adjusts radius to be more accurate)
# a and b are parameters of this complex kernel
def complex_kernel_1d(radius, scale, a, b):
    kernel_radius = radius
    kernel_size = kernel_radius * 2 + 1
    ax = np.arange(-kernel_radius, kernel_radius + 1., dtype=np.float32)
    ax = np.power(ax * scale * (1 / kernel_radius), 2)
    p1 = np.exp(-a * ax)
    p2 = b * ax
    real_part = p1 * np.cos(p2)
    imag_part = 1j * p1 * np.sin(p2)
    kernel_complex = real_part + imag_part
    return kernel_complex.reshape((1, kernel_size))


@njit()
def normalise_kernels(kernels: List, params: List):
    # Normalises with respect to A*real+B*imag
    total = 0

    for k, p in zip(kernels, params):
        # 1D kernel - applied in 2D
        for i in range(k.shape[1]):
            for j in range(k.shape[1]):
                # Complex multiply and weighted sum
                total += p.A * (k[0, i].real * k[0, j].real - k[0, i].imag * k[0, j].imag) + p.B * (
                        k[0, i].real * k[0, j].imag + k[0, i].imag * k[0, j].real)

    scalar = 1 / math.sqrt(total)
    for kernel in kernels:
        kernel *= scalar


# Combine the real and imaginary parts of an image, weighted by A and B
def weighted_sum(kernel: np.ndarray, param_a: float, param_b: float):
    return np.add(kernel.real * param_a, kernel.imag * param_b)


# Produce a 2D kernel by self-multiplying a 1d kernel. This would be slower to use
# than the separable approach, mostly for visualisation below
def multiply_kernel(kernel):
    kernel_size = kernel.shape[1]
    a = np.repeat(kernel, kernel_size, 0)
    b = np.repeat(kernel.transpose(), kernel_size, 1)
    return np.multiply(a, b)


# Visualise one or more kernel components
def show_kernel(kernels, params, component_index=None):
    import matplotlib.pyplot as plt
    if component_index is not None:
        kernel = kernels[component_index]
        kernel_2d = multiply_kernel(kernel)
        kernel_total = weighted_sum(kernel_2d, params[component_index]['A'], params[component_index]['B'])

        f, axarr = plt.subplots(1, 3)
        axarr[0].imshow(kernel_2d.real, cmap='gray', interpolation='nearest')
        axarr[1].imshow(kernel_2d.imag, cmap='gray', interpolation='nearest')
        axarr[2].imshow(kernel_total, cmap='gray', interpolation='nearest')
        plt.show()
    else:
        kernel_total = reduce(np.add,
                              (weighted_sum(multiply_kernel(k), p) for k, p in zip(kernels, params['A'], params['B'])))
        plt.imshow(kernel_total, cmap='gray', interpolation='nearest')
        plt.show()
