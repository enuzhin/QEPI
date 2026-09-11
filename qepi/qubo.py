import numpy as np

def binary_vector_to_value(x,num_bits,v_max = 1):
    from .grid import Nx, Nv
    scale = (2 ** num_bits - 1) / v_max
    x = x.reshape(num_bits,Nx,Nv)
    weights = (2**np.arange(num_bits)) / scale
    value = (x * weights[:,np.newaxis,np.newaxis]).sum(0)
    return value

def value_to_binary_vector(value,num_bits,v_max = 1):
    from .grid import Nx, Nv
    scale = (2 ** num_bits - 1) / v_max
    nums = (value * scale).astype(int)
    bin_nums = ((nums[np.newaxis,:,:] & (2**np.arange(num_bits)[:,np.newaxis,np.newaxis])) != 0).astype(int)
    x = bin_nums.reshape(num_bits*Nx*Nv)
    return x

def linear_operator_to_qubo(A,num_bits,v_max):
    from .grid import Nx, Nv
    scale = (2 ** num_bits - 1) / v_max
    P4 = A[np.newaxis,:,np.newaxis,:]
    weights = (2**np.arange(num_bits)) / scale
    P4 = weights[:,np.newaxis,np.newaxis,np.newaxis] * P4 * weights[np.newaxis,np.newaxis,:,np.newaxis]
    P = P4.reshape(num_bits*Nx*Nv,num_bits*Nx*Nv)
    return P

def vector_to_cubo(b,num_bits,v_max):
    from .grid import Nx, Nv
    scale = (2 ** num_bits - 1) / v_max
    p2 = b[np.newaxis,:]
    weights = (2**np.arange(num_bits)) / scale
    p2 = weights[:,np.newaxis] * p2
    p = p2.reshape(num_bits*Nx*Nv)
    return p

def linear_equation_to_qubo(A,b,num_bits,v_max):
    A_o = A.T @ A
    b_o = b @ A
    Pa = linear_operator_to_qubo(A_o,num_bits,v_max)
    pb = vector_to_cubo(b_o,num_bits,v_max)
    P = Pa
    np.fill_diagonal(P, np.diag(P) + 2 * pb)
    return P

def check_encoding(L, b, value, num_bits, v_max):
    L, b = np.asarray(L, dtype=float), np.asarray(b, dtype=float)
    P = linear_equation_to_qubo(L, b, num_bits, v_max)
    x = value_to_binary_vector(-value, num_bits, v_max)
    decoded = -binary_vector_to_value(x, num_bits, v_max)
    return x @ P @ x + (b * b).sum(), ((L @ decoded.reshape(-1) - b) ** 2).sum()