from dm import *
from qnn import *
from typing import Tuple
from typing import Optional
import json
import pandas

def CHSH_max(dm: DensityMatrix) -> float:
    T = dm.to_correlation_matrix_q2()
    U = T.T @ T
    eigens = np.linalg.eigvals(U).real
    eigens.sort()
    return 2.0 * np.sqrt(eigens[-1] + eigens[-2])


def rand_state(n_qubits: int) -> np.ndarray:
    d = 2**n_qubits
    # randomize [-1, 1] real and imag part
    re = np.random.uniform(-1, 1, (d,)) # Real part
    im = np.random.uniform(-1, 1, (d,)) # Imag part
    state = re + 1.0j * im
    return state / np.linalg.norm(state)    # normalize

def rand_density_matrix(n_qubits: int, m: int) -> DensityMatrix:
    """Randomize density matrix.

    Args:
        n_qubits (int): Number of qubits.
        m (int): Number of components.

    Returns:
        DensityMatrix: Random density matrix.
    """
    d = 2**n_qubits

    # probabilities sum to 1
    probabilities = np.random.random((m, ))
    probabilities /= probabilities.sum()

    rho = np.zeros((d, d), dtype=np.complex64)
    rho = np.asmatrix(rho)
    for p in probabilities:
        state = np.asmatrix(rand_state(n_qubits))
        rho += p * state.H @ state
    
    return DensityMatrix.build(rho)

def run_random2():
    m = 3
    for epoch in range(100):
        print(f"epoch: {epoch}")
        rho_rand = rand_density_matrix(2, 3)
        v_opt = CHSH_max(rho_rand)
        v_est = CHSH_qnn(rho_rand)
        print(f"CHSH max: {v_opt:.7f}")
        print(f"Estimate: {v_est:.7f}")

        if np.abs(v_opt - v_est) > 1e-5:
            print(f"Error!")
            print(rho_rand.rho)
            break


def count_nonlocality(data_generator, solver, n: int, threshold):
    """Count proportions of data that is greater than threshold.

    Args:
        data_generator (fun): Data generator.
        solver (fun): Calculate the data from `data_generator`
        n (int): Total number.
        threshold (float):

    Returns:
        float: Proportions of data that is greater than threshold.
    """
    cnt = 0
    for i in range(n):
        dm = data_generator()
        sol = solver(dm)
        if sol > threshold + EPS:
            cnt += 1
    return cnt / n


# def generate_positive(n: int, m: int) -> np.matrix:
#     P = np.zeros(shape=(n, n), dtype=np.complex64)
#     P = np.asmatrix(P)
#     for i in range(m):
#         v_re = np.random.random(size=(n, 1))
#         v_im = np.random.random(size=(n, 1))
#         v = np.asmatrix(v_re + 1j * v_im)
#         c = np.random.random()
#         P += c * v @ v.H
#     return P

# def generate_real_positive(n: int, m: int) -> np.matrix:
#     P = np.zeros(shape=(n, n), dtype=np.float64)
#     P = np.asmatrix(P)
#     for i in range(m):
#         v_re = np.random.random(size=(n, 1))
#         v = np.asmatrix(v_re)
#         c = np.random.random()
#         P += c * v @ v.T
#     return P

def check_positive(P: np.matrix) -> bool:
    # P is Hermitian
    if not np.allclose(P, P.H):
        return False
    
    es = np.linalg.eigvals(P)
    for e in es:
        # e should be real and e should be positive
        if np.abs(e.imag) > 1e-5 or e < -1e-5 :
            return False
    return True


def GHZ(sign: int) -> np.matrix:
    ghz = np.matrix([1, 0, 0, 0, 0, 0, 0, 0]) + sign * np.matrix([0, 0, 0, 0, 0, 0, 0, 1])
    ghz = ghz / np.sqrt(2)
    ghz = ghz.H 
    return ghz

def generate_examples_Werner_states(p: float) -> Tuple[DensityMatrix, float]:
    # randome choose |GHZ>+ or |GHZ>-
    if np.random.random() < 0.5:
        ghz = GHZ(1)
    else:
        ghz = GHZ(-1)
    rho = p * ghz @ ghz.H
    rho += (1 - p) * np.eye(8) / 8
    dm = DensityMatrix.build(rho)
    if dm is None:
        raise ValueError
    return dm, 4 * np.sqrt(2) * p

def run_Werner_states():
    for p in np.linspace(0, 1, 20, dtype=np.float32):
        rho, s = generate_examples_Werner_states(p)
        print(f"p = {p: .7f}")
        opt = s
        est = Svetlichny_qnn(rho)
        print(f"OPT = {opt: .7f}")
        print(f"EST = {est: .7f}")
        if np.abs(opt - est) > 1e-5:
            print(f"Error! p = {p:.7f}")
            break


def generate_examples_GHZ_symmetric_states(p: float, q: float) -> Tuple[DensityMatrix, float]:
    if q < -1 / (4 * np.sqrt(3)) - EPS or q > np.sqrt(4) / 4 + EPS:
        raise ValueError("q not in the range.")
    if np.abs(p) > np.sqrt(3) / 2 * q + 1 / 8 + EPS:
        raise ValueError("p not in the range.")

    ghz_p = GHZ(1)
    ghz_m = GHZ(-1)
    rho =  (2 * q / np.sqrt(3) + p) * ghz_p @ ghz_p.H
    rho += (2 * q / np.sqrt(3) - p) * ghz_m @ ghz_m.H
    rho += (1 - 4 * q / np.sqrt(3)) * np.eye(8) / 8
    dm = DensityMatrix.build(rho)
    if dm is None:
        raise ValueError
    return dm, 8 * np.sqrt(2) * np.abs(p)

def run_GHZ_symmetric_states():
    # -1 / (4 * sqrt(3)) <= q <= sqrt(3) / 4
    # |p| <= sqrt(3) * q / 2 + 1 / 8
    q_min = -1 / (4 * np.sqrt(3))
    q_max = np.sqrt(3) / 4

    for q in np.linspace(q_min, q_max, 20, dtype=np.float32):
        p_max = np.sqrt(3) * q / 2 + 1 / 8
        for p in np.linspace(-p_max, p_max, 20, dtype=np.float32):
            rho, s = generate_examples_GHZ_symmetric_states(p, q)
            print(f"(p, q) = ({p: .7f}, {q: .7f})")
            opt = s
            est = Svetlichny_qnn(rho)
            print(f"OPT = {opt: .7f}")
            print(f"EST = {est: .7f}")
            if np.abs(opt - est) > 1e-5:
                print(f"Error! (p, q) = ({p:.7f}, {q:.7f})")
                break


# def from_correlation_matrices3(T1: np.matrix, T2: np.matrix, T3: np.matrix) -> Optional[np.matrix]:
#     """Construct density operator from correlation matrices.

#     Args:
#         T1 (np.matrix): t[i,j,1]
#         T2 (np.matrix): t[i,j,2]
#         T3 (np.matrix): t[i,j,3]

#     Raises:
#         ValueError: T1, T2 and T3 must be 3x3 matrices.
#         TypeError: T1, T2 and T3 must be real matrices.

#     Returns:
#         np.matrix: Density operator.

#         .. math::
#             \rho = \frac{I}{8} + \frac{1}{8} \sum_{i,j,k=1}^3 t_{i,j,k} \sigma_i \otimes \sigma_j \otimes \sigma_k
#     """
    
#     if not check_dim(T1, 3, 3) or not check_dim(T2, 3, 3) or not check_dim(T3, 3, 3):
#         raise ValueError("T1, T2 and T3 should be 3x3 matrices.")
#     if not np.allclose(T1.imag, np.zeros((3,3))) or not np.allclose(T2.imag, np.zeros((3,3))) or not np.allclose(T3.imag, np.zeros((3,3))):
#         raise TypeError("T1, T2 and T3 should be real matrices.")
    
#     X = np.matrix([[0, 1],
#                    [1, 0]])
#     Y = np.matrix([[0, -1j],
#                    [1j, 0]])
#     Z = np.matrix([[1, 0],
#                    [0, -1]])
#     sigma = [X, Y, Z]
#     T = [T1, T2, T3]
#     rho = np.eye(8, 8, dtype=np.complex64)
#     rho = np.asmatrix(rho)

#     for i in range(3):
#         for j in range(3):
#             for k in range(3):
#                 rho += T[k][i, j] * np.kron(np.kron(sigma[i], sigma[j]), sigma[k])
    
#     rho /= rho.trace()

#     if not check_positive(rho):
#         return None
#     else:
#         return rho

# def from_correlation_matrices3_unchecked(T1: np.matrix, T2: np.matrix, T3: np.matrix) -> np.matrix:
#     if not check_dim(T1, 3, 3) or not check_dim(T2, 3, 3) or not check_dim(T3, 3, 3):
#         raise ValueError("T1, T2 and T3 should be 3x3 matrices.")
#     if not np.allclose(T1.imag, np.zeros((3,3))) or not np.allclose(T2.imag, np.zeros((3,3))) or not np.allclose(T3.imag, np.zeros((3,3))):
#         raise TypeError("T1, T2 and T3 should be real matrices.")
    
#     X = np.matrix([[0, 1],
#                    [1, 0]])
#     Y = np.matrix([[0, -1j],
#                    [1j, 0]])
#     Z = np.matrix([[1, 0],
#                    [0, -1]])
#     sigma = [X, Y, Z]
#     T = [T1, T2, T3]
#     rho = np.eye(8, 8, dtype=np.complex64)
#     rho = np.asmatrix(rho)

#     for i in range(3):
#         for j in range(3):
#             for k in range(3):
#                 rho += T[k][i, j] * np.kron(np.kron(sigma[i], sigma[j]), sigma[k])
    
#     rho /= rho.trace()
#     return rho

# def from_correlation_matrices3_safe(T1: np.matrix, T2: np.matrix, T3: np.matrix) -> np.matrix:
#     if not check_dim(T1, 3, 3) or not check_dim(T2, 3, 3) or not check_dim(T3, 3, 3):
#         raise ValueError("T1, T2 and T3 should be 3x3 matrices.")
#     if not np.allclose(T1.imag, np.zeros((3,3))) or not np.allclose(T2.imag, np.zeros((3,3))) or not np.allclose(T3.imag, np.zeros((3,3))):
#         raise TypeError("T1, T2 and T3 should be real matrices.")
    
#     X = np.matrix([[0, 1],
#                    [1, 0]])
#     Y = np.matrix([[0, -1j],
#                    [1j, 0]])
#     Z = np.matrix([[1, 0],
#                    [0, -1]])
#     sigma = [X, Y, Z]
#     T = [T1, T2, T3]
    
#     rho = np.zeros(shape=(8, 8), dtype=np.complex64)
#     rho = np.asmatrix(rho)

#     for i in range(3):
#         for j in range(3):
#             for k in range(3):
#                 rho += T[k][i, j] * np.kron(np.kron(sigma[i], sigma[j]), sigma[k])
    
#     eigens = np.linalg.eigvals(rho)
#     eigen_min = eigens.real.min()
#     if eigen_min < 0:
#         rho -= eigen_min * np.eye(8, dtype=np.complex64)
    
#     rho /= rho.trace()
#     return rho


# def to_correlation_matrices3(rho: np.matrix) -> Tuple[np.matrix, np.matrix, np.matrix]:
#     if not check_positive(rho):
#         raise ValueError("rho must be positive.")
#     if not np.allclose(rho.trace(), 1):
#         raise ValueError("The trace of rho must be 1.")
    
#     X = np.matrix([[0, 1],
#                    [1, 0]])
#     Y = np.matrix([[0, -1j],
#                    [1j, 0]])
#     Z = np.matrix([[1, 0],
#                    [0, -1]])
#     sigma = [X, Y, Z]
#     T = np.zeros((3,3,3), dtype=np.float64)

#     for i in range(3):
#         for j in range(3):
#             for k in range(3):
#                 T[i,j,k] = (rho @ np.kron(np.kron(sigma[i], sigma[j]), sigma[k])).trace().real
#     return np.asmatrix(T[:, :, 0]), np.asmatrix(T[:, :, 1]), np.asmatrix(T[:, :, 2])

# def generate_examples_onlyOne_correlation_matrix(m: int = 3) -> Tuple[np.matrix, float]:
#     T = generate_real_positive(3, m)
#     eigens = np.linalg.eigvals(T.T @ T)
#     eigens = eigens.real
#     eig_max = eigens.max()

#     p = np.random.random()
#     if p < 1/3:
#         rho = from_correlation_matrices3(T, np.zeros((3,3)), np.zeros((3,3)))
#     elif p < 2 / 3:
#         rho = from_correlation_matrices3(np.zeros((3,3)), T, np.zeros((3,3)))
#     else:
#         rho = from_correlation_matrices3(np.zeros((3,3)), np.zeros((3,3)), T)
    
#     if rho is None:
#         return generate_examples_onlyOne_correlation_matrix(m)
#     else:
#         return rho, 4 * np.sqrt(eig_max)

# def run_onlyOne_correlation_matrix():
#     m = 3
#     for i in range(100):
#         rho, s = generate_examples_onlyOne_correlation_matrix(m)
#         opt = s
#         est = Svetlichny_qnn(rho)
#         print(f"i = {i:5}")
#         print(f"OPT = {opt:.7f}")
#         print(f"EST = {est:.7f}")

#         if np.abs(opt - est) > 1e-5:
#             print(f"Error! rho is:")
#             print(rho)
#             break


def generate_examples_sphere(theta: np.float32, phi: np.float32) -> DensityMatrix:
    psi1 = np.matrix([0, 1, 0, 0, 0, 0, 0, 0]) # row vector
    psi2 = np.matrix([0, 0, 1, 0, 0, 0, 0, 0])
    psi3 = np.matrix([0, 0, 0, 0, 1, 0, 0, 0])
    a = np.sin(theta) * np.cos(phi)
    b = np.sin(theta) * np.sin(phi)
    c = np.cos(theta)
    psi = a * psi1 + b * psi2 + c * psi3
    rho = psi.H @ psi   # col * row
    return DensityMatrix(rho)

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def run_sphere():
    n1, n2 = 50, 50

    thetas = np.linspace(0, np.pi / 2, n1, dtype=np.float32)
    phis = np.linspace(0, np.pi / 2, n2, dtype=np.float32)
    estimates = []
    for theta in thetas:
        row = []
        for phi in phis:
            rho = generate_examples_sphere(theta, phi)
            est = Svetlichny_qnn(rho)
            row.append(est)
            # if np.abs(est - 4.0) > 1e-3:
            #     print(f"({theta:.7f}, {phi:.7f}): {est:.7f}")
        estimates.append(row)

    # store data
    data = dict()
    data["theta"] = thetas
    data["phi"] = phis
    data["estimate"] = estimates

    with open(f"estimate_{n1}x{n2}_pi_div_2.json", "w") as f:
        json.dump(data, f, cls=MyEncoder)

    pd = pandas.DataFrame(estimates, columns=None)
    pd.to_csv(f"estimate_{n1}x{n2}_pi_div_2.csv", header=None, index=0)


def three_entanglement_pure(psi: np.ndarray) -> float:
    if psi.shape != (8, ):
        raise ValueError("Only consider pure 3-qubit state.")
    
    def a(x: int, y: int, z: int):
        return psi[(x << 2) + (y << 1) + z]
    
    d1 = a(0,0,0)**2 * a(1,1,1)**2 + a(0,0,1)**2 * a(1,1,0)**2 + a(0,1,0)**2 * a(1,0,1)**2 + a(1,0,0)**2 * a(0,1,1)**2
    d2 = a(0,0,0) * a(1,1,1) * a(0,1,1) * a(1,0,0) \
        + a(0,0,0) * a(1,1,1) * a(1,0,1) * a(0,1,0) \
        + a(0,0,0) * a(1,1,1) * a(1,1,0) * a(0,0,1) \
        + a(0,1,1) * a(1,0,0) * a(1,0,1) * a(0,1,0) \
        + a(0,1,1) * a(1,0,0) * a(1,1,0) * a(0,0,1) \
        + a(1,0,1) * a(0,1,0) * a(1,1,0) * a(0,0,1)
    d3 = a(0,0,0) * a(1,1,0) * a(1,0,1) * a(0,1,1) + a(1,1,1) * a(0,0,1) * a(0,1,0) * a(1,0,0)

    return 4 * np.abs(d1 - 2 * d2 + 4 * d3)


if __name__ == "__main__":
    # for m in range(1, 4):
    #     p = count_nonlocality(
    #         data_generator=lambda : rand_density_matrix(3, m),
    #         solver=Svetlichny_qnn,
    #         n=200,
    #         threshold=4.0
    #         )
    #     print(f"m = {m}, p = {p}")

    n = 100
    for i in range(n):
        psi = rand_state(3)
        entanglement = three_entanglement_pure(psi)

        psi = np.asmatrix(psi)
        rho = psi.H @ psi
        dm = DensityMatrix(rho)
        s = Svetlichny_qnn(dm)

        print(f"entanglement = {entanglement:.7f}")
        print(f"nonlocality  = {s:.7f}")
