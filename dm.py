from typing import Optional, Tuple
import numpy as np

EPS = 1e-5
I = np.matrix([[1, 0],
               [0, 1]])
X = np.matrix([[0, 1],
               [1, 0]])
Y = np.matrix([[0, -1j],
               [1j, 0]])
Z = np.matrix([[1, 0],
               [0, -1]])
SIGMA = [I, X, Y, Z]

class DensityMatrix:
    def __init__(self, mat: np.matrix) -> None:
        self.rho = mat
    
    @staticmethod
    def build(mat: np.ndarray):
        mat = np.asmatrix(mat)
        ret = DensityMatrix(mat)
        if DensityMatrix.is_valid(ret.rho):
            return ret
        else:
            return None

    @staticmethod
    def is_postive(mat: np.matrix) -> bool:
        eigens = np.linalg.eigvals(mat)
        eigens_re = eigens.real

        # eigens should be real
        if not np.allclose(eigens, eigens_re):
            return False
        
        eigens_re.sort()
        return eigens_re[0] > -EPS
    
    @staticmethod
    def is_valid(mat: np.matrix) -> bool:
        """Check whether a matrix is a valid density operator.
        It should be positive and its trace should be equal to 1.

        Returns:
            bool: Whether a matrix is a valid density operator.
        """
        (n, m) = mat.shape
        if n != m:  # n should be equal to m
            return False

        lowbit = n & (-n)
        if n != lowbit: # n should be 2^x
            return False
        
        return DensityMatrix.is_postive(mat) and np.allclose(mat.trace(), 1)
    
    def get_eigens(self) -> np.ndarray:
        eigens = np.linalg.eigvals(self.rho).real
        eigens.sort()
        return eigens
    
    def to_correlation_matrix_q2(self) -> np.matrix:
        T = np.zeros(shape=(3,3), dtype=np.float64)
        for i in range(1, 4):
            for j in range(1, 4):
                T[i-1, j-1] = (self.rho @ np.kron(SIGMA[i], SIGMA[j])).trace().real
        return np.asmatrix(T)
    
    def to_correlation_matrices_q3(self) -> Tuple[np.matrix, np.matrix, np.matrix]:
        T = np.zeros(shape=(3, 3, 3), dtype=np.float64)
        
        for i in range(1, 4):
            for j in range(1, 4):
                for k in range(1, 4):
                    T[i-1, j-1, k-1] = (self.rho @ np.kron(np.kron(SIGMA[i], SIGMA[j]), SIGMA[k])).trace().real
        
        return np.asmatrix(T[:, :, 0]), np.asmatrix(T[:, :, 1]), np.asmatrix(T[:, :, 2])
    
    @staticmethod
    def from_correlation_matrices_q3(T1: np.matrix, T2: np.matrix, T3: np.matrix):
        if not T1.shape == (3,3) or not T2.shape == (3,3) or not T3.shape == (3,3):
            raise ValueError("T1, T2 and T3's shape should be (3, 3).")
        if not np.allclose(T1, T1.real) or not np.allclose(T2, T2.real) or not np.allclose(T3, T3.real):
            raise ValueError("T1, T2 and T3 should be real matrices.")
        
        rho = np.eye(8, dtype=np.complex64)
        rho = np.asmatrix(rho)
        T = [T1, T2, T3]

        for i in range(1, 4):
            for j in range(1, 4):
                for k in range(1, 4):
                    rho += T[k-1][i-1, j-1] * np.kron(np.kron(SIGMA[i], SIGMA[j]), SIGMA[k])
        
        rho = rho / 8
        return DensityMatrix.build(rho)


if __name__ == "__main__":
    while True:
        T1 = np.random.uniform(-1, 1, size=(3,3))
        T2 = T3 = np.zeros(shape=(3,3))

        dm = DensityMatrix.from_correlation_matrices_q3(T1, T2, T3)
        if dm is not None:
            # print(f"{dm.rho}")
            (t1, t2, t3) = dm.to_correlation_matrices_q3()
            if np.allclose(T1, t1) and np.allclose(T2, t2) and np.allclose(T3, t3):
                print("true!")
            else:
                print("false!")
            break
    
