from dm import *

import numpy as np
from mindquantum.simulator import Simulator
from mindquantum.core.circuit import Circuit
from mindquantum.core.parameterresolver import ParameterResolver
from mindquantum.core.operators import Hamiltonian, QubitOperator
from mindquantum.core.gates import RZ, RY
from mindquantum.framework import MQAnsatzOnlyOps

import mindspore as ms
ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")

def generate_circuit2(p1: int, p2: int) -> Circuit:
    circ = Circuit()

    circ += RZ(pr=ParameterResolver(
        data={f"phi{p1}": -1},
        const=np.pi,
    )).on(0)
    circ += RY(f"theta{p1}").on(0)

    circ += RZ(pr=ParameterResolver(
        data={f"phi{p2}": -1},
        const=np.pi,
    )).on(1)
    circ += RY(f"theta{p2}").on(1)

    return circ

def generate_circuit3(p1: int, p2: int, p3: int) -> Circuit:
    circ = Circuit()

    circ += RZ(pr=ParameterResolver(
        data={f"phi{p1}": -1},
        const=np.pi,
    )).on(0)
    circ += RY(f"theta{p1}").on(0)

    circ += RZ(pr=ParameterResolver(
        data={f"phi{p2}": -1},
        const=np.pi,
    )).on(1)
    circ += RY(f"theta{p2}").on(1)

    circ += RZ(pr=ParameterResolver(
        data={f"phi{p3}": -1},
        const=np.pi,
    )).on(2)
    circ += RY(f"theta{p3}").on(2)

    return circ


class QNet2(ms.nn.Cell):
    def __init__(self, grad_ops1, grad_ops2, grad_ops3, grad_ops4) -> None:
        super().__init__()
        self.w = ms.Parameter(ms.Tensor(np.random.uniform(-np.pi, np.pi, 8).astype(np.float32)), name="weight")
        self.q_ops1 = MQAnsatzOnlyOps(grad_ops1)
        self.q_ops2 = MQAnsatzOnlyOps(grad_ops2)
        self.q_ops3 = MQAnsatzOnlyOps(grad_ops3)
        self.q_ops4 = MQAnsatzOnlyOps(grad_ops4)
    
    def construct(self):
        # w[[0, 1]] -- theta1, phi1
        # w[[2, 3]] -- theta2, phi2
        # w[[4, 5]] -- theta3, phi3
        # w[[6, 7]] -- theta4, phi4
        e1 = self.q_ops1(self.w[[0, 1, 4, 5]])
        e2 = self.q_ops2(self.w[[0, 1, 6, 7]])
        e3 = self.q_ops3(self.w[[2, 3, 4, 5]])
        e4 = self.q_ops4(self.w[[2, 3, 6, 7]])
        return -(e1 + e2 + e3 - e4)

def num_to_params3(n: int):
    b1 = 1 if (n & 4) > 0 else 0
    b2 = 3 if (n & 2) > 0 else 2
    b3 = 5 if (n & 1) > 0 else 4
    return b1, b2, b3

class QNet3(ms.nn.Cell):
    def __init__(self, grads) -> None:
        if len(grads) != 8:
            raise ValueError("There should be 8 grad_ops.")
        
        super().__init__()
        self.w = ms.Parameter(ms.Tensor(np.random.uniform(-np.pi, np.pi, 12).astype(np.float32)), name="weight")
        
        self.op0 = MQAnsatzOnlyOps( grads[0] )
        self.op1 = MQAnsatzOnlyOps( grads[1] )
        self.op2 = MQAnsatzOnlyOps( grads[2] )
        self.op3 = MQAnsatzOnlyOps( grads[3] )
        self.op4 = MQAnsatzOnlyOps( grads[4] )
        self.op5 = MQAnsatzOnlyOps( grads[5] )
        self.op6 = MQAnsatzOnlyOps( grads[6] )
        self.op7 = MQAnsatzOnlyOps( grads[7] )
        self.ops = [self.op0, self.op1, self.op2, self.op3, self.op4, self.op5, self.op6, self.op7]
    
    def construct(self):
        def g(n: int):
            b1, b2, b3 = num_to_params3(n)
            return [b1*2, b1*2+1, b2*2, b2*2+1, b3*2, b3*2+1]

        def f(n: int):
            return self.ops[n](self.w[g(n)])
        
        return -( f(0) + f(1) + f(2) - f(3) - f(4) + f(5) + f(6) + f(7) )

def check_dim(P: np.matrix, n: int, m: int) -> bool:
    return P.shape == (n, m)

def CHSH_qnn(dm: DensityMatrix):
    rho = dm.rho
    if not check_dim(rho, 4, 4):
        raise ValueError("CHSH only considers 2 qubits.")

    sim = Simulator(backend="mqmatrix", n_qubits=2)
    rho = np.asarray(rho)
    sim.set_qs(rho)

    circ1 = generate_circuit2(1, 3)
    circ2 = generate_circuit2(1, 4)
    circ3 = generate_circuit2(2, 3)
    circ4 = generate_circuit2(2, 4)

    ham = Hamiltonian(QubitOperator("Z0 Z1", 1))

    grad_ops1 = sim.get_expectation_with_grad(ham, circ1)
    grad_ops2 = sim.get_expectation_with_grad(ham, circ2)
    grad_ops3 = sim.get_expectation_with_grad(ham, circ3)
    grad_ops4 = sim.get_expectation_with_grad(ham, circ4)

    qnet = QNet2(grad_ops1, grad_ops2, grad_ops3, grad_ops4)
    optimizer = ms.nn.Adam(qnet.trainable_params(), learning_rate=0.1)
    train_net = ms.nn.TrainOneStepCell(qnet, optimizer)

    last = np.float32(0)
    for i in range(1000):
        f = train_net().asnumpy()
        if i % 10 == 0:
            # print(f"epoch = {i: 5}\testimite={-f[0]:.7f}")
            if np.abs(f[0] - last) < 1e-7:
                break
            last = f[0]
    return -last

def Svetlichny_qnn(dm: DensityMatrix):
    rho = dm.rho
    if not check_dim(rho, 8, 8):
        raise ValueError("Svetlichny considers only 3 qubits.")
    
    sim = Simulator(backend="mqmatrix", n_qubits=3)
    rho = np.asarray(rho)
    sim.set_qs(rho)

    circs = []
    for i in range(8):
        b1, b2, b3 = num_to_params3(i)
        circs.append(generate_circuit3(b1, b2, b3))

    ham = Hamiltonian(QubitOperator("Z0 Z1 Z2", 1))

    grad_ops = []
    for i in range(8):
        grad_ops.append(sim.get_expectation_with_grad(ham, circs[i]))

    qnet = QNet3(grads=grad_ops)

    optimizer = ms.nn.Adam(qnet.trainable_params(), learning_rate=0.1)
    train_net = ms.nn.TrainOneStepCell(qnet, optimizer)

    last = np.float32(0)
    for i in range(1000):
        f = train_net().asnumpy()
        if i % 10 == 0:
            # print(f"epoch = {i: 5}\testimite={-f[0]:.7f}")
            if np.abs(f[0] - last) < 1e-7:
                break
            last = f[0]
    return -last
