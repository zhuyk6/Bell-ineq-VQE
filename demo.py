from mindquantum.core import gates as G
from mindquantum.core.circuit import Circuit
from mindquantum.simulator import Simulator
from mindquantum.core.operators import Hamiltonian, QubitOperator
from mindquantum.framework import MQAnsatzOnlyOps
from mindspore import nn
import mindspore as ms
import numpy as np

ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")

circ1 = Circuit().rx('theta1', 0).ry('theta3', 0)
circ2 = Circuit().rx('theta1', 0).ry('theta4', 0)
circ3 = Circuit().rx('theta2', 0).ry('theta3', 0)
circ4 = Circuit().rx('theta2', 0).ry('theta4', 0)

h1 = Hamiltonian(QubitOperator('Z0'))
h2 = Hamiltonian(QubitOperator('Z0'))
h3 = Hamiltonian(QubitOperator('Z0'))
h4 = Hamiltonian(QubitOperator('Z0'))

grad_ops1 = Simulator('mqvector', circ1.n_qubits).get_expectation_with_grad(h1, circ1)
grad_ops2 = Simulator('mqvector', circ1.n_qubits).get_expectation_with_grad(h2, circ2)
grad_ops3 = Simulator('mqvector', circ1.n_qubits).get_expectation_with_grad(h3, circ3)
grad_ops4 = Simulator('mqvector', circ1.n_qubits).get_expectation_with_grad(h4, circ4)


class QNet(nn.Cell):
    def __init__(self):
        super(QNet, self).__init__()
        self.w = ms.Parameter(ms.Tensor(np.random.uniform(-3, 3, 4).astype(np.float32)), name='weight')
        self.q_ops1 = MQAnsatzOnlyOps(grad_ops1)
        self.q_ops2 = MQAnsatzOnlyOps(grad_ops2)
        self.q_ops3 = MQAnsatzOnlyOps(grad_ops3)
        self.q_ops4 = MQAnsatzOnlyOps(grad_ops4)

    def construct(self):
        e1 = self.q_ops1(self.w[[0, 2]])
        e2 = self.q_ops2(self.w[[0, 3]])
        e3 = self.q_ops3(self.w[[1, 2]])
        e4 = self.q_ops4(self.w[[1, 3]])
        return -(e1 + e2 + e3 + e4)


qnet = QNet()
opti = nn.Adam(qnet.trainable_params(), learning_rate=0.1)
train_net = nn.TrainOneStepCell(qnet, opti)
for i in range(100):
    f = train_net()
    print(-f)
