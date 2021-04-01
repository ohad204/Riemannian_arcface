import torch
import torch.nn as nn
from torch.autograd import Variable as V
from torch.autograd import Function

# EigLayer
class EigLayerF(Function):
    @staticmethod
    def forward(self, input):
        n = input.shape[0]
        S = torch.zeros(input.shape)
        U = torch.zeros(input.shape)

        for i in range(n):
            value, vector = torch.eig(input[i], eigenvectors=True)
            S[i] = torch.diag(value[:, 0])
            U[i] = vector

        self.save_for_backward(input, S, U)
        return S, U

    @staticmethod
    def backward(self, grad_S, grad_U):
        input, S, U = self.saved_tensors
        n = input.shape[0]
        dim = input.shape[1]

        grad_input = V(torch.zeros(input.shape))

        e = torch.eye(dim)

        P_i = torch.matmul(S, torch.ones(dim, dim))

        P = (P_i - P_i.permute(0, 2, 1)) + e
        epo = (torch.ones(P.shape)) * 0.000001
        P = torch.where(P != 0, P, epo)
        P = (1 / P) - e

        g1 = torch.matmul(U.permute(0, 2, 1), grad_U)
        g1 = (g1 + g1.permute(0, 2, 1)) / 2
        g1 = torch.mul(P.permute(0, 2, 1), g1)
        g1 = 2 * torch.matmul(torch.matmul(U, g1), U.permute(0, 2, 1))
        g2 = torch.matmul(torch.matmul(U, torch.mul(grad_S, e)), U.permute(0, 2, 1))
        grad_input = g1 + g2

        return grad_input

class EigLayer(nn.Module):
    def __init__(self):
        super(EigLayer, self).__init__()

    def forward(self, input1):
        return EigLayerF().apply(input1)


# m_log
class M_Log(nn.Module):
    def __init__(self):
        super(M_Log, self).__init__()
        self.beta = 0.000001

    def forward(self, input1):
        n = input1.shape[0]
        dim = input1.shape[1]

        espison = torch.eye(dim)*self.beta
        espison = espison
        espison = torch.unsqueeze(espison,0)
        input2 = torch.where(input1-espison < 0, espison,input1)

        one = torch.ones(input2.shape)
        e = torch.eye(dim)
        output = torch.log(input2+one-e)

        return output


class MatrixLog(nn.Module):
    def __init__(self):
        super(MatrixLog, self).__init__()
        self.eiglayer = EigLayer()
        self.mlog = M_Log()

    def forward(self, spd):
        S, U = self.eiglayer(spd)
        S_log = self.mlog(S)
        spd_log = torch.matmul(torch.matmul(U, S_log), U.permute(0, 2, 1))

        return spd_log