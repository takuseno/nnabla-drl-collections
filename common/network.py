import nnabla as nn
import nnabla.parametric_functions as PF
import nnabla.functions as F


def nature_head(obs):
    out = PF.convolution(obs, 32, (8, 8), stride=(4, 4), name='conv1')
    out = F.relu(out)
    out = PF.convolution(out, 64, (4, 4), stride=(2, 2), name='conv2')
    out = F.relu(out)
    out = PF.convolution(out, 64, (3, 3), stride=(1, 1), name='conv3')
    out = F.relu(out)
    out = PF.affine(out, 512, name='fc1')
    out = F.relu(out)
    return out


def nips_head(obs):
    out = PF.convolution(obs, 16, (8, 8), stride=(4, 4), name='conv1')
    out = F.relu(out)
    out = PF.convolution(out, 32, (4, 4), stride=(2, 2), name='conv2')
    out = F.relu(out)
    out = PF.affine(out, 256, name='fc1')
    out = F.relu(out)
    return out
