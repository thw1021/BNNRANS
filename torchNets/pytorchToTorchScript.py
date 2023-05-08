import argparse
import torch as th
from torch.nn.parameter import Parameter
import os

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

dtype = th.FloatTensor

base_path = os.getcwd()
class TurbNN(th.nn.Module):
    """
    Note: This must be fully consistent with the architecture
    """
    def __init__(self, D_in, H, D_out):
        """
        Architecture of the turbulence deep neural net
        Args:
            D_in (Int) = Number of input parameters
            H (Int) = Number of hidden paramters
            D_out (Int) = Number of output parameters
        """
        super(TurbNN, self).__init__()
        self.linear1 = th.nn.Linear(D_in, H)
        self.f1 = th.nn.LeakyReLU()
        self.linear2 = th.nn.Linear(H, H)
        self.f2 = th.nn.LeakyReLU()
        self.linear3 = th.nn.Linear(H, H)
        self.f3 = th.nn.LeakyReLU()
        self.linear4 = th.nn.Linear(H, H)
        self.f4 = th.nn.LeakyReLU()
        self.linear5 = th.nn.Linear(H, int(H/5))
        self.f5 = th.nn.LeakyReLU()
        self.linear6 = th.nn.Linear(int(H/5), int(H/10))
        self.f6 = th.nn.LeakyReLU()
        self.linear7 = th.nn.Linear(int(H/10), D_out)
        self.log_beta = Parameter(th.Tensor([1.]).type(dtype))

    def forward(self, x):
        """
        Forward pass of the neural network
        Args:
            x (th.DoubleTensor): [N x D_in] column matrix of training inputs
        Returns:
            out (th.DoubleTensor): [N x D_out] matrix of neural network outputs
        """
        lin1 = self.f1(self.linear1(x))
        lin2 = self.f2(self.linear2(lin1))
        lin3 = self.f3(self.linear3(lin2))
        lin4 = self.f4(self.linear4(lin3))
        lin5 = self.f5(self.linear5(lin4))
        lin6 = self.f6(self.linear6(lin5))
        out = self.linear7(lin6)

        return out
    def loadNeuralNet(self, filename):
        '''
        Load the current neural network state
        Args:
            filename (string): name of the file to save the neural network in

        For saving/loading across devices, see https://blog.csdn.net/bc521bc/article/details/85623515
        '''
        self.load_state_dict(th.load(filename, map_location=lambda storage, loc: storage))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script used to convert PyTorch model to Torch Script.")
    parser.add_argument("-p", "--Path", type=str, default="./torchNets/",
                        help="path to PyTorch neural network model (should end with backslash )")
    # parser.add_argument("-n", "--networkNumber", type=int, default=0,
    #                     help="number of neural network model")

    args = parser.parse_args()
    netDir = args.Path
    # netNum = args.networkNumber

    # First load up the neural network
    # Must be consistent with the saved neural network
    for netNum in range(20):
        turb_nn = TurbNN(D_in=5, H=200, D_out=10)

        print('Reading PyTorch network : {}/foamNet-{:d}.pth'.format(netDir, netNum))
        turb_nn.loadNeuralNet('{}/foamNet-{:d}.pth'.format(netDir, netNum))
        turb_nn.eval()

        example_input = th.rand(1, 5)
        traced_turb_nn = th.jit.trace(turb_nn, example_input)
        example_output = traced_turb_nn(th.ones(1, 5))

        if not os.path.exists(base_path+"/turb_nn/"):
            print(bcolors.WARNING+"do not find turb_nn directory in {}, make one now".format(base_path)+bcolors.ENDC)
            os.mkdir(base_path+"/turb_nn")

        traced_turb_nn.save("{}/turb_nn_{:d}.pt".format("./turb_nn", netNum))

        print(example_output)
