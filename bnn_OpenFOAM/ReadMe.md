### Implement PyTorch model into OpenFOAM

#### Compile code

1. compile OpenFOAM.

2. download `libtorch` from [PyTorch](https://pytorch.org/) and extract the zip file to `/opt`. set `export TORCH_LIBRARIES=/opt/libtorch` in `.bashrc`.

   **Note**: the `TORCH_LIBRARIES` is added following this [guide](https://ml-cfd.com/openfoam/pytorch/docker/2020/12/29/running-pytorch-models-in-openfoam.html) and you can also refer to this [repo](https://github.com/AndreWeiner/of_pytorch_docker/blob/master/Dockerfile).

3. Make sure to use  c++14 (**Check `c++` file firstly**)

   ```bash
   # For OpenFOAM Foundation OpenFOAM4.x
   sed -i "s/-std=c++0x/-std=c++14/g" OpenFOAM-7/wmake/rules/linux64Gcc/c++ && \
   sed -i "s/-Wold-style-cast/-Wno-old-style-cast/g" OpenFOAM-7/wmake/rules/linux64Gcc/c++

   # For OpenFOAM Foundation OpenFOAM7.0
   sed -i "s/-std=c++11/-std=c++14/g" OpenFOAM-7/wmake/rules/linux64Gcc/c++ && \
   sed -i "s/-Wold-style-cast/-Wno-old-style-cast/g" OpenFOAM-7/wmake/rules/linux64Gcc/c++

   # For ESI OpenFOAM
   sed -i "s/-std=c++11/-std=c++14/g" OpenFOAM-7/wmake/rules/General/Gcc/c++ && \
   sed -i "s/-Wold-style-cast/-Wno-old-style-cast/g" OpenFOAM-7/wmake/rules/General/Gcc/c++
   ```

4. Add `libtorch` path to global compile setting `OpenFOAM-7/wmake/rules/General/general b/wmake/rules/General/general`

   ```
   # For OpenFOAM Foundation OpenFOAM7.0
   GINC = -I$(TORCH_LIBRARIES)/include \
          -I$(TORCH_LIBRARIES)/include/torch/csrc/api/include
   ```

5. copy `reynoldsNet` into `OpenFOAM-7/src/TurbulenceModels/turbulenceModels`.

   modify `reynoldsNet/Make/files` as follows

   ```
   reynoldsNet.C

   LIB = $(FOAM_LIBBIN)/libreynoldsNet
   ```

   modify `reynoldsNet/Make/options` as follows

   ```
   EXE_INC = \
       -I$(LIB_SRC)/finiteVolume/lnInclude \
       -I$(LIB_SRC)/meshTools/lnInclude \
       -I$(TORCH_LIBRARIES)/include \
       -I$(TORCH_LIBRARIES)/include/torch/csrc/api/include

   LIB_LIBS = \
       -lpthread -ldl -lrt \
       -lfiniteVolume \
       -lmeshTools \
       -Wl,-rpath,$(TORCH_LIBRARIES)/lib $(TORCH_LIBRARIES)/lib/libtorch.so $(TORCH_LIBRARIES)/lib/libc10.so \
       -Wl,--no-as-needed,$(TORCH_LIBRARIES)/lib/libtorch_cpu.so \
       -Wl,--as-needed $(TORCH_LIBRARIES)/lib/libc10.so \
       -Wl,--no-as-needed,$(TORCH_LIBRARIES)/lib/libtorch.so
   ```

6. run `wclean && wmake` in `reynoldsNet` folder to generate `libreynoldsNet.so` library.

7. copy `linearViscousStress` into `OpenFOAM-7/src/TurbulenceModels/turbulenceModels`.

8. modify `OpenFOAM-7/src/TurbulenceModels/turbulenceModels/Make/options` as follows

   ```
   EXE_INC = \
       -I$(LIB_SRC)/finiteVolume/lnInclude \
       -I$(LIB_SRC)/meshTools/lnInclude \

   LIB_LIBS = \
       -lfiniteVolume \
       -lmeshTools \
       -lreynoldsNet
   ```

9. copy `IncompressibleTurbulenceModel` folder into `OpenFOAM-7/src/TurbulenceModels/incompressible` to cover original files.

10. copy `incompressibleTurbulenceModel.H` into `OpenFOAM-7/src/TurbulenceModels/incompressible` to cover original files.

11. modify `OpenFOAM-7/src/TurbulenceModels/incompressible/Make/options` as follows

    ```
    EXE_INC = \
        -I../turbulenceModels/lnInclude \
        -I$(LIB_SRC)/transportModels \
        -I$(LIB_SRC)/finiteVolume/lnInclude \
        -I$(LIB_SRC)/meshTools/lnInclude

    LIB_LIBS = \
        -lincompressibleTransportModels \
        -lturbulenceModels \
        -lfiniteVolume \
        -lmeshTools \
        -lreynoldsNet

    ```

12. delete `OpenFOAM-7/src/TurbulenceModels/turbulenceModels/lnInclude` folder and regenerate by `wmakeLnInclude -u ../turbulenceModels` in `OpenFOAM-7/src/TurbulenceModels/incompressible`

13. compile the solver.
14. recompile OpenFOAM.


**Note**: If using docker, you may get this [problem](https://bugs.openfoam.org/view.php?id=3163) when running a simulation in parallel. Solution can be found [here](https://github.com/open-mpi/ompi/issues/4948#issuecomment-395468231), i.e., put `export OMPI_MCA_btl_vader_single_copy_mechanism=none` in your `.bashrc` file.

#### Convert PyTorch model into torch script

Sample code

```python
import argparse
import torch as th
from torch.nn.parameter import Parameter

dtype = th.FloatTensor
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
    parser.add_argument("-n", "--networkNumber", type=int, default=0,
                        help="number of neural network model")

    args = parser.parse_args()
    netDir = args.Path
    netNum = args.networkNumber

    # First load up the neural network
    # Must be same with the saved neural network
    turb_nn = TurbNN(D_in=5, H=200, D_out=10)

    print('Reading PyTorch network : {}/foamNet-{:d}.pth'.format(netDir, netNum))
    turb_nn.loadNeuralNet('{}/foamNet-{:d}.pth'.format(netDir, netNum))
    turb_nn.eval()

    example_input = th.rand(1, 5)
    traced_turb_nn = th.jit.trace(turb_nn, example_input)
    example_output = traced_turb_nn(th.ones(1, 5))
    traced_turb_nn.save("turb_nn.pt")

    print(example_output)
```

#### Load torch script model in OpenFOAM

Sample code

`reynoldsNet.C`

```C++
/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2013-2016 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.
\*---------------------------------------------------------------------------*/
#include "reynoldsNet.H"

torch::jit::script::Module turb_nn_module;

void torch_script::reynoldsNet::readNetFromFile() const {
    std::string turb_nn_model_file = "./turb_nn.pt";
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        turb_nn_module = torch::jit::load(turb_nn_model_file);
        std::cout << "Successfully load tensor basis neural network." << std::endl;
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        exit (EXIT_FAILURE);
  }

}

std::vector<float> torch_script::reynoldsNet::run_forward(std::vector<float> inputdata) const {
    auto opts = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor input = torch::from_blob(inputdata.data(), {1, (int)inputdata.size()}, opts).to(torch::kFloat64);
    std::vector<torch::jit::IValue> input_invar{input};
    at::Tensor output = turb_nn_module.forward(input_invar).toTensor();

    std::vector<float> out_vect(output.data_ptr<float>(), output.data_ptr<float>() + output.numel());

    return out_vect;
}
```

`reynoldsNet.H`

```c++
/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2013-2016 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.
Class
    torch_script

Group
    n/a

Description
    Loads neural net for deviatoric Reynolds stress calculation

SourceFiles
    reynoldsNet.C
\*---------------------------------------------------------------------------*/
#ifndef reynoldsNet_H__
#define reynoldsNet_H__

#pragma push_macro("TypeName")
#undef TypeName

#include "torch/script.h"

#pragma pop_macro("TypeName")

namespace torch_script{

class reynoldsNet {

  protected:
    // Protected data

  public:

  void readNetFromFile() const;
  // get data driven prediction
  std::vector<float> run_forward(std::vector<float> inputdata) const;
};
}  // namespace torch_script

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //

```

