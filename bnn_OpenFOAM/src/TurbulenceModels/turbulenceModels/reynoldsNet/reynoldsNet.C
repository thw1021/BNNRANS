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
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        exit (EXIT_FAILURE);
  }

}

std::vector<float> torch_script::reynoldsNet::run_forward(std::vector<float> inputdata) const {

    auto opts = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor input = torch::from_blob(inputdata.data(), {1, (int)inputdata.size()}, opts).to(torch::kFloat32);
    std::vector<torch::jit::IValue> input_invar{input};
    at::Tensor output = turb_nn_module.forward(input_invar).toTensor();

    std::vector<float> out_vect(output.data_ptr<float>(), output.data_ptr<float>() + output.numel());

    return out_vect;
}