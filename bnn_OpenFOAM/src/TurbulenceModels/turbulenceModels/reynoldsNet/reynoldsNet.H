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
