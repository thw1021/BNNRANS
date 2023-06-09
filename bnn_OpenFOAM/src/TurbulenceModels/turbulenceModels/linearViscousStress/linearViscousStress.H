/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2013-2018 OpenFOAM Foundation
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
    Foam::linearViscousStress

Description
    Linear viscous stress turbulence model base class

SourceFiles
    linearViscousStress.C

\*---------------------------------------------------------------------------*/

#ifndef linearViscousStress_H
#define linearViscousStress_H
#include <vector>
#include "reynoldsNet.H"
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                           Class linearViscousStress Declaration
\*---------------------------------------------------------------------------*/

template<class BasicTurbulenceModel>
class linearViscousStress
:
    public BasicTurbulenceModel
{

public:

    friend class torch_script::reynoldsNet;
    torch_script::reynoldsNet rn;

    // Fields
    volTensorField ident_;

    mutable volTensorField b0;
    mutable volTensorField a_dd;
    mutable volTensorField a_0;
    mutable volTensorField a_star;
    mutable volTensorField a_nutL;
    mutable volTensorField a_aniso;
    mutable volScalarField G0;
    mutable volScalarField G1;
    mutable volScalarField G2;
    mutable volScalarField G3;
    mutable volScalarField G4;
    mutable volScalarField G5;
    mutable volScalarField G6;
    mutable volScalarField G7;
    mutable volScalarField G8;
    mutable volScalarField G9;
    mutable volScalarField nutOptimal;
    mutable std::vector<float> outdata;

    typedef typename BasicTurbulenceModel::alphaField alphaField;
    typedef typename BasicTurbulenceModel::rhoField rhoField;
    typedef typename BasicTurbulenceModel::transportModel transportModel;


    // Constructors

        //- Construct from components
        linearViscousStress
        (
            const word& modelName,
            const alphaField& alpha,
            const rhoField& rho,
            const volVectorField& U,
            const surfaceScalarField& alphaRhoPhi,
            const surfaceScalarField& phi,
            const transportModel& transport,
            const word& propertiesName
        );


    //- Destructor
    virtual ~linearViscousStress()
    {}


    // Member Functions

        //- Re-read model coefficients if they have changed
        virtual bool read() = 0;

        //- Return the effective stress tensor
        virtual tmp<volSymmTensorField> devRhoReff() const;

        //- Return the source term for the momentum equation
        virtual tmp<fvVectorMatrix> divDevRhoReff(volVectorField& U) const;

        //- Return the source term for the momentum equation
        virtual tmp<fvVectorMatrix> divDevRhoReff
        (
            const volScalarField& rho,
            volVectorField& U
        ) const;

        //- Return the source term for the momentum equation using NN surrogate
        virtual tmp<fvVectorMatrix> divDevRhoReff
        (
            volVectorField& U,
            volTensorField& S,
            volTensorField& R
        ) const;

        //- Solve the turbulence equations and correct the turbulence viscosity
        virtual void correct() = 0;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#ifdef NoRepository
    #include "linearViscousStress.C"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
