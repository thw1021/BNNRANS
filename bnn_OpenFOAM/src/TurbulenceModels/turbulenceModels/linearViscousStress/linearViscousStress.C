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

\*---------------------------------------------------------------------------*/

#include "linearViscousStress.H"
#include "fvc.H"
#include "fvcSmooth.H"
#include "fvm.H"
#include "wallDist.H"
#include <typeinfo>
#include <math.h>
#include <cmath>
#include <algorithm>

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
Foam::linearViscousStress<BasicTurbulenceModel>::linearViscousStress
(
    const word& modelName,
    const alphaField& alpha,
    const rhoField& rho,
    const volVectorField& U,
    const surfaceScalarField& alphaRhoPhi,
    const surfaceScalarField& phi,
    const transportModel& transport,
    const word& propertiesName
)
:
    BasicTurbulenceModel
    (
        modelName,
        alpha,
        rho,
        U,
        alphaRhoPhi,
        phi,
        transport,
        propertiesName
    ),
    ident_
    (
        IOobject
        (
            "ident_",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        this->mesh_,
        tensor(1, 0, 0, 0, 1, 0, 0, 0, 1)
    ),
    b0
    (
        IOobject
        (
            "b0",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        this->mesh_,
        dimensionedTensor("b0", dimless, tensor(0, 0, 0, 0, 0, 0, 0, 0, 0))
    ),
    G0
    (
        IOobject
        (
            "G0",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar("G0", dimless, 0.0)
    ),
    G1
    (
        IOobject
        (
            "G1",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar("G1", dimless, 0.0)
    ),
    G2
    (
        IOobject
        (
            "G2",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar("G2", dimless, 0.0)
    ),
    G3
    (
        IOobject
        (
            "G3",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar("G3", dimless, 0.0)
    ),
    G4
    (
        IOobject
        (
            "G4",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar("G4", dimless, 0.0)
    ),
    G5
    (
        IOobject
        (
            "G5",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar("G5", dimless, 0.0)
    ),
    G6
    (
        IOobject
        (
            "G6",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar("G6", dimless, 0.0)
    ),
    G7
    (
        IOobject
        (
            "G7",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar("G7", dimless, 0.0)
    ),
    G8
    (
        IOobject
        (
            "G8",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar("G8", dimless, 0.0)
    ),
    G9
    (
        IOobject
        (
            "G9",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar("G9", dimless, 0.0)
    ),

    a_dd
    (
        IOobject
        (
            "a_dd",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedTensor("a_dd", dimensionSet(0, 2, -2, 0, 0, 0, 0), tensor(0, 0, 0, 0, 0, 0, 0, 0, 0))
    ),
    a_0
    (
        IOobject
        (
            "a_0",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedTensor("a_0", dimensionSet(0, 2, -2, 0, 0, 0, 0), tensor(0, 0, 0, 0, 0, 0, 0, 0, 0))
    ),
    a_star
    (
        IOobject
        (
            "a_star",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedTensor("a_star", dimensionSet(0, 2, -2, 0, 0, 0, 0), tensor(0, 0, 0, 0, 0, 0, 0, 0, 0))
    ),
    a_nutL
    (
        IOobject
        (
            "a_nutL",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedTensor("a_nutL", dimensionSet(0, 2, -2, 0, 0, 0, 0), tensor(0, 0, 0, 0, 0, 0, 0, 0, 0))
    ),
    a_aniso
    (
        IOobject
        (
            "a_aniso",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedTensor("a_aniso", dimensionSet(0, 2, -2, 0, 0, 0, 0), tensor(0, 0, 0, 0, 0, 0, 0, 0, 0))
    ),
    nutOptimal
    (
        IOobject
        (
            "nutOptimal",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar("nutOptimal", dimensionSet(0, 2, -1, 0, 0, 0, 0), 0.0)
    )
{
    //Read in neural network from file
    rn.readNetFromFile();
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
bool Foam::linearViscousStress<BasicTurbulenceModel>::read()
{
    return BasicTurbulenceModel::read();
}


template<class BasicTurbulenceModel>
Foam::tmp<Foam::volSymmTensorField>
Foam::linearViscousStress<BasicTurbulenceModel>::devRhoReff() const
{
    return volSymmTensorField::New
    (
        IOobject::groupName("devRhoReff", this->alphaRhoPhi_.group()),
        (-(this->alpha_*this->rho_*this->nuEff()))
       *dev(twoSymm(fvc::grad(this->U_)))
    );
}


template<class BasicTurbulenceModel>
Foam::tmp<Foam::fvVectorMatrix>
Foam::linearViscousStress<BasicTurbulenceModel>::divDevRhoReff
(
    volVectorField& U
) const
{
    return
    (
      - fvc::div((this->alpha_*this->rho_*this->nuEff())*dev2(T(fvc::grad(U))))
      - fvm::laplacian(this->alpha_*this->rho_*this->nuEff(), U)
    );
}


template<class BasicTurbulenceModel>
Foam::tmp<Foam::fvVectorMatrix>
Foam::linearViscousStress<BasicTurbulenceModel>::divDevRhoReff
(
    const volScalarField& rho,
    volVectorField& U
) const
{
    return
    (
      - fvc::div((this->alpha_*rho*this->nuEff())*dev2(T(fvc::grad(U))))
      - fvm::laplacian(this->alpha_*rho*this->nuEff(), U)
    );
}

template<class BasicTurbulenceModel>
Foam::tmp<Foam::fvVectorMatrix>
Foam::linearViscousStress<BasicTurbulenceModel>::divDevRhoReff
(
    volVectorField& U,
    volTensorField& S,
    volTensorField& R
) const
{

    size_t sizei = 5;
    size_t sizet = 10;
    dimensionedScalar alpha1 = dimensionedScalar("alpha1", dimless, 24);

    label timeIndex = this->mesh_.time().timeIndex();
    label startTime = this->runTime_.startTimeIndex();

    if(timeIndex - 1 == startTime){
        Info << "[WARNING] Entered Data-driven turbulence predictor..." << endl;
        Info << "Hopefully theres an turb_nn.pt for me to run." << endl;
        Info << "Calculating Flow Field Invariants" << endl;

        //First get the invariant inputs to the NN
        volTensorField s2 (S&S);
        volTensorField r2 (R&R);
        volTensorField s3 (s2&S);
        volTensorField r2s (r2&S);
        volTensorField r2s2 (r2&s2);

        //Get the invariant inputs to the NN
        std::vector<volScalarField*> invar(sizei);
        std::vector<volScalarField*> invar0(sizei);
        invar[0] = new volScalarField(tr(s2));
        invar[1] = new volScalarField(tr(r2));
        invar[2] = new volScalarField(tr(s3));
        invar[3] = new volScalarField(tr(r2s));
        invar[4] = new volScalarField(tr(r2s2));

        invar0[0] = new volScalarField(*invar[0]);
        invar0[1] = new volScalarField(*invar[1]);
        invar0[2] = new volScalarField(*invar[2]);
        invar0[3] = new volScalarField(*invar[3]);
        invar0[4] = new volScalarField(*invar[4]);

        // Normalize the invariants by the sigmoid
        forAll(this->mesh_.C(), cell){
            (*invar0[0])[cell] = Foam::sign((*invar[0])[cell])*(1 - std::exp(-abs((double)(*invar[0])[cell])))/ \
            (1 + std::exp(-abs((double)(*invar[0])[cell])));
            (*invar0[1])[cell] = Foam::sign((*invar[1])[cell])*(1 - std::exp(-abs((double)(*invar[1])[cell])))/ \
            (1 + std::exp(-abs((double)(*invar[1])[cell])));
            (*invar0[2])[cell] = Foam::sign((*invar[2])[cell])*(1 - std::exp(-abs((double)(*invar[2])[cell])))/ \
            (1 + std::exp(-abs((double)(*invar[2])[cell])));
            (*invar0[3])[cell] = Foam::sign((*invar[3])[cell])*(1 - std::exp(-abs((double)(*invar[3])[cell])))/ \
            (1 + std::exp(-abs((double)(*invar[3])[cell])));
            (*invar0[4])[cell] = Foam::sign((*invar[4])[cell])*(1 - std::exp(-abs((double)(*invar[4])[cell])))/ \
            (1 + std::exp(-abs((double)(*invar[4])[cell])));
        }

        //calculate tensor functions
        std::vector<volTensorField*> tensorf(sizet);
        tensorf[0] = new volTensorField(S);
        tensorf[1] = new volTensorField((S&R) - (R&S));
        tensorf[2] = new volTensorField(s2 - 1./3*ident_*(*invar[0]));
        tensorf[3] = new volTensorField(r2 - 1./3*ident_*(*invar[1]));
        tensorf[4] = new volTensorField((R&s2) - (s2&R));
        tensorf[5] = new volTensorField((r2&S) + (S&r2) - 2./3*ident_*tr(S&r2));
        tensorf[6] = new volTensorField(((R&S)&r2) - ((r2&S)&R));
        tensorf[7] = new volTensorField(((S&R)&s2) - ((s2&R)&S));
        tensorf[8] = new volTensorField((r2&s2) + (s2&r2) - 2./3*ident_*tr(s2&r2));
        tensorf[9] = new volTensorField(((R&s2)&r2) - ((r2&s2)&R));

        Info << "Normalize the tensors by L2 norm" << endl;
        forAll(this->mesh_.C(), cell){
            for ( int i = 0; i < 10; i++){
                double l2norm = 0.0;
                Foam::Tensor<double> t_array = (*tensorf[i])[cell];

                l2norm += pow(t_array.xx(), 2);
                l2norm += pow(t_array.xy(), 2);
                l2norm += pow(t_array.xz(), 2);
                l2norm += pow(t_array.yx(), 2);
                l2norm += pow(t_array.yy(), 2);
                l2norm += pow(t_array.yz(), 2);
                l2norm += pow(t_array.zx(), 2);
                l2norm += pow(t_array.zy(), 2);
                l2norm += pow(t_array.zz(), 2);
                l2norm = sqrt(l2norm);

                t_array.replace(0, t_array.xx()/l2norm);
                t_array.replace(1, t_array.xy()/l2norm);
                t_array.replace(2, t_array.xz()/l2norm);
                t_array.replace(3, t_array.yx()/l2norm);
                t_array.replace(4, t_array.yy()/l2norm);
                t_array.replace(5, t_array.yz()/l2norm);
                t_array.replace(6, t_array.zx()/l2norm);
                t_array.replace(7, t_array.zy()/l2norm);
                t_array.replace(8, t_array.zz()/l2norm);

                (*tensorf[i])[cell] = t_array;
            }
        }

        // Now get the data-driven prediction
        Info << "Executing forward pass of the neural network" << endl;
        // Iterate over cells
        forAll(this->mesh_.C(), cell){
            // Define the test inputs
            std::vector<float> inputdata({ (float)(*invar0[0])[cell], (float)(*invar0[1])[cell], (float)(*invar0[2])[cell], \
            (float)(*invar0[3])[cell], (float)(*invar0[4])[cell] });

            // Forward pass of the neural network
            this->outdata = rn.run_forward(inputdata);

            for(int i = 0; i < (int)outdata.size(); i++){
                this->b0[cell] = this->b0[cell] + outdata[i]*(*tensorf[i])[cell];
            }

            this->G0[cell] = outdata[0];
            this->G1[cell] = outdata[1];
            this->G2[cell] = outdata[2];
            this->G3[cell] = outdata[3];
            this->G4[cell] = outdata[4];
            this->G5[cell] = outdata[5];
            this->G6[cell] = outdata[6];
            this->G7[cell] = outdata[7];
            this->G8[cell] = outdata[8];
            this->G9[cell] = outdata[9];
        }
        // Now calculate the Reynolds stress field
        this->a_dd = this->alpha_*this->rho_*this->k()*(this->b0);
        this->a_0 = -1*this->alpha_*this->rho_*(this->nut()*(ident_ & twoSymm(fvc::grad(U))));
        this->a_star = this->a_dd + this->a_0;
        this->nutOptimal = 0.5 * (this->a_star && symm(fvc::grad(U))) / magSqr(symm(fvc::grad(U)));

        this->a_nutL = this->a_star - (alpha1/10) * this->alpha_*this->rho_*(this->nutOptimal*(ident_ & twoSymm(fvc::grad(U))));
        Info << "Job Done" << '\n' << endl;
    }

    this->a_aniso = this->a_nutL - this->alpha_*this->rho_*(this->nut()*(ident_ & twoSymm(fvc::grad(U))));

    return
    (
        - fvm::laplacian(this->alpha_*this->rho_*this->nuEff(), U)
        - fvc::div((this->alpha_*this->rho_*this->nuEff())*dev2(T(fvc::grad(U))))
        + fvc::div(this->a_nutL)
    );
}


template<class BasicTurbulenceModel>
void Foam::linearViscousStress<BasicTurbulenceModel>::correct()
{
    BasicTurbulenceModel::correct();
}


// ************************************************************************* //
