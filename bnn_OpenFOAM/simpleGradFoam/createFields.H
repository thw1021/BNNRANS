Info<< "Reading field p\n" << endl;
volScalarField p
(
    IOobject
    (
        "p",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

Info<< "Reading field U\n" << endl;
volVectorField U
(
    IOobject
    (
        "U",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

Info<< "Reading field Rotational\n" << endl;
volTensorField R
(
    IOobject
    (
        "R",
        runTime.timeName(),
        mesh,
        IOobject::READ_IF_PRESENT,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedTensor("R", dimensionSet(0, 0, 0, 0, 0, 0, 0), tensor( 0, 0, 0, 0, 0, 0, 0, 0, 0))
);

Info<< "Reading field Symetric\n" << endl;
volTensorField S
(
    IOobject
    (
        "S",
        runTime.timeName(),
        mesh,
        IOobject::READ_IF_PRESENT,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedTensor("S", dimensionSet(0, 0, 0, 0, 0, 0, 0), tensor( 0, 0, 0, 0, 0, 0, 0, 0, 0))
);

Info<< "Reading field Velocity Gradients\n" << endl;
volTensorField UGrad
(
    IOobject
    (
        "UGrad",
        runTime.timeName(),
        mesh,
        IOobject::READ_IF_PRESENT,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedTensor("UGrad", dimensionSet(0, 0, -1, 0, 0, 0, 0), tensor( 0, 0, 0, 0, 0, 0, 0, 0, 0))
);

Info<< "Reading field pressure Gradients\n" << endl;
volVectorField pGrad
(
    IOobject
    (
        "pGrad",
        runTime.timeName(),
        mesh,
        IOobject::READ_IF_PRESENT,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedVector("pGrad", dimensionSet(0, 1, -2, 0, 0, 0, 0), vector( 0, 0, 0))
);

Info<< "Reading field TKE Gradients\n" << endl;
volVectorField kGrad
(
    IOobject
    (
        "kGrad",
        runTime.timeName(),
        mesh,
        IOobject::READ_IF_PRESENT,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedVector("kGrad", dimensionSet(0, 1, -2, 0, 0, 0, 0), vector(0, 0, 0))
);

volTensorField a_0
(
    IOobject
    (
        "a_0",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedTensor("a_0", dimensionSet(0, 2, -2, 0, 0, 0, 0), tensor(0, 0, 0, 0, 0, 0, 0, 0, 0))
);

volTensorField ident
(
    IOobject
    (
        "ident",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    tensor(1, 0, 0, 0, 1, 0, 0, 0, 1)
);

#include "createPhi.H"


label pRefCell = 0;
scalar pRefValue = 0.0;
setRefCell(p, simple.dict(), pRefCell, pRefValue);
mesh.setFluxRequired(p.name());


singlePhaseTransportModel laminarTransport(U, phi);

autoPtr<incompressible::turbulenceModel> turbulence
(
    incompressible::turbulenceModel::New(U, phi, laminarTransport)
);

#include "createMRF.H"
#include "createFvOptions.H"
