/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2016 OpenFOAM Foundation
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

#include "parabolicVelocityFvPatchVectorField.H"
#include "addToRunTimeSelectionTable.H"
#include "fvPatchFieldMapper.H"
#include "volFields.H"
#include "surfaceFields.H"

// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::parabolicVelocityFvPatchVectorField::
parabolicVelocityFvPatchVectorField
(
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF
)
:
    fixedValueFvPatchVectorField(p, iF),
    maxValue_(0),
    n_(1, 0, 0),
    y_(0, 1, 0)
{
}


Foam::parabolicVelocityFvPatchVectorField::
parabolicVelocityFvPatchVectorField
(
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF,
    const dictionary& dict
)
:
    fixedValueFvPatchVectorField(p, iF),
    maxValue_(readScalar(dict.lookup("maxValue"))),
    n_(dict.lookup("n")),
    y_(dict.lookup("y"))
{
    Info << "Using the parabolicVelocity boundary condition" << endl;
    if (mag(n_) < SMALL || mag(y_) < SMALL)
    {
        FatalErrorIn("parabolicVelocityFvPatchVectorField(dict)") 
        << "n or y given with zeor size not correct" 
        << abort(FatalError);
    }

    n_ /= mag(n_);
    y_ /= mag(y_);

    fixedValueFvPatchVectorField::evaluate();
}


Foam::parabolicVelocityFvPatchVectorField::
parabolicVelocityFvPatchVectorField
(
    const parabolicVelocityFvPatchVectorField& ptf,
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF,
    const fvPatchFieldMapper& mapper
)
:
    fixedValueFvPatchVectorField(ptf, p, iF, mapper),
    maxValue_(ptf.maxValue_),
    n_(ptf.n_),
    y_(ptf.y_)
{}


Foam::parabolicVelocityFvPatchVectorField::
parabolicVelocityFvPatchVectorField
(
    const parabolicVelocityFvPatchVectorField& ptf
)
:
    fixedValueFvPatchVectorField(ptf),
    maxValue_(ptf.maxValue_),
    n_(ptf.n_),
    y_(ptf.y_)
{}


Foam::parabolicVelocityFvPatchVectorField::
parabolicVelocityFvPatchVectorField
(
    const parabolicVelocityFvPatchVectorField& ptf,
    const DimensionedField<vector, volMesh>& iF
)
:
    fixedValueFvPatchVectorField(ptf, iF),
    maxValue_(ptf.maxValue_),
    n_(ptf.n_),
    y_(ptf.y_)
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

/* void Foam::parabolicVelocityFvPatchVectorField::autoMap
(
    const fvPatchFieldMapper& m
)
{
    fixedValueFvPatchVectorField::autoMap(m);
    fieldData_.autoMap(m);
}


void Foam::parabolicVelocityFvPatchVectorField::rmap
(
    const fvPatchVectorField& ptf,
    const labelList& addr
)
{
    fixedValueFvPatchVectorField::rmap(ptf, addr);

    const parabolicVelocityFvPatchVectorField& tiptf =
        refCast<const parabolicVelocityFvPatchVectorField>(ptf);

    fieldData_.rmap(tiptf.fieldData_, addr);
} */


void Foam::parabolicVelocityFvPatchVectorField::updateCoeffs()
{
    if (updated())
    {
        return;
    }
    
    // Get range and orientation
    boundBox bb(patch().patch().localPoints(), true);

    vector ctr = 0.5 * (bb.max() + bb.min());

    const vectorField& c = patch().Cf();

    // Calculate local 1-D coordinate for the parabolic profile
    // The velocity profile is specified as U_m * (1.0 - 4 * (y - H/2)^2 / H^2)
    // The average velocity is 2/3 * U_m
    scalarField coord = 2 * ((c - ctr) & y_) / ((bb.max() - bb.min()) & y_);

    vectorField::operator = (n_ * maxValue_ * (1.0 - sqr(coord)));


    fixedValueFvPatchVectorField::updateCoeffs();
}


void Foam::parabolicVelocityFvPatchVectorField::write
(
    Ostream& os
) const
{
    fvPatchVectorField::write(os);
    os.writeKeyword("maxValue") << maxValue_ << token::END_STATEMENT << nl;
    os.writeKeyword("n") << n_ << token::END_STATEMENT << nl;
    os.writeKeyword("y") << y_ << token::END_STATEMENT << nl;
    writeEntry(os, "value", *this);
}


// * * * * * * * * * * * * * * Build Macro Function  * * * * * * * * * * * * //

namespace Foam
{
    makePatchTypeField
    (
        fvPatchVectorField,
        parabolicVelocityFvPatchVectorField
    );
}

// ************************************************************************* //
