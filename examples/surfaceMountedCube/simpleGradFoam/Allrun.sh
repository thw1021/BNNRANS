#!/bin/sh
cd ${0%/*} || exit 1    # Run from this directory
cp ./turb_nn/turb_nn_0.pt turb_nn.pt

turb_nn=`find . -name "turb_nn.pt"`
if [ -z "$turb_nn" ]; then
    echo "There is no nn model in current working directory $(pwd)" && exit
fi
# Source tutorial run functions
. $WM_PROJECT_DIR/bin/tools/RunFunctions

application=`getApplication`

runApplication decomposePar

runParallel $application

runApplication reconstructPar -latestTime

runApplication foamLog log.simpleGradFoam


#------------------------------------------------------------------------------
