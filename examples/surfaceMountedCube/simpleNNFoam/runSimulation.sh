#!/bin/bash
prefix="turb_nn_"
for appDir in {0..19}
do
(
    [ -d $prefix$appDir ] && cd $prefix$appDir || exit

    turb_nn=`find . -name "turb_nn.pt"`
    if [ -z "$turb_nn" ]; then
        echo "There is no nn model in current working directory $(pwd)" && exit
    fi

    . $WM_PROJECT_DIR/bin/tools/RunFunctions

    # Get application directory
    application=`getApplication`

    runApplication checkMesh
    runApplication decomposePar
    runParallel $application

    runApplication reconstructPar -latestTime

    runApplication foamLog log.simpleNNFoam

    touch case.foam
)
done




