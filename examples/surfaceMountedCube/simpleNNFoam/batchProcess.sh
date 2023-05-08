#!/bin/bash
prefix="turb_nn_"
for appDir in {0..19}
do
(
    mkdir $prefix$appDir
    [ -d $prefix$appDir ] && cd $prefix$appDir || exit
    # rm -rf *
    cp -r ../baseSimpleNNFoam/0/ 0/
    cp -r ../baseSimpleNNFoam/constant/ .
    cp -r ../baseSimpleNNFoam/system/ .
    cp ../turb_nn/$prefix$appDir.pt turb_nn.pt
)
done




