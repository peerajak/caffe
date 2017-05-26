#!/bin/bash
model_mat=$(ls ExpParams*.mat)
pwd_model_mat=$PWD\/$model_mat
sed -i "s|model_mat=.*|model_mat='$pwd_model_mat'|" python_*train*.py
sed -i "s|model_mat=.*|model_mat='$pwd_model_mat'|" python_create_lmdb.py
sed -i "s|model_mat=.*|model_mat='$pwd_model_mat'|"  /working3/peerajak/ChulaQE/Semister9/1_caffe2/python/LossAugmentedInfLoss.py
