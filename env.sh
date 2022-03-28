conda create -n fcaf3d python==3.6.8
conda activate fcaf3d 
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install mmcv-full==1.3.17 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
pip install mmdet==2.19.0 
pip install mmsegmentation==0.20.0  
pip install -v -e .  # or "python setup.py develop"
pip install torch ninja
pip install -U MinkowskiEngine --install-option="--blas=openblas" -v --no-deps
cd Rotated_IoU/cuda_op
python setup.py install
cp -r Rotated_IoU/cuda_op mmdet3d/ops/rotated_iou