
conda create -n groove python=3.8
conda activate groove
CONDA_OVERRIDE_CUDA=12.3 conda install "jaxlib=0.4.12=*cuda*" jax=0.4.12 cuda-nvcc -c conda-forge -c nvidia

pip3 install -r setup/requirements-base.txt
pip3 install urllib3==1.26.16  chex==0.1.6 orbax-checkpoint==0.2.6 scipy==1.9.3

CONDA_OVERRIDE_CUDA=12.3 conda install "jaxlib=0.4.12=*cuda*" jax=0.4.12 cuda-nvcc -c conda-forge -c nvidia
