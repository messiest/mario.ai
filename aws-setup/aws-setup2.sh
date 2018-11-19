# h/t https://gist.github.com/8enmann/reinstall.sh
# Updated to use correct drivers for p2.xlarge instance
sudo apt-get install libcurl3-dev swig
sudo apt-get install -y gcc g++ gfortran  git linux-image-generic linux-headers-generic linux-source linux-image-extra-virtual libopenblas-dev
sudo apt-get install -y cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig

sudo service lightdm stop
sudo ~/nvidia/NVIDIA-Linux-x86_64-396.44.run --no-opengl-files
sudo ~/nvidia/cuda_9.2.148_396.37_linux.run --no-opengl-libs
# Verify installation
nvidia-smi
cat /proc/driver/nvidia/version
