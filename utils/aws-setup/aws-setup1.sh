# h/t https://gist.github.com/8enmann/reinstall.sh
# Updated to use correct drivers for p2.xlarge instance
# Download installers
mkdir ~/nvidia
cd ~/nvidia
wget https://developer.nvidia.com/compute/cuda/9.2/Prod2/local_installers/cuda_9.2.148_396.37_linux
wget http://us.download.nvidia.com/tesla/396.44/NVIDIA-Linux-x86_64-396.44.run
sudo chmod +x NVIDIA-Linux-x86_64-396.44.run
sudo chmod +x cuda_9.2.148.1_linux
./cuda_8.0.61_375.26_linux-run -extract=~/Downloads/nvidia/
# Uninstall old stuff
sudo apt-get --purge remove nvidia-*
sudo nvidia-uninstall

# Reboot
sudo reboot
