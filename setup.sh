sudo apt-get install xvfb libav-tools ffmpeg

conda install pytorch-nightly cuda92 -c pytorch
conda install torchvision -c pytorch --no-deps

pip install --upgrade pip

pip install gym gym-super-mario-bros xvfbwrapper emoji setuptools==40.5.0
