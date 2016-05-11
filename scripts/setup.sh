# setup tensorflow and char-rnn in AWS ec2 ubuntu machine.
wget https://bootstrap.pypa.io/get-pip.py
python get-pip.py
apt-get update
apt-get upgrade
apt-get install git
pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.8.0-cp27-none-linux_x86_64.whl
git clone https://github.com/crazydonkey200/tensorflow-char-rnn.git
