# traffic_rl
## Setup
install sumo sumo-tools sumo-doc

echo 'export SUMO_HOME="/usr/share/sumo"' >> ~/.bashrc

source ~/.bashrc

## Install
python3 setup.py install
## Run
python3 single-intersection.py -route roadnets/single-intersection.rou.xml

by default runs for 20000sec, to control and debug reduce the second count

python3 single-intersection.py -route roadnets/single-intersection.rou.xml -s 100
