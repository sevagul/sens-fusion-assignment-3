# Assignment 3 for Sensor Fusion subject in ELTE, Budapest by Vsevolod Hulchuk<br/>
The pointclouds are obtained from the previous assignments: [Assignment1](https://github.com/sevagul/sens-fusion-2022) on Stereo Vision and [Assignment2](https://github.com/sevagul/sens_fusion_assignment2) on Filtering and Upsamling. <br/>
See analysis of ICP parameters in [Analysis.pdf](https://github.com/sevagul/sens-fusion-assignment-3/blob/main/Analysis.pdf) <br/>
## Usage
Build using standad procedure:
```
cd build
cmake ..
make
cd ..
```
To see help message on script usage:
```
./build/TrICP --help
```
Be sure to download data before running:
```
cd data
./load_all_data.sh
cd ..
```
## Results
Initial allignment of the pointclouds: <br/>
![init](http://server.seva-hul.com/media/IFRoS/SensFusion/init.jpg)<br/>
Naive ICP:<br/>
![icp](http://server.seva-hul.com/media/IFRoS/SensFusion/naive.jpg)<br/>
Trimmed ICP:<br/>
![icp](http://server.seva-hul.com/media/IFRoS/SensFusion/trimmed.jpg)<br/>


