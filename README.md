**peakonly**
________

*peakonly* is a novel approach written in Python (v3.5) for peaks (aka features) detection in raw LC-MS data. The main idea underlying the approach is the training of two subsequent artificial neural networks to firstly classify ROIs (regions of interest) into three classes (noise, peaks, uncertain peaks) and then to determine boundaries for every peak to integrate its area. Current approach was developed for the high-resolution LC-MS data for the purposes of metabolomics, but can be applied with several adaptations in other fields that utilize data from high-resolution GC- or LC-MS techniques.

- **Article**: currently this work is in preparation for publication
- **Source code**: https://github.com/Arseha/peakonly


Supported formats: 

- .mzML

Operating System Compatibility
------------------------------
peakonly has been tested successfully with:

- Ubuntu 16.04 
- macOS Catalina

Nevertheless, all the dependencies of the project can be installed and run on Windows (especially Windows 10 since it includes a subsystem to run Linux Bash shell). Be sure that your python version is at least 3.5.


Processing your own spectrum
----------------------------
To process your single spectrum you should do a few simple steps:

- download current repository
- install requirements in the following automated way (or you can simply open reqirements.txt file and download listed libraries in any other convenient way): 
```
pip3 install -r requirements.txt
```
- run **run.py** in the following format (**path** - path to your file from current repository, **delta_mz** - a parameters for mz window in ROI detection, **roi_minimum_points** - minimum ROI length in points, **peak_minimum_points** - minimum peak length in points):
```
python3 run.py path delta_mz roi_minimum_points peak_minimum_points
```
The resulted file will be saved as results.csv and will be near **peakonly** folder. The file will contain information about m/z and rt for every peak as well as calculated areas.
(All the commands should be written in your terminal. If you are using Windows 10 you can install the Linux bash shell. Feel free to ask any questions if you have any problems)

To download data used during the training simply run **download_data.sh**, which is inside the folder **data**. There are also **download_mix_example.sh** to dowload a testing LC-MS data. You can also download a testing LC-MS file and data used during training via [googledrive](https://drive.google.com/drive/u/3/folders/1thIvYk72Js7128PCjnwU2OVLMwHc5jpu). If you want to retrain models, train/val/test data should be inside folder **data** (in peakonly repository), don't forget to unzip it (as a result folder **data** shoul include 3 folders: **train**, **val**, **test**). 
To retrain models one can simply run **train_classifier.py** and **train_integrator.py**:
```
python3 train_classifier.py
python3 train_integrator.py
```


Call for Contributions
----------------------

peakonly appreciates help from a wide range of different backgrounds.
Small improvements or fixes are always appreciated.
If you are considering larger contributions, have some questions or an offer for cooperation,
please contact the main contributor (melnikov.arsenty@gmail.com).
