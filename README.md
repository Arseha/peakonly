**peakonly**
________

*peakonly* is a novel approach for peaks (features) detection in raw LC-MS data written in Python version 3.5. The main idea underlying the approach is the training of two subsequent artificial neural networks to firstly classify ROIs (regions of interest) into three classes (noise, peaks, uncertain peaks) and then to determine boundaries for every peak to integrate its area. Current approach was developed for the high-resolution LC-MS data for the purposes of metabolomics, but can be applied with several adaptations in other fields that utilize data from GC- or LC-MS techniques.

- **Article**: currently this work is in preparation for publication
- **Source code**: https://github.com/Arseha/peakonly


Supported formats: 

- .mzML

To preprocess your first spectrum you should do a few simple steps:

- download current repository
- install requirements: 
```
pip3 install -r requirements.txt
```
- run **run.py** in the following format (**path** - path to your file from current repository, **delta_mz** - a parameters for mz window in ROI detection, **required_points** - minimum ROI length in points):
```
python3 run.py path delta_mz required_points
```
(All the commands should be written in your terminal. If you are using Windows 10 you can install the Linux bash shell. Feel free to ask any questions if you have any problems)

To download data used during the training simply run **download_data.sh**, which is inside the folder **data**. There are also **download_mix_example.sh** to dowload a testing LC-MS data. 
To retrain models one can run **train_classifier.py** and **train_integrator.py**


Call for Contributions
----------------------

peakonly appreciates help from a wide range of different backgrounds.
Small improvements or fixes are always appreciated.
If you are considering larger contributions, have some questions or an offer for cooperation,
please contact the main contributor (melnikov.arsenty@gmail.com).



