peakonly
________

peakonly is a novel approach for peak detection in raw LC-MS spectrum. The main idea behind it is using neural nets to firstly classify ROI (region of interest) into three classes and then to determine peak boundaries. 

- **Article**: currently this work is not published
- **Source code**: https://github.com/Arseha/peakonly


Supported formats: 

- .mzML

To preprocess your first spectrum you should do a few similiar steps:

- download current repository
- install requirements: 
```
pip3 install -r requirements.txt
```
- run **run.py** in the following format (**path** - path to your file from current repository, **delta_mz** - a parameters for mz window in ROI detection, **required_points** - minimum ROI length in points):
```
python3 run.py path delta_mz required_points
```

To download data used during the training simply run **download_data.sh**, which is inside the folder **data**. There are also **download_mix_example.sh** to dowload a testing spectrum. 
To retrain models one can run **train_classifier.py** and **train_integrator.py**


Call for Contributions
----------------------

peakonly appreciates help from a wide range of different backgrounds.
Small improvements or fixes are always appreciated.
If you are considering larger contributions, have some questions or an offer for cooperation,
please contact the main contributor (melnikov.arsenty@gmail.com).



