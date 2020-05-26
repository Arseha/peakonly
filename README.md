**peakonly**
________

*peakonly* is a novel approach written in Python (v3.6) for peaks (aka features) detection in raw LC-MS data. The main idea underlying the approach is the training of two subsequent artificial neural networks to firstly classify ROIs (regions of interest) into ~~three~~ two classes (noise, peaks, ~~uncertain peaks~~) and then to determine boundaries for every peak to integrate its area. Current approach was developed for the high-resolution LC-MS data for the purposes of metabolomics, but can be applied with several adaptations in other fields that utilize data from high-resolution GC- or LC-MS techniques.

- **Article**: [Deep learning for the precise peak detection in high-resolution LC-MS data, *Analytical Chemistry*.](http://dx.doi.org/10.1021/acs.analchem.9b04811)
- **Releases**: https://github.com/arseha/peakonly/releases/
- **Instruction:** [detailed instruction for *peakonly* v.0.2.0-beta](https://drive.google.com/file/d/1pzJBpUINsdjWKQy0SkxXae79e5j0r3sG/view)


Supported formats: 

- .mzML

Operating System Compatibility
------------------------------
peakonly has been tested successfully with:

- Ubuntu 16.04 
- macOS Catalina
- Windows 10 
- Windows 7

For Windows7/10 commands should be entered through Windows PowerShell. [Detailed instruction is available](https://drive.google.com/file/d/1pzJBpUINsdjWKQy0SkxXae79e5j0r3sG/view). Be sure that your python version is at least 3.6.


Installing and running the application
----------------------------
To install and run *peakonly* you should do a few simple steps:

- download [the latest release of *peakonly*](https://github.com/Arseha/peakonly/releases)
- install requirements in the following automated way (or you can simply open reqirements.txt file and download listed libraries in any other convenient way): 
```
pip3 install -r requirements.txt
```
- run **peakonly.py**:
```
python3 peakonly.py
```

The more detailed instruction on how to install and run the application as well as a thorough manual on how to use it is available via [the link.](https://drive.google.com/file/d/1pzJBpUINsdjWKQy0SkxXae79e5j0r3sG/view)


Call for Contributions
----------------------

peakonly appreciates help from a wide range of different backgrounds.
Small improvements or fixes are always appreciated.
If you are considering larger contributions, have some questions or an offer for cooperation,
please contact the main contributor (melnikov.arsenty@gmail.com).
