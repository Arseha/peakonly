**peakonly** 
________
THIS BRANCH IS IN A DEVELOPMENT. ALL THE FEATURES (THE OLD ONE AND THE NEW ONE) MAY WORK IMPROPERLY!

Main differences with the **master** branch:
- ROI detection is speeded up.
- Added functionality to process a batch of the spectra (with some limitations for now; contact the main contributor if you want to know more).

To process a batch of the spectra one can use **run_batch.py** in the way similiar with **run.py**, which is described in the **master** branch. Instead of the path to the file (with the name), in this case, you just need to specify the path to the dicrectory with files (the source directory) as the **path** parameter (the program will find all the *.mzML files in the source dicrectory and in all internal dicrectories). The result table will be saved in the source directory.


*peakonly* is a novel approach written in Python (v3.5) for peaks (aka features) detection in raw LC-MS data. The main idea underlying the approach is the training of two subsequent artificial neural networks to firstly classify ROIs (regions of interest) into three classes (noise, peaks, uncertain peaks) and then to determine boundaries for every peak to integrate its area. Current approach was developed for the high-resolution LC-MS data for the purposes of metabolomics, but can be applied with several adaptations in other fields that utilize data from high-resolution GC- or LC-MS techniques.

- **Article**: currently this work is in preparation for publication
- **Source code**: https://github.com/Arseha/peakonly

Call for Contributions
----------------------

peakonly appreciates help from a wide range of different backgrounds.
Small improvements or fixes are always appreciated.
If you are considering larger contributions, have some questions or an offer for cooperation,
please contact the main contributor (melnikov.arsenty@gmail.com).
