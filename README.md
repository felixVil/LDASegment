# LDASegment
An official implementation of the LDASegment tracker. The paper can be found here:
https://www.researchgate.net/publication/344468009_LDA-segmentation_based_tracking

The deep neural nets used by the tracker are implemented using Keras over TF 1.x.

## Prerequisites
Please visit https://davischallenge.org/ for the DAVIS dataset and tools. We worked on DAVIS2016.
Please visit https://www.votchallenge.net/ for the dataset and instructions about VOT results. We worked on VOT2016, VOT2018 and VOT2019.
A python interpreter with Keras, scikit-learn, scikit-image and tensorflow(Python 3.6or newer).
MATLAB 2017a or newer.    

## Running on a single video sequence
The parameters, defined in the paper and other important paths that has to be modified to run the code on your computer are concentrated in the KerasTracker/Tracker_Params.py.

After modifying the Tracker_Params.py file you can:

1.Run on a specific VOT video sequence. Please update the sequence names in the KerasTracker/ Trackingmain.py script.  
2.Run on DAVIS2016 video dataset (validation set). Please run the KerasTracker/Trackingmain_VOS.py script.  Please update the path to the validation set (or other set) list of sequences in the "validation_filepath" in the script. 

Both scripts produce result images with the tracking rect overlaid in the path specified in the "track_results_path" parameter.
Debug plots are generated in the path specified in the "debug_images_path" parameter.
The VOT sequences should be located in the path specified by the "base_vot_path" parameter.

## Reproducing VOT results
We use the matlab VOT intergration to call our python implemented tracker functions. The tracker is intergrated in VOT as LDATracker.m file.

To reproduce the VOT results presented in the paper please use the vot-workspace folder and put it on your VOT toolkit folder.
Please update the vot-toolkit path in the begining of each file (addpath...) to fit your system. 
In addition in LDATracker.m update the folowing path variables:

pythonInterpreterPath to point to your python interpreter.

kerasTrackerPath to point to your KerasTracker folder.

## Auxiliary Scripts
To create result/debug videos out of the result images out of your runs on VOT sequences. you can use the CreateTrackingVids.m MATLAB script.




