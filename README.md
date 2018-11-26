# Smooth pursuit detection tool (sp_tool) for eye tracking recordings.

This is a python implementation of the SP-DBSCAN algorithm described in [1]. The project page is located here: http://michaeldorr.de/smoothpursuit/. Please refer to it for the related data set (http://michaeldorr.de/smoothpursuit/GazeCom.zip) and other algorithm implementations and performance statistics.

If you use the code, please cite our paper:

    @inproceedings{agtzidis2016smooth,
      title={Smooth pursuit detection based on multiple observers},
      author={Agtzidis, Ioannis and Startsev, Mikhail and Dorr, Michael},
      booktitle={Proceedings of the Ninth Biennial ACM Symposium on Eye Tracking Research \& Applications},
      pages={303--306},
      year={2016},
      organization={ACM}
    }

For the related data set, please temporarily cite its web page:

    @misc{sp-detection-site, 
      author = {Startsev, Mikhail and Agtzidis, Ioannis and Dorr, Michael}, 
      title = {Smooth Pursuit}, 
      howpublished = {\url{http://michaeldorr.de/smoothpursuit/}}, 
      year = {2016} 
    }

This is an updated repository, with more evaluation metrics included, in particular. The original version's archive can be downloaded from http://michaeldorr.de/smoothpursuit/.

# I. INSTALLATION.

To use the package's python interface system-wide (and not just from its source directory), please run either
    
    $> python setup.py install --user
    
or, to install it for all users:

    $> sudo python setup.py install

If the dependencies are not installed automatically, you can manually install them with pip
(installed by `sudo apt install python-pip`):
    
    $> pip install --user liac-arff numpy
    
or, alternatively:
    
    $> sudo pip install liac-arff numpy


### NB: The underlying ARFF library is NOT called "arff", but "liac-arff"! If you get an error from within arff_helper.py, try uninstalling the "arff" library and/or re-installing the "liac-arff".


# II. USAGE

The main interface of this distribution is the file run_detection.py.
For console interface, see `python run_detection.py --help`. Default config file is provided with
the distibution, so to run on GazeCom data set you can just run 

    $> python run_detection.py --config default_parameters.conf.json 

IMPORTANT: the config file contains the path to GazeCom [3] gaze_arff folder; it assumes
that you have downloaded the [GazeCom.zip](http://michaeldorr.de/smoothpursuit/GazeCom.zip) and
unzipped it in the parent directory into the folder GazeCom. If that is not the case, you can 
either change the config file, or pass an additional --input-folder /path/to/your/GazeCom/folder
argument.

For programmatic interface, see documentation and comments in run_detection.py.
For more examples of data conversion and programmatic interface usage, see examples/ as well.

If you desire to use this method on another data set, you need to 

  1. Ensure the right format (ARFF? see examples/convert_*.py) and placement of the data set files:
      The top-level folder has to contain the folders, which correspond to the stimuli 
      (i.e. each video stimulus gets its own folder here). Each of those folders has to contain
      the respective eye tracking recordings for all the available observers. 
      The structure is, therefore, something like this:

    data_set_folder/
        video_clip_01/
            participant01_clip01.arff
            participant02_clip01.arff
            participant03_clip01.arff
            participant05_clip01.arff  
        video_clip_02/
            participant02_clip02.arff
            participant03_clip02.arff
            participant04_clip02.arff
        video_clip_03/
            participant01_clip03.arff
            participant02_clip03.arff
            participant03_clip03.arff
            participant04_clip03.arff
            participant05_clip03.arff
  2. Adjust the config file. We recommend copying the provided default.conf.json file inside the data set folder to avoid confusion.
      
        * There, you need to set the path (absolute or relative) to the data_set_folder/ from above.

        * You would probably want to specify the output_folder parameter as well, to get the resulting labelled ARFF files stored to a specific location. If it is not specified, a temporary folder will be automatically created.

        * The last thing that needs to be changed is the min_pts parameter (close to the end of the config file). This integer value should be changed proportionally to the number of observers and the sampling frequency of the recordings. The default value (provided in the config file) is optimized for a set with an average of 46.9 observers per clip, recorded at 250 Hz. 

          If your data set has an average of N observers per stimulus, and was recorded at the frequency of F Hz,
          you should set min_pts to:

              (default value, e.g. 160) * (N / 46.9) * (F / 250).

          You should round the result accordingly.

          For example, at 12.9 observers having seen each clip, and 500 Hz recordings, 
          the new min_pts should be equal to 160 * 12.9/46.9 * 500/250 ~= 88 samples.

          After this, you should be able to simply run 
  
              $> ./run_detection.py --config /path/to/the/config/file/your_config.json
      
          in order to get the labels on your data. You might want to add --verbose to get runtime updates.

# III. CONTENTS

The rest of the distribution mainly consists of the following files:
  - arff_helper.py - a set of tools to handle the ARFF file format with certain additions to support various recording attributes, such as the physical dimensions of the monitor during recording.
  - saccade_detector.py, blink_detector.py, fixation_detector.py and sp_detector.py, which perform the actual eye        movement classification. sp_detection.py contains the actual SP-DBSCAN implementation. It supports one extra mode of operation compared to regular DBSCAN (in class DBSCANWithMinObservers): setting a threshold for the number of different observers withing the neighbourhood (instead of specifying a threshold for the number of different points).
  - evaluate.py - comparing the output of the algorithm (.arff files with an EYE_MOVEMENT_TYPE field in each) with the hand-labelled ground truth (in the format of the output of the tool in [2])
  - recording_processor.py - a set of methods to load the recording data and its preprocessing.
  - examples/ - a folder with various use-cases for this tool, including converting data produced by EyeLink or SMI eye trackers to ARFF format and valuating on the GazeCom hand-labelled ground truth [4] labels.

# IV. LICENSE

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
license - GPL.


# V. CONTACT

For feedback and collaboration you can contact Mikhail Startsev via mikhail.startsev@tum.de, or any of the authors of [1] at < firstname.lastname > @tum.de.

# VI. REFERENCES

  [1] http://dl.acm.org/citation.cfm?id=2857521 : "Smooth pursuit detection based on multiple observers", Ioannis Agtzidis, Mikhail Startsev, Michael Dorr. ETRA'16
  
  [2] http://ieeexplore.ieee.org/abstract/document/7851169/ : "In the pursuit of (ground) truth: a hand-labelling tool for eye movements recorded during dynamic scene viewing", Ioannis Agtzidis, Mikhail Startsev, Michael Dorr. ETVIS'16
  
  [3] http://jov.arvojournals.org/article.aspx?articleid=2121333 : "Variability of eye movements when viewing dynamic natural scenes", Michael Dorr, Thomas Martinetz, Karl R. Gegenfurtner, Erhardt Barth. JOV (2010)
  
  [4] http://www.michaeldorr.de/smoothpursuit : Our eye movement classification project with an emphasis on smooth pursuit
