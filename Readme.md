# curAItor

This repository contains the code to run the curAItor tool. This tool has two components:
1. A ResNet-based model that predicts the aesthetic rating class of an input image. The model is trained on the SaatchiArt Artistic Media data set (SAAM) that can be found here: # TODO: insert link

2. A tool that extracts a wide array of features from images, such as colorfulness, contrast, and an array of metrics from the Pyradiomics Python package.

Check out the paper for more information: # TODO: insert link

General instructions:  
-Clone this repo  
-Install the packages in requirements.txt   
-Check out the two tutorial Jupyter notebooks to see how the tool can be used

Note:  
The Pyradiomics package that is used in this repo has a bug in it. This bug breaks the feature extraction. To fix it, manually apply the change mentioned in this commit:   
https://github.com/AIM-Harvard/pyradiomics/commit/3064793b0dd7b83de9971f09beb597449f83107c