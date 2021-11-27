#!/bin/bash

#this downloads the zip files that contains the data
curl https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Training_Data.zip --output train_input.zip
curl https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Training_GroundTruth.zip --output train_label.zip
curl https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Test_Data.zip --output test_input.zip
curl https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Test_GroundTruth.zip --output test_label.zip

# this unzips the zip files - you will get a directory named "data" containing the data
unzip train_input.zip
unzip train_label.zip
unzip test_input.zip
unzip test_label.zip

# this cleans up the zip file, as we will no longer use it
rm train_input.zip
rm train_label.zip
rm test_input.zip
rm test_label.zip

echo downloaded data
