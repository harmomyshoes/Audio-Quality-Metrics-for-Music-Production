import numpy as np
import pandas as pd
from SignalProcessor import SignalProcessorClass
import NoiseEvalUtil as NEUtil
import argparse
import os

base_dir = "/home/codecrack/Jnotebook/48k_16bit/Reggea"

# List all first-layer folders
#first_layer_dirs = [
#    os.path.join(base_dir, name) + '/' for name in os.listdir(base_dir)
#    if os.path.isdir(os.path.join(base_dir, name))
#]

# Print results
#for Mixing_Path_48k_16bit in first_layer_dirs:
Noise_Generator_MP3_48k_16bit = SignalProcessorClass(filename="mixture.wav", foldpath=base_dir, TrackType = NEUtil.MixingType.File,bitdepth="PCM_16")
Referece_File_48k_16bit = Noise_Generator_MP3_48k_16bit.TestNoisedOnlyFile([0,0,0,0],"Refer_T.wav")
Mp3_Referece_File_48k_16bit = Noise_Generator_MP3_48k_16bit.GeneratingMP3RefFile(Referece_File_48k_16bit, 64)

PEAQ_score_mp3 = Noise_Generator_MP3_48k_16bit.MeasurePEAQOutputsVsRef(Referece_File_48k_16bit,64,Mp3_Referece_File_48k_16bit,'basic') 
PEAQ_score_mp3