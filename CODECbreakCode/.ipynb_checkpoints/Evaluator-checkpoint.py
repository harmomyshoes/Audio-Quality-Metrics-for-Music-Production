##The function that mearsure the file and its codec counterpart, let us say we had "a.wav" we compre "a.wav" and "a.64kbps.wav"
import re
import os
#import wave
def MeasurePEAQOutputsVsRefencefile(FilePath,bitrate,RefFile):
    '''The function is used to measure the PEAQ score of the file and its codec counterpart.
    The function takes three arguments, the first argument is the file path of the file to be measured, 
    the second argument is the bitrate of the codec, and the third argument is the reference file path.
    The function returns the PEAQ score of the file and its codec counterpart.
    
    The function uses the lame codec to encode the file and the peaq to measure the PEAQ score.
    So, the FilePath is the file in the WAV format, the bitrate is the bitrate of the codec, 
    and the RefFile is the reference file in the WAV format as well.'''

    command_out = os.popen("sh /home/codecrack/Jnotebook/CODECbreakCode/Audio_Lame_Peaq_VSRef.sh -a %s -b %s -r %s" %(FilePath,bitrate,RefFile)).read()
    match = re.search(r'Objective Difference Grade: (-?\d+\.\d+)', command_out)
    if match:
        Objective_sccore = match.group(1)
        #print("Value:",Objective_sccore)
        return Objective_sccore
    else:
        print("Notihing out, possible something wrong in the lame or peaq")
        return 0.0

def MeasurePEAQOutputwithoutCodec(RefFile, ComFile, Version='basic'):
    '''The function is used to measure the PEAQ score of the file and its codec counterpart.
    The function takes two arguments, 
    the first argument is the reference file path and the second argument is the file path of the file to be measured.'''
    command_out = os.popen("peaq --%s  %s  %s" % (Version,RefFile, ComFile)).read()
    match = re.search(r'Objective Difference Grade: (-?\d+\.\d+)', command_out)
    if match:
        Objective_sccore = match.group(1)
        #print("Value:",Objective_sccore)
        return Objective_sccore
    else:
        print("Notihing out, possible something wrong in the lame or peaq")
        return 0.0
    
def Mp3LameLossyCompress(FilePath,bitrate):
    '''The function is used to compress the file using the lame codec.'''
    command_out = os.popen("sh /home/codecrack/Jnotebook/CODECbreakCode/Audio_LameCompress.sh -a %s -b %s " %(FilePath,bitrate)).read()
    match = re.search(r"outputMp3toWavfilepath=\s*(.+?)\s+by FFMPEG", command_out)

    if match:
        file_path = match.group(1)  # Capture the file path
        return file_path
    else:
        print("File path not found in the output.") 
        return "File path not found in the output."


import numpy as np
import clarity
import clarity.evaluator.haaqi as haaqi
import librosa
from clarity.utils.audiogram import Audiogram
class MeasureHAAQIOutput:
    '''The HAAQI becasue its with some configurations, it runs as a class to be able to set the configurations'''
    def __init__(self, ref_audio_path, levels_1 = np.array([0, 0, 0, 0, 0, 0, 0, 0])):
        '''The function is used to initialize the HAAQI class with the reference audio path and hearing loss levels(No heraring loss).'''
        self._audiogram_NH_ = Audiogram(levels=levels_1)
        self._reference_audio_data_, self._srate_ = librosa.load(ref_audio_path, sr=None)
        
    def set_reference_audio_data(self, ref_audio_path):
        '''The function is used to set the reference audio data.'''
        self._reference_audio_data_, self._srate_ = librosa.load(ref_audio_path, sr=None)

    def MeasureHAQQIOutput(self, com_audio_path):
        '''The function is used to measure the HAAQI score of the reference file and its codec counterpart.'''
        com_audio_data, _ = librosa.load(com_audio_path, sr=None)
        #return round(haaqi.compute_haaqi(com_audio_data, self._reference_audio_data_, self._srate_, self._srate_, self._audiogram_NH_),2)
        return haaqi.compute_haaqi(com_audio_data, self._reference_audio_data_, self._srate_, self._srate_, self._audiogram_NH_)
