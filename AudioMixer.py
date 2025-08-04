import os
import numpy as np
import shutil
import NoiseEvalUtil as NEUtil
import NoiseEvalEffect as NoiseEvalEffect
from audiomentations import Gain,Normalize,LoudnessNormalization,AddGaussianSNR,ClippingDistortion,Mp3Compression

dtype = np.int16 
class AudioMixerClass:
###The fold should including the four type of stems files 'vocals''drum''bass''other'
###isMONO will decide whether the file be mixed to. by default setting, the input files will in Stereo.
    def __init__(self, originalData, n_channels,sample_width, framerate, n_frames, duration,startingTime=0):
        """###The fold should including the four type of stems files 'vocals''drum''bass''other'"""
        self.Foldpath = os.getcwd()
        self.BitDepth = sample_width
        self.Channels = n_channels
        self.OutputMixingFold = self.Foldpath +'/Mixing_Result/'
        isExist = os.path.exists(self.OutputMixingFold)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(self.OutputMixingFold)
        self.SampleRate = framerate
        self.Duration = duration
        self.StartingTime = startingTime
        audio_array = np.frombuffer(originalData, dtype=dtype)
        max_val = np.iinfo(dtype).max
        audio_array = audio_array / max_val
        self.originalData = audio_array
        self.InitalData= self.LoadSingleFile(audio_array, self.Channels, self.StartingTime)

    def LoadSingleFile(self, originalData, n_channels, startingTime=0):
        if n_channels > 1:
            audio_array = originalData[::2]  # Take just one channel
        # Normalize amplitude to [-1, 1]

        # Handle audio longer than 8 seconds
        max_duration = 8
        if self.Duration > max_duration:
            audio_array = audio_array[startingTime * self.SampleRate +
                                       startingTime:int((max_duration+startingTime) * self.SampleRate)]

            self.Duration = max_duration
            return audio_array
        else:
            # print(f"Vocal duration orginal is {mixing_data_duration} seconds, now is the {librosa.get_duration(y=mixing_data,sr=mixing_sr)}, the audio keep the Stereo")
            None
        return audio_array

    def GeneratingMP3Ref(self, audioData, bitrate=64): 
        transform = Mp3Compression(
            min_bitrate=bitrate,
            max_bitrate=bitrate,
            backend="fast-mp3-augment",
            preserve_delay=False,
            p=1.0
        )

        refMp3Data = transform(audioData, sample_rate=self.SampleRate)
        return refMp3Data
    
    def TestNoisedOnlyFile(self,file_Manipul_list):
        HumNoiseValue = file_Manipul_list[0]
        GaussianNoiseValue = file_Manipul_list[1]
        ClipPercentValue = file_Manipul_list[2]
        DropOutSampleNum = file_Manipul_list[3]

        #print(f"In the initialize, there is {NEUtil.count_zeros(self.InitalData)} in zero")
        mixing_data = np.asarray(self.InitalData.copy())
        # mixing_data = self.InitalData.copy()
        if mixing_data.ndim == 1:
            mixing_data = mixing_data[np.newaxis, :]

        mixing_sr = self.SampleRate
        if(HumNoiseValue>0):
            mixing_data,mixing_sr = self.AddingHumNoise_Single(mixing_data, mixing_sr,HumNoiseValue)
        if(GaussianNoiseValue>0):
            mixing_data,mixing_sr = self.AddingGaussianNoise_Single(mixing_data, mixing_sr,GaussianNoiseValue)
        if(ClipPercentValue>0):
            mixing_data,mixing_sr = self.AddingClippingDistortionByFloater_Single(mixing_data, mixing_sr, ClipPercentValue)
        if(DropOutSampleNum>0):
            mixing_data,mixing_sr = self.AddingSampleSizeDropout_Single(mixing_data, mixing_sr, DropOutSampleNum)
        #print("ClippingNoise is done")
        mixing_data,mixing_sr = self.MixingSingleAudio(mixing_data,mixing_sr)

        return mixing_data,mixing_sr
    
    def AddingHumNoise_Single(self,data,srate,manipulation_value):
#        print(f"Before, there is {NEUtil.count_zeros(data)} in zero")
        if manipulation_value!= 0:
            data = NoiseEvalEffect.Add_HummingNoise(data, srate, manipulation_value)
#        print(f"After, there is {NEUtil.count_zeros(data)} in zero")
        return data,srate

    def AddingGaussianNoise_Single(self,data,srate,manipulation_value):
        if manipulation_value!= 0:
            V_AddGaussian_Transform = AddGaussianSNR(min_snr_db=manipulation_value,max_snr_db=manipulation_value,p=1.0)
            data = V_AddGaussian_Transform(data, sample_rate=srate)
        return data,srate
    
    def AddingClippingDistortionByFloater_Single(self,data,srate,manipulation_value):
        if manipulation_value!= 0:
            #V_AddClipping_Transform = Lambda(transform=NoiseEvalEffect.ClippingDistortionWithFloatingThreshold, p=1.0)
            #data = V_AddClipping_Transform(data, srate, manipulation_value)
            data = NoiseEvalEffect.ClippingDistortionWithFloatingThreshold(data, srate, manipulation_value)
        return data,srate

    def AddingSampleSizeDropout_Single(self,data,srate,manipulation_value):
#        print(f"Before, there is {NEUtil.count_zeros(data)} in zero")
        if manipulation_value!= 0:
            data = NoiseEvalEffect.DropingSamplesBySampleSizeAndNum(data, srate, manipulation_value)
#        print(f"After, there is {NEUtil.count_zeros(data)} in zero")
        return data,srate
    
    def MixingSingleAudio(self,mixing_data,mixing_sr):
        #Normalize_Transform = Normalize(p=1.0)
        #mixing_data = Normalize_Transform(mixing_data, mixing_sr)
        Lufs_Transform = LoudnessNormalization(min_lufs=-14.0,max_lufs=-14.0,p=1.0)
        mixing_data = Lufs_Transform(mixing_data, mixing_sr)
        self.MixingRMS = round(NEUtil.calculate_rms_dB(mixing_data),2)
        self.MixingClippingPercentage, self.MixingClippingSamplesNum = NEUtil.calcaulate_cliped_samples(mixing_data)
        print(f"After LUFS, the mixing ouput in the RMS, Total: {round(NEUtil.calculate_rms_dB(mixing_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(mixing_data)}")
        return mixing_data,mixing_sr
