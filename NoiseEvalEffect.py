from audiomentations import Lambda
import numpy as np
from numpy.typing import NDArray
from cylimiter import Limiter as CLimiter
import NoiseEvalUtil as NoiseEvalUtil

## adding certain random dropouts on the sample, the drop samplenum was pointed.
## using np to setting the randomness on the samples 
def DropingSamplesByNum(samples,sample_rate  ,drop_samplenum):
    if(drop_samplenum > 0):
        num_samples = len(samples[0])
        drop_indices = np.random.choice(num_samples,drop_samplenum,replace=False)
        samples[0][drop_indices] = 0
    return samples


##Because the Drop happen in most time by CPU is busy to handle the current runing clock by drop or throw a random number to sample;
##therefroce by default the 32samples is chosen, and drop time means how many drop happen in this transmission
def DropingSamplesBySampleSizeAndNum(samples, sample_rate, drop_time,sampleSize=32):
    if(drop_time > 0):
        audio_signal = samples.copy()
        #print(f"There is {NoiseEvalUtil.count_zeros(audio_signal[0])} zero samples before")
        num_samples = len(audio_signal[0])
        num_packages = num_samples // sampleSize
        print(f"There are {num_packages} packages")
        if(drop_time >= num_samples):
            raise ValueError(
            "It is not possible set all the samples to zero"
            )
        drop_indices = np.random.choice(num_packages, drop_time, replace=False)
            
        for idx in drop_indices:
            start_idx = idx * sampleSize
            end_idx = min(start_idx + sampleSize, num_samples)  # Handle boundary conditions
            audio_signal[0][start_idx:end_idx] = 0
        #print(f"There is {NoiseEvalUtil.count_zeros(audio_signal[0])} zero samples after")
        return audio_signal
    else:
        return samples


def DropingFixedSamplesBySampleSizeAndNum(samples, sample_rate, position, drop_time,sampleSize=32):
    if(drop_time > 0):
        audio_signal = samples.copy()
        print(f"There is {NoiseEvalUtil.count_zeros(audio_signal[0])} zero samples before")
        num_samples = len(audio_signal[0])
        num_packages = num_samples // sampleSize
        #print(f"There are {num_packages} packages")
        if(drop_time >= num_samples):
            raise ValueError(
            "It is not possible set all the samples to zero"
            )
        drop_indices = position           
        for idx in drop_indices:
            start_idx = idx * sampleSize
            end_idx = min(start_idx + sampleSize, num_samples)  # Handle boundary conditions
            audio_signal[0][start_idx:end_idx] = 0
        #print(f"There is {NoiseEvalUtil.count_zeros(audio_signal[0])} zero samples after")
        return audio_signal
    else:
        return samples

##It is the function to introducing certain amount of Humming noise into context
##by default use the 50Hz and its third hamonic frequency 150Hz.
##The method adding them regarding to the SNR level to the original signal.currently the both 
##sub freqnecy setting in the same level
def Add_HummingNoise(samples, sample_rate, snr_db, frequencies=[50,150]):
    #if amplitudes is None:
    # Set default amplitude to 0.5 for all frequencies if not provided
    #    amplitudes = [0.5] * len(frequencies)
    if (snr_db>0):
        originalRMS = NoiseEvalUtil.calculate_rms(samples)
#        print(f"The original level of signal is {originalRMS}")
        noise_RMS = NoiseEvalUtil.calculate_desired_noise_rms(originalRMS,snr_db)
#        print(f"The noise level of signal is {noise_RMS}")

        # Create a time array based on the length of the audio signal
        t = np.arange(len(samples[0])) / sample_rate

        # Initialize the new signal as a copy of the original audio signal
        new_audio_signal = np.copy(samples)

        # Add each sine wave to the audio signal
        for freq in frequencies:
            sine_wave = noise_RMS * np.sin(2 * np.pi * freq * t)
            new_audio_signal += sine_wave
        return new_audio_signal
    else:
        return samples


##The simple method that create the clipping ratio in fixed
def ClippingDistortionWithFloatingThreshold(samples, sample_rate, clipping_rate):
    clipping_rate = round(clipping_rate, 1)
    lower_percentile_threshold = clipping_rate / 2
    lower_threshold, upper_threshold = np.percentile(
            samples, [lower_percentile_threshold, 100 - lower_percentile_threshold]
        )
    samples = np.clip(samples, lower_threshold, upper_threshold)
    return samples

###This Method Only using Test Climiter with out the Delay setting
def Dynamic_FullPara_BClimiter(samples,srate,threshold_db,attack_seconds,release_seconds):
    print("Running Limiter")
    attack = NoiseEvalUtil.convert_time_to_coefficient(
        attack_seconds, srate
    )
    release = NoiseEvalUtil.convert_time_to_coefficient(
        release_seconds, srate
    )
    # instead of delaying the signal by 60% of the attack time by default
    delay = 1
    #delay = max(round(0.6 * attack_seconds * srate), 1)
    threshold_factor = NoiseEvalUtil.get_max_abs_amplitude(samples)
    threshold_ratio  = threshold_factor * NoiseEvalUtil.convert_decibels_to_amplitude_ratio(threshold_db)
    
    #print(f"applied configuration attack:{attack},release:{release},threshold:{threshold_ratio},delay:{delay}")
    limiter = CLimiter(
        attack=attack,
        release=release,
        delay=delay,
        threshold=threshold_ratio,
    )
    samples = limiter.limit(samples)
    return samples,srate