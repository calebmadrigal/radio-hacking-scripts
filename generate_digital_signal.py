import numpy as np
import sys

SAMPLE_BITSIZE = 16
MAX_AMP_16BIT = int(2**SAMPLE_BITSIZE/2 - 1)

DEFAULT_RATIOS = {
    '_': 1,
    '0': 1,
    '1': 3
}

DEFAULT_AMP_MAP = {
    'LOW': MAX_AMP_16BIT * .02,
    'HIGH': MAX_AMP_16BIT
}

def get_modulation_array(binary_data, sample_rate, baud, sig_ratios, amp_map, dtype=np.int16):
    data_points_in_bit = int(sample_rate * 1/baud)
    modulation_array = np.array([], dtype=dtype)

    # To describe this general algorithms, I'll use the specific concrete pulse ratios:
    #    '_': 1,
    #    '0': 1,
    #    '1': 3
    # Meaning that a "1" should be 3x longer than a "0" or a "space" pulse. Now since we need a space
    # between "1"s (as well as "0"), we can calculate that the pulse for a "1" should be 3/4 of the bit
    # and the pulse for a "0" should be 1/4 of the bit (since for the 1, it's 3 parts "1" and 1 part "space")
    one_pulse_len = int((sig_ratios['1'] / (sig_ratios['1'] + sig_ratios['_'])) * data_points_in_bit)
    one_space_len = data_points_in_bit - one_pulse_len
    zero_pulse_len = int((sig_ratios['0'] / (sig_ratios['1'] + sig_ratios['_'])) * data_points_in_bit)
    zero_space_len = data_points_in_bit - zero_pulse_len
    
    modulated_one_bit = np.append(np.full(one_pulse_len, amp_map['HIGH'], dtype=dtype),
                                  np.full(one_space_len, amp_map['LOW'], dtype=dtype))
    modulated_zero_bit = np.append(np.full(zero_pulse_len, amp_map['HIGH'], dtype=dtype),
                                   np.full(zero_space_len, amp_map['LOW'], dtype=dtype))
    
    for bit in binary_data:
        modulated_bit = modulated_one_bit if bit == '1' else modulated_zero_bit
        modulation_array = np.append(modulation_array, modulated_bit)
    return modulation_array

def generate_on_off_key_signal(binary_data, carrier_wave_freq, sample_rate,
                               baud, sig_ratios=DEFAULT_RATIOS, amp_map=DEFAULT_AMP_MAP, dtype=np.int16):
    signal_len_secs = len(binary_data) * (1/baud)
    t = np.linspace(0, signal_len_secs, sample_rate * signal_len_secs)
    
    # Using Euler's formula to generate a complex sinusoidal wave
    carrier_wave = 1 * np.e**(carrier_wave_freq * 2 * np.pi * (0+1j) * t)
    modulation_array = get_modulation_array(binary_data, sample_rate, baud, sig_ratios, amp_map, dtype)
    
    # Pad (or trim) the modulation array to match the length of the carrier wave
    if len(carrier_wave) > len(modulation_array):
        pad_len = len(carrier_wave) - len(modulation_array)
        modulation_array = np.append(modulation_array, np.full(pad_len, amp_map['LOW'], dtype=dtype))
    elif len(carrier_wave) < len(modulation_array):
        modulation_array = modulation_array[:len(carrier_wave)]
    
    # Modulate by superwave
    super_wave_freq = carrier_wave_freq / (160*2)
    super_wave = 1 * np.e**(super_wave_freq * 2 * np.pi * (0+1j) * t)
    
    return t, carrier_wave * modulation_array * super_wave

def generate_pulse(bit_val, carrier_wave_freq, sample_rate, baud, multiple_of_bit_len,
                   amp_map=DEFAULT_AMP_MAP, dtype=np.int16):
    signal_len_secs = multiple_of_bit_len * (1/baud)
    t = np.linspace(0, signal_len_secs, sample_rate * signal_len_secs)
    
    high_or_low = 'HIGH' if bit_val == '1' else 'LOW'
    pulse = amp_map[high_or_low] * np.e**(carrier_wave_freq * 2 * np.pi * (0+1j) * t)
    return t, pulse

def join_all_arrays(array_list):
    joined = array_list[0]
    for a in array_list[1:]:
        joined = np.append(joined, a)
    return joined

def write_pcm_file(signal_data, file_path, dtype='complex64'):
    np.array(signal_data).astype(dtype).tofile(file_path)

on_signal  = '0110100010000000'
off_signal = '0110100001000000'

binary_data = off_signal
carrier_wave_freq = 315e6
sample_rate = 2e6
baud = 205 * 2
#out_path = 'raw_data/generated_outlet_on.pcm'
out_path = 'raw_data/generated_outlet_off.pcm'

complex64_amp_map = {
    'LOW': 1.4 * .02,
    'HIGH': 1.4
}

t, complex_signal = generate_on_off_key_signal(binary_data, carrier_wave_freq, sample_rate, baud, amp_map=complex64_amp_map, dtype='complex64')
t2, signal_header = generate_pulse('1', carrier_wave_freq, sample_rate, baud, 3.85, amp_map=complex64_amp_map, dtype='complex64')
t3, signal_spacer = generate_pulse('0', carrier_wave_freq, sample_rate, baud, 3.78, amp_map=complex64_amp_map, dtype='complex64')

full_signal = join_all_arrays([signal_header] + ([complex_signal, signal_spacer] * 12))
write_pcm_file(full_signal, out_path, dtype='complex64')
