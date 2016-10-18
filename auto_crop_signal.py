import sys
import scipy
import numpy as np

def read_pcm_file(file_path, file_type=scipy.complex64):
    with open(file_path, 'rb') as f:
        return scipy.fromfile(f, dtype=file_type)

def write_pcm_file(file_path, signal_data, file_type='complex64'):
    np.array(signal_data).astype('complex64').tofile(file_path)

def auto_crop_signal(signal_data, margin_percent=5, num_chunks=16):
    """ Break the signal into chunks, and find the chunk there is the largest
    jump from quiet to loud (start index), and the largest jump from 
    loud to quiet (stop index). """
    chunk_size = int(len(signal_data) / num_chunks)
    largest_increase_index = 0
    largest_increase_size = -999999999
    largest_decrease_index = chunk_size * num_chunks
    largest_decrease_size = 999999999
    last_chunk_sum = sum([abs(i) for i in signal_data[0:chunk_size]])
    for chunk_start in range(0, len(signal_data), chunk_size):
        chunk = signal_data[chunk_start:chunk_start+chunk_size]
        # Don't consider the last chunk if it's not a full chunk,
        # since that will likely yield the smallest sum
        if len(chunk) < chunk_size:
            continue
        chunk_sum = sum([abs(i) for i in chunk])
        chunk_diff = chunk_sum - last_chunk_sum
        last_chunk_sum = chunk_sum
        if chunk_diff > largest_increase_size:
            largest_increase_size = chunk_diff
            largest_increase_index = chunk_start
        if chunk_diff < largest_decrease_size:
            largest_decrease_size = chunk_diff
            largest_decrease_index = chunk_start
    margin = int((largest_decrease_index - largest_increase_index) * (margin_percent / 100))
    return signal_data[largest_increase_index-margin:largest_decrease_index+margin]

if __name__ == '__main__':
    try:
        in_file_path = sys.argv[1]
        out_file_path = sys.argv[2]
    except IndexError:
        print('Usage: python auto_crop_signal.py <in file> <out file>')
        sys.exit(1)
    signal_data = read_pcm_file(in_file_path)
    cropped_signal = auto_crop_signal(signal_data)
    write_pcm_file(out_file_path, cropped_signal)
    print('Wrote auto-cropped signal to:', out_file_path)

