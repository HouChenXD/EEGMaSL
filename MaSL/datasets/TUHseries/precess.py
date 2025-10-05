import os
import pickle

from multiprocessing import Pool
import numpy as np
import mne

# we need these channels
# (signals[signal_names['EEG FP1-REF']] - signals[signal_names['EEG F7-REF']],  # 0
# (signals[signal_names['EEG F7-REF']] - signals[signal_names['EEG T3-REF']]),  # 1
# (signals[signal_names['EEG T3-REF']] - signals[signal_names['EEG T5-REF']]),  # 2
# (signals[signal_names['EEG T5-REF']] - signals[signal_names['EEG O1-REF']]),  # 3
# (signals[signal_names['EEG FP2-REF']] - signals[signal_names['EEG F8-REF']]),  # 4
# (signals[signal_names['EEG F8-REF']] - signals[signal_names['EEG T4-REF']]),  # 5
# (signals[signal_names['EEG T4-REF']] - signals[signal_names['EEG T6-REF']]),  # 6
# (signals[signal_names['EEG T6-REF']] - signals[signal_names['EEG O2-REF']]),  # 7
# (signals[signal_names['EEG FP1-REF']] - signals[signal_names['EEG F3-REF']]),  # 14
# (signals[signal_names['EEG F3-REF']] - signals[signal_names['EEG C3-REF']]),  # 15
# (signals[signal_names['EEG C3-REF']] - signals[signal_names['EEG P3-REF']]),  # 16
# (signals[signal_names['EEG P3-REF']] - signals[signal_names['EEG O1-REF']]),  # 17
# (signals[signal_names['EEG FP2-REF']] - signals[signal_names['EEG F4-REF']]),  # 18
# (signals[signal_names['EEG F4-REF']] - signals[signal_names['EEG C4-REF']]),  # 19
# (signals[signal_names['EEG C4-REF']] - signals[signal_names['EEG P4-REF']]),  # 20
# (signals[signal_names['EEG P4-REF']] - signals[signal_names['EEG O2-REF']]))) # 21


def split_and_dump(params):
    fetch_folder, dump_folder = params
    filelist = os.listdir(fetch_folder)
    
    for file in filelist:
        print("process", file)
        file_path = os.path.join(fetch_folder, file)
        raw = mne.io.read_raw_edf(file_path, preload=True)
        # raw.resample(200)
        ch_name = raw.ch_names
        raw_data = raw.get_data()
        channeled_data = raw_data.copy()[:16]
        try:
            channeled_data[0] = (
                raw_data[ch_name.index("EEG FP1-REF")]
                - raw_data[ch_name.index("EEG F7-REF")]
            )
            channeled_data[1] = (
                raw_data[ch_name.index("EEG F7-REF")]
                - raw_data[ch_name.index("EEG T3-REF")]
            )
            channeled_data[2] = (
                raw_data[ch_name.index("EEG T3-REF")]
                - raw_data[ch_name.index("EEG T5-REF")]
            )
            channeled_data[3] = (
                raw_data[ch_name.index("EEG T5-REF")]
                - raw_data[ch_name.index("EEG O1-REF")]
            )
            channeled_data[4] = (
                raw_data[ch_name.index("EEG FP2-REF")]
                - raw_data[ch_name.index("EEG F8-REF")]
            )
            channeled_data[5] = (
                raw_data[ch_name.index("EEG F8-REF")]
                - raw_data[ch_name.index("EEG T4-REF")]
            )
            channeled_data[6] = (
                raw_data[ch_name.index("EEG T4-REF")]
                - raw_data[ch_name.index("EEG T6-REF")]
            )
            channeled_data[7] = (
                raw_data[ch_name.index("EEG T6-REF")]
                - raw_data[ch_name.index("EEG O2-REF")]
            )
            channeled_data[8] = (
                raw_data[ch_name.index("EEG FP1-REF")]
                - raw_data[ch_name.index("EEG F3-REF")]
            )
            channeled_data[9] = (
                raw_data[ch_name.index("EEG F3-REF")]
                - raw_data[ch_name.index("EEG C3-REF")]
            )
            channeled_data[10] = (
                raw_data[ch_name.index("EEG C3-REF")]
                - raw_data[ch_name.index("EEG P3-REF")]
            )
            channeled_data[11] = (
                raw_data[ch_name.index("EEG P3-REF")]
                - raw_data[ch_name.index("EEG O1-REF")]
            )
            channeled_data[12] = (
                raw_data[ch_name.index("EEG FP2-REF")]
                - raw_data[ch_name.index("EEG F4-REF")]
            )
            channeled_data[13] = (
                raw_data[ch_name.index("EEG F4-REF")]
                - raw_data[ch_name.index("EEG C4-REF")]
            )
            channeled_data[14] = (
                raw_data[ch_name.index("EEG C4-REF")]
                - raw_data[ch_name.index("EEG P4-REF")]
            )
            channeled_data[15] = (
                raw_data[ch_name.index("EEG P4-REF")]
                - raw_data[ch_name.index("EEG O2-REF")]
            )
        except:
            with open("tuar-process-error-files.txt", "a") as f:
                f.write(file + "\n")
            continue
        for i in range(channeled_data.shape[1] // 2560):
            dump_path = os.path.join(
                dump_folder, file.split(".")[0] + "_" + str(i) + ".pkl"
            )
            pickle.dump(
                {"X": channeled_data[:, i * 2560 : (i + 1) * 2560]},
                open(dump_path, "wb"),
            )


if __name__ == "__main__":
    """
    TUAB dataset is downloaded from https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml
    """
    parameters = []
    file = "/mnt/replace_disk/EEG_data/TUHseries/TUSZ/"
    root = os.path.join(file, "edf")
    savepath = os.path.join(file, "processed")
    
    parameters.append((root,savepath))
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    
    # split and dump in parallel
    with Pool(processes=1) as pool:
        # Use the pool.map function to apply the square function to each element in the numbers list
        result = pool.map(split_and_dump, parameters)
