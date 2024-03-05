import os
import glob
import pickle
import random
import numpy as np
import obspy

from scipy.signal.windows import tukey


def save_obj(dictionary, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dictionary, f, protocol=0)


def load_obj(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def str2bool(string):
    """
    Returns a bool from a string of "True" or "False"
    :param string: type str, {"True", "False"}
    :return: type bool
    :raise: ValueError
            If string ist not "True", "true", "False", "false"

    # >>> str2bool("True")
    # True
    # >>> str2bool("False")
    # False
    """
    if string not in ["True", "true", "False", "false"]:
        msg = "string is not True or False"
        raise ValueError(msg)
    else:
        if string in ["True", "true"]:
            return True
        elif string in ["False", "false"]:
            return False


def readtxt(fname):
    """
    readtxt reads a file containing parameters. "#" are comments and will be ignored. The parameter's name and its value
    are seperated by "=". First mention the name and afterwards the value. Example for a short file:

    # Settings for a test
    numbers = 12         # Gives numbers
    output  = save.pdf   # Saves the output

    The function returns a dictionary, containing parameter and its value.

    :param fname: Path and name of the file that will be read
    :type fname: str

    :return: dict
    """

    # Open file and creating empty dictionary
    fopen = open(fname, "r")
    parameters = {}
    line = fopen.readline()

    # Reading through each line in fname
    # Lines containing "#" in first values are ignored
    while line:
        if "#" not in line[:5] and "=" in line:
            param = line.split("=")          # Split between value name and rest
            name = param[0]                  # Get name of value
            value = param[1].split("#")[0]   # Split between comment and value and extract value
            name = name.strip()
            value = value.strip()            # Remove whitespace

            # Try to convert value into float
            try:
                value = float(value)
            except ValueError:
                pass

            # Otherwise value is of type string
            if isinstance(value, str) is True:
                value = value.split("\n")[0]

                # Test for None
                if value.lower() == 'none':
                    value = None

                # Try to convert value into bool
                try:
                    value = str2bool(value)
                except ValueError:
                    pass

                # Test for tuple
                try:
                    value = tuple(map(int, value.split(', ')))
                except:
                    pass

            parameters.update({name: value})
        line = fopen.readline()

    fopen.close()

    return parameters


def rms(x):
    """
    Root-mean-square of array x
    :param x:
    :return:
    """
    # Remove mean
    x = x - np.mean(x)
    return np.sqrt(np.sum(x ** 2) / x.shape[0])


def signal_to_noise_ratio(signal, noise, decibel=True):
    """
    SNR in dB
    :param signal:
    :param noise:
    :param decibel:
    :return:
    """
    if decibel is True:
        return 20 * np.log10(rms(signal) / rms(noise))
    else:
        return rms(signal) / rms(noise)


def snr(signal, noise, decibel=True):
    """
    Wrapper for signal-to-noise ratio
    """
    return signal_to_noise_ratio(signal=signal, noise=noise, decibel=decibel)


def snr_pick(trace: obspy.Trace,
             picktime: obspy.UTCDateTime,
             window=5,
             **kwargs):
    """
    Computes SNR with a certain time window around a pick
    """
    pick_sample = int((picktime - trace.stats.starttime) * trace.stats.sampling_rate)
    window_len = int(window * trace.stats.sampling_rate)

    if pick_sample - window_len < 0:
        noise_win_begin = 0
    else:
        noise_win_begin = pick_sample - window_len

    return snr(signal=trace.data[pick_sample:pick_sample + window_len],
               noise=trace.data[noise_win_begin:pick_sample],
               **kwargs)


def pick_dict(picks):
    """
    Create dictionary for each station that contains P and S phases

    returns: dict
             keys: network_code.station_code
             values: dict with phase_hints and time of pick
    """
    station_pick_dct = {}
    station_status_dct = {}
    for pick in picks:
        station_id = f"{pick.waveform_id.network_code}.{pick.waveform_id.station_code}"
        if station_id in station_pick_dct.keys():
            station_pick_dct[station_id].update({pick["phase_hint"]: pick["time"]})
            station_status_dct[station_id].update({f"{pick['phase_hint']}_status": pick.evaluation_mode})
        else:
            station_pick_dct.update({station_id: {pick["phase_hint"]: pick["time"]}})
            station_status_dct.update({station_id: {f"{pick['phase_hint']}_status": pick.evaluation_mode}})

    return station_pick_dct, station_status_dct


def get_picks(event):
    """
    Function creates a dictionary with all picks from 'event.origins[0].arrivals' and 'event.picks'.
    The keys of the returned dictionary are named 'network.station' and contains a further dictionary with
    the phase hints and UTCDateTimes of the different phases.

    Works only for FORGE events
    """
    # Test if events has arrivals and picks
    if len(event.origins[0].arrivals) == 0:
        msg = "Event does not have arrivals."
        raise ValueError(msg)

    if len(event.picks) == 0:
        msg = "Event does not have picks."
        raise ValueError(msg)

    # Create list of all pick resource_ids
    pick_rid = [pick.resource_id.id.split("/")[-1] for pick in event.picks]

    # Loop over each arrival
    for arrival in event.origins[0].arrivals:
        # Find resource_id of arrival in pick_rid
        try:
            pick_index = pick_rid.index(arrival.resource_id.id.split("/")[-1])
            # Add phase to picks
            event.picks[pick_index].phase_hint = arrival.phase
        except ValueError:
            pass

    return pick_dict(event.picks)


def sort_event_list(event_list):
    """
    Sort event list by dates, from early to late
    :param event_list:
    :return:
    """
    # Create list with all dates
    dates = [event.origins[0].time.datetime for event in event_list]

    # Sort event_list by dates
    sorted_events = [x for _, x in zip(sorted(dates), event_list)]

    return sorted_events


def merge_catalogs(catalogs: list, sort: bool = True) -> obspy.Catalog:
    all_events = []
    for catalog in catalogs:
        for event in catalog.events:
            all_events.append(event)

    # Sort events by date
    if sort is True:
        all_events = sort_event_list(event_list=all_events)

    return obspy.Catalog(events=all_events)


def normalize(array: np.array):
    """
    Removing mean from array and dividing by its standard deviation.
    :param array: numpy array

    :returns: normalized array
    """
    return (array - np.mean(array)) / np.std(array)


def is_nan(num):
    return num != num


def taper_array(array: np.array, alpha: float = 0.05):
    """
    Applies an tukey/cosine taper on array.
    :param array: numpy array
    :param alpha: float, optional, default is 0.05
                 Shape parameter of the Tukey window, representing the fraction of the window inside the cosine tapered
                 region. If zero, the Tukey window is equivalent to a rectangular window. If one, the Tukey window is
                 equivalent to a Hann window. (See. scipy.signal.window.tukey)
    :return: tapered array
    """
    taper = tukey(len(array), alpha=alpha)
    tapered_array = array * taper

    return tapered_array


def shift_array(array: np.array,
                length: int = 6001,
                taper=False):
    """
    Shifts numpy array randomly to the left or right for data augmentation.
    For shift zeros are added to keep the same length.

    :param array: numpy array
    :param length: Final length of the input array. Default is 6001
    :param taper: If True, array is tapered with a cosine taper. Default is False.
    :return: shifted numpy array
    """
    # Crop array by given length. Note the signal of interest has to be in this part!
    array = array[:length]
    shift = random.randint(-int(len(array) / 2), int(len(array) / 2))
    result = np.empty_like(array)
    if shift > 0:
        # Shift to the right
        result[:shift] = 0                     # fill_value
        shifted_array = array[:-shift]
        if taper is True:
            result[shift:] = taper_array(shifted_array)
        else:
            result[shift:] = shifted_array
    elif shift < 0:
        # Shift to the left
        result[shift:] = 0                     # fill_value
        shifted_array = array[-shift:]
        if taper is True:
            result[:shift] = taper_array(shifted_array)
        else:
            result[:shift] = shifted_array
    else:
        result[:] = array

    return result


def remove_file(filename: str, verbose=False):
    """
    Function deletes filenames.
    """
    if verbose is True:
        print("File with error:", filename)
    os.remove(filename)


def check_npz(npz_filename: str, zero_check=True, verbose=False):
    """
    Trys to read npz file. If not the file is deleted
    :param npz_filename: filename to check
    :param zero_check: If True, files that contains almost zeros will be deleted.
    :param verbose: Default False. If True, files that are deleted will be printed.
    """
    try:
        dataset = np.load(npz_filename)
        data = dataset["data"]
        if zero_check is True:
            if np.max(np.abs(data)) <= 1e-15:
                remove_file(filename=npz_filename, verbose=verbose)
    except Exception:
        remove_file(filename=npz_filename, verbose=verbose)


def check_signal_files(signal_dir: str, extension="npz", verbose=False):
    """
    Function checks if signal files contain a file with zeros, since these files are not allowed for training.
    :param signal_dir: Full pathname for signal files
    :param extension: Extension of filenames, default is npz.
    :param verbose: Default False. If True, files that are deleted will be printed.
    """
    files = glob.glob(os.path.join(signal_dir, f"*.{extension}"))
    # Loop over all files and check whether the data can be read and the file does not contain zeros only.
    for filename in files:
        check_npz(npz_filename=filename, zero_check=True, verbose=verbose)


def check_noise_files(noise_dirname: str, extension="npz", **kwargs):
    """
    Reads all files in noise_dirname to check all .npz files.
    If a file cannot be read it is deleted.

    :param noise_dirname: Directory of noise npz files
    :param extension: Extension of filenames, default is npz.
    """
    files = glob.glob(os.path.join(noise_dirname, f"*.{extension}"))
    for filename in files:
        check_npz(npz_filename=filename, **kwargs)


def old_data_augmentation(signal_npz_file, noise_npz_file, ts_length=6001):
    signal = np.load(signal_npz_file)
    noise = np.load(noise_npz_file)
    # Read signal and noise from npz files
    try:
        p_samp = signal["itp"]  # Sample of P-arrival
        s_samp = signal["its"]  # Sample of S-arrival
    except KeyError:
        p_samp = None
        s_samp = None

    # Read data arrays from signal and noise
    signal = signal["data"]
    noise = noise["data"][:ts_length]

    # epsilon = 0  # Avoiding zeros in added arrays
    # shift1 = np.random.uniform(low=-1, high=1, size=int(self.ts_length - s_samp)) * epsilon
    # TODO: Check data augmentation if correct; Ignore itp and its
    # signal = shift_array(array=signal)
    # signal = signal[:self.ts_length]
    if p_samp and s_samp:
        if int(ts_length - s_samp) < 0:
            shift1 = np.zeros(0)
        else:
            shift1 = np.zeros(shape=int(ts_length - s_samp))
        signal = np.concatenate((shift1, signal))
        # Cut signal to length of ts_length and arrival of P-phase is included
        p_samp += len(shift1)
        s_samp += len(shift1)
        start = random.randint(0, p_samp)
        signal = signal[start:start + ts_length]
    else:  # XXX Add case just for p_samp
        if ts_length > len(signal):
            start = random.randint(0, len(signal) - ts_length - 1)
            signal = signal[start:int(start + ts_length)]
        else:
            signal = signal[:ts_length]

    return signal, noise


def phase_color(phase):
    if phase == "P":
        return "b"
    elif phase == "S":
        return "r"
    else:
        raise Exception



if __name__ == "__main__":
    from obspy import read_events

    cat = read_events("/home/jheuel/scratch/ai_datasets/forge/quakeml.xml")
    pick_dct = get_picks(cat.events[0])
    print(pick_dct)
