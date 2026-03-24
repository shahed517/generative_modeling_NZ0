from torch.utils.data import Dataset
from scipy.stats import zscore
import os, glob, h5py, torch
from tqdm import tqdm 
import numpy as np

class NWBDataset(Dataset):
    def __init__(
        self, root_dir, data_type='Both', duration=2,
        channels_to_keep=None, debug_mode=False
    ):
        """Dataset class for unlabeled data
        Args:
            data_type (str): HG | LFC | Both
            duration (int): Length of a training sample in seconds
            channels_to_keep: Tell which channels to keep. Must be 0-indexed. Note: EKG channels...
            ... have already been removed internally in this class.
        """
        self.root_dir = root_dir
        self.duration = duration
        self.data_type = data_type
        self.channels_to_keep = channels_to_keep if channels_to_keep\
            is None else np.array(channels_to_keep)

        self.samp_rate = None
        
        nwb_files = glob.glob(
            os.path.join(
                self.root_dir, 'NZ*.nwb'
            )
        )

        if debug_mode:
            nwb_files = nwb_files[:3]

        self.data = []
        
        for ii, nwb in enumerate(tqdm(nwb_files)):
            with h5py.File(
                nwb, mode='r', libver='latest',
                swmr=True
            ) as f:
                if ii == 0:
                    ekg_channels = f['general']['extracellular_ephys']['electrodes']['bad'][:]
                
                if data_type.lower() == 'hg':
                    raw_data = f['processing']['ecephys']['LFP']\
                        ['high gamma (CAR 200.0Hz)']['data'][:, ~ekg_channels]

                    if self.samp_rate is None:
                        self.samp_rate = int(f['processing']['ecephys']['LFP']\
                        ['high gamma (CAR 200.0Hz)']['starting_time'].attrs['rate'].item())

                elif data_type.lower() == 'lfc':
                    raw_data = f['processing']['ecephys']['LFP']\
                        ['preprocessed LFC (CAR 200.0Hz)']['data'][:, ~ekg_channels]

                    if self.samp_rate is None:
                        self.samp_rate = int(f['processing']['ecephys']['LFP']\
                        ['preprocessed LFC (CAR 200.0Hz)']['starting_time'].attrs['rate'].item())
                elif data_type.lower() == 'both':
                    raw_data = np.concat(
                        (f['processing']['ecephys']['LFP']\
                            ['high gamma (CAR 200.0Hz)']['data'][:, ~ekg_channels],
                        f['processing']['ecephys']['LFP']\
                            ['preprocessed LFC (CAR 200.0Hz)']['data'][:, ~ekg_channels]),
                        axis=-1
                    )

                    if self.samp_rate is None:
                        hg_rate = int(f['processing']['ecephys']['LFP']\
                        ['high gamma (CAR 200.0Hz)']['starting_time'].attrs['rate'].item())
                        lfc_rate = int(f['processing']['ecephys']['LFP']\
                        ['preprocessed LFC (CAR 200.0Hz)']['starting_time'].attrs['rate'].item())

                        assert hg_rate == lfc_rate, "High Gamma and LFCs must have the same sampling rate."

                        self.samp_rate = hg_rate
                else:
                    raise ValueError("data_type must be 'HG' or 'LFC' or 'Both'.")
                
                self.data.append(raw_data)

        num_of_duration_segments = [
            len(dat)//(self.duration * self.samp_rate) for dat in self.data
        ]

        self.all_duration_segments = [
            [(nwb_idx, m_idx) for m_idx in range(min_idx)] for nwb_idx, min_idx
            in zip(range(len(self.data)), num_of_duration_segments)
        ]
        self.all_duration_segments = np.concat(self.all_duration_segments).tolist()

    def __len__(self):
        return len(self.all_duration_segments)

    def __getitem__(self, idx):
        nwb_idx, segment_idx = self.all_duration_segments[idx]

        start_idx = segment_idx * self.duration * self.samp_rate
        end_idx = start_idx + self.duration * self.samp_rate
        selected_data = self.data[nwb_idx][start_idx:end_idx]

        norm_data = zscore(
            selected_data,
            axis=0
        )
        
        tensor_data = torch.from_numpy(norm_data)

        return tensor_data


if __name__ == "__main__":
    ds = NWBDataset(root_dir="/depot/jgmakin/data/NZ0000/NWB/", data_type='both', duration=2, debug_mode='True') 
    ## debug_mode = True only loads a small number of .nwb files for quick debugging
    print(len(ds))
    random_idx = np.random.randint(len(ds))
    sample_eeg = ds[random_idx]
    print(sample_eeg.shape)
