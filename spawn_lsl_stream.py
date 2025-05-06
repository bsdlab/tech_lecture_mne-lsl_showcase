import time
import uuid

import numpy as np
from matplotlib import pyplot as plt
from mne.io import read_raw_fif

from mne_lsl.datasets import sample
from mne_lsl.player import PlayerLSL


def stream_data_from_fif_file() -> PlayerLSL:
    """
    Streams data from a FIF file using the MNE-LSL library.

    This function reads a FIF file containing MEG data, sets up a PlayerLSL object
    to stream the data, and starts the streaming process. The data is streamed in
    chunks of 200 samples.

    Returns
    -------
        PlayerLSL: An instance of PlayerLSL configured to stream the data from the FIF file.
    """
    fname = sample.data_path() / "mne-sample" / "sample_audvis_raw.fif"
    raw = read_raw_fif(fname, preload=False).pick(("meg", "stim")).load_data()

    # reduce number of channels
    raw = raw.pick(raw.ch_names[:5] + ["STI 014"])

    source_id = uuid.uuid4().hex  # get a UUID for the source stream

    # Uses:
    # `from concurrent.futures import ThreadPoolExecutor``
    # to spawn a thread that is streaming data from the file
    player = PlayerLSL(
        raw,
        chunk_size=100,
        name="tutorial-epochs-1",
        source_id=source_id,
        annotations=False,
    ).start()

    return player


def main():
    player = stream_data_from_fif_file()
    q = input("Press any key to stop streaming...")
    player.stop()
    print("Streaming stopped.")


if __name__ == "__main__":
    main()
