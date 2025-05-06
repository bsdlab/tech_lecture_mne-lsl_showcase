# Extending the example from [mne-lsl](https://mne.tools/mne-lsl/stable/generated/tutorials/40_epochs.html#sphx-glr-generated-tutorials-40-epochs-py)
# for use within the ABCI lecture
#
# NOTE:to future lecturer:
#   - run the code before the lecture, so that data is downloaded
#
import time
import uuid

import numpy as np
from matplotlib import pyplot as plt
from mne import Epochs, EpochsArray, annotations_from_events, find_events
from mne.io import read_raw_fif
from scipy.sparse import data

from mne_lsl.datasets import sample
from mne_lsl.lsl import resolve_streams
from mne_lsl.player import PlayerLSL
from mne_lsl.stream import EpochsStream, StreamLSL

THRESHOLD = 2.5 * 1e-11  # used for plotting
EVENT_CHANNEL = "STI 014"


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

    source_id = uuid.uuid4().hex  # get a UUID for the source stream

    # Uses:
    # `from concurrent.futures import ThreadPoolExecutor``
    # to spawn a thread that is streaming data from the file
    player = PlayerLSL(
        raw,
        chunk_size=200,
        name="tutorial-epochs-1",
        source_id=source_id,
        annotations=False,
    ).start()

    return player


def connect_to_lsl_stream(source_id: str) -> StreamLSL:
    """
    Connect to the `tutorial-epochs-1` LSL stream with a data buffer.

    This function sets up a StreamLSL object to connect to an LSL stream named
    `tutorial-epochs-1`. It configures the stream with a buffer size of 4, sets an
    acquisition delay of 0.1 seconds.
    Additionally, it marks a specific channel as bad and applies a low-pass filter
    to the gradient data.
    The `processing_flags` option is set to "all" for automated `clocksync`, `dejitter`, `monotize`, and `threadsafe`.
    See the documentation for more information: https://mne.tools/mne-lsl/stable/generated/api/mne_lsl.lsl.StreamInlet.html#mne_lsl.lsl.StreamInlet

    Returns
    -------
        StreamLSL: An instance of StreamLSL configured to connect to the specified LSL stream.
    """
    stream = StreamLSL(bufsize=4, name="tutorial-epochs-1", source_id=source_id)
    stream.connect(acquisition_delay=0.01, processing_flags="all")
    stream.info["bads"] = ["MEG 2443"]  # remove bad channel
    stream.filter(1, 40, picks="grad")  # filter signal

    return stream


def slice_data_to_epochs(stream: StreamLSL) -> EpochsStream:
    """
    Slice continuous data into epochs using an LSL stream.

    Parameters
    ----------
    stream : StreamLSL
        The LSL stream object containing the continuous data.

    Returns
    -------
    EpochsStream
        An EpochsStream object configured to slice the continuous data into epochs.

    Notes
    -----
    This function sets up an EpochsStream object to slice the continuous data from
    the provided LSL stream into epochs. It configures the stream with a buffer size
    of 20 epochs, sets the event ID to 2, specifies the event channel as "STI 014",
    and defines the epoch time window from -0.2 to 0.5 seconds relative to the event.
    The baseline correction is applied from the start of the epoch to time 0.
    Only gradient channels are picked for the epochs.
    """
    epochs = EpochsStream(
        stream,
        bufsize=20,  # number of epoch held in the buffer
        event_id=2,
        event_channels=EVENT_CHANNEL,
        tmin=-0.2,
        tmax=0.5,
        baseline=(None, 0),
        picks="grad",
    ).connect(acquisition_delay=0.01)

    return epochs


def threshold_decode_epoch(epochs: np.ndarray) -> int:
    """
    Decode the target integer from the given epochs using a threshold-based approach.
    Just check if the max value of the first channel is above a threshold

    Parameters
    ----------
    epochs : np.ndarray
        The epochs data to be decoded. A single epochs is expected with (n_channels, n_samples)

    Returns
    -------
    int
        The decoded target integer.
    """

    val = np.max(epochs[0, :])
    if val > THRESHOLD:
        target_int = 1
    else:
        target_int = 0
    return target_int


def get_forwad_indices(buff_idxs: np.ndarray, curr_i: int) -> np.ndarray:
    """Helper function provides the indices in the ring buffer for going forward"""

    return np.hstack([buff_idxs[curr_i:], buff_idxs[:curr_i]])


def plot_streaming(stream: StreamLSL, epochs: EpochsStream):

    # plot continously
    plt.ion()
    fig, axs = plt.subplots(2, 1, figsize=(20, 16))
    fig.suptitle("Streaming data")

    # draw blank and store
    axs[1].set_ylim(-2.6 * 1e-11, 2.6 * 1e-11)
    axs[0].set_ylim(-3.6 * 1e-11, 3.6 * 1e-11)

    fig.canvas.draw()
    bg1 = fig.canvas.copy_from_bbox(axs[0].bbox)
    bg2 = fig.canvas.copy_from_bbox(axs[1].bbox)

    # set the labels for the plots and axis
    axs[0].set_title("Raw data")
    axs[0].set_xlabel("Samples")
    axs[1].set_xlabel("Time [s]")
    axs[0].set_ylabel("Amplitude [T/m]")  # grad channels
    axs[1].set_ylabel("Amplitude [T/m]")

    # initial fill from buffer
    data_stream, times = stream.get_data()
    t_last = times[-1]

    # the raw plot
    buff_idxs, y = np.arange(data_stream.shape[1]), data_stream[0, :]
    (line,) = axs[0].plot(buff_idxs, y)
    (line_th,) = axs[0].plot(np.ones(len(times)) * THRESHOLD, "-.", color="#aaa")
    (vline,) = axs[0].plot([0, 0], [-3.6 * 1e-11, 3.6 * 1e-11], "--", color="#f93")
    vbox = axs[0].axvspan(0, 0, color="#f93", alpha=0.3)
    idx_pre = int(epochs.times[0] * stream.info["sfreq"])
    idx_post = int(epochs.times[-1] * stream.info["sfreq"])

    y_ring = np.copy(data_stream[0, :])  # initialize with current data
    curr_i = 0  # current head in the ring buffer

    # the epoch plot
    y_curr_epoch = epochs._buffer[-1, :, 0]
    # y_curr_epoch = epochs.get_data()[-1, 0, :]
    (line2,) = axs[1].plot(epochs.times, y_curr_epoch)

    # need to redraw full background -> do only if class changed
    last_class = 0

    t_start = time.time()
    dt = 0.05

    while plt.fignum_exists(fig.number):  # Runs until window is closed

        # - Streaming
        # get new data and append if new data received
        data_stream, times = stream.get_data()
        if times[-1] > t_last:

            tmask = times > t_last
            new_data = data_stream[:, tmask]

            # cut to buffer size
            if new_data.shape[1] > y_ring.shape[0]:
                new_data = new_data[:, -y_ring.shape[0] :]

            # get indices for forward filling
            idxs = get_forwad_indices(buff_idxs, curr_i)
            idxs = idxs[: new_data.shape[1]]

            # check for events and update the last event line if necessary
            (ev_idx,) = np.where(
                new_data[stream.ch_names.index("STI 014") - 1, :]
                == 2  # -1 as one bad channels was selected, usually more carefull selection needed
            )
            if len(ev_idx) > 0:
                if all(np.diff(ev_idx) == 1):
                    last_up = ev_idx[0]
                else:
                    last_up = ev_idx[np.where(np.diff(ev_idx) > 1)[0][-1] + 1]
                vline.set_data([idxs[last_up], idxs[last_up]], list(axs[0].get_ylim()))
                vbox.set_x(max(0, idxs[last_up] + idx_pre))
                vbox.set_width(min(len(y_ring), idx_post - idx_pre))

            # fill buffer
            ch1_data = new_data[0, :]
            y_ring[idxs] = ch1_data

            # update current head
            curr_i = (curr_i + len(ch1_data)) % y_ring.shape[0]
            t_last = times[-1]

            fig.canvas.restore_region(bg1)
            line.set_data(buff_idxs, y_ring)
            axs[0].draw_artist(line)  # ch1 data
            axs[0].draw_artist(line_th)  # threshold horizontal line
            axs[0].draw_artist(vline)  # epo start vertical line
            axs[0].draw_artist(vbox)  # epo range

            fig.canvas.blit(axs[0].bbox)

            # we can only have new epochs if new data in stream
            if epochs.n_new_epochs > 0:
                # for epochs, we always redraw, as this is infrequent anyways

                fig.canvas.restore_region(bg2)
                data_epochs = epochs.get_data()
                epo_class = threshold_decode_epoch(data_epochs[-1, :, :])
                axs[1].set_facecolor("#afa" if epo_class == 1 else "#fff")

                line2.set_data(epochs.times, data_epochs[-1, 0, :])
                axs[1].draw_artist(line2)

                fig.canvas.draw()  # Full redraw

            fig.canvas.flush_events()

        # fix a rough frame rate
        t_pause = max(0.0001, dt - (time.time() - t_start))
        time.sleep(t_pause)
        # print("FPS: ", 1 / (time.time() - t_start))
        t_start = time.time()


def main():

    # This is a mock-up of an online data source providing MEG data to an LSL stream
    player = stream_data_from_fif_file()

    # Connect to the LSL stream -> get data into a ring buffer
    stream = connect_to_lsl_stream(source_id=player.source_id)

    # Slice the continuous data into epochs
    epochs = slice_data_to_epochs(stream)

    # wait until we have an epoch
    while epochs.n_new_epochs < 1:
        time.sleep(0.1)

    data_stream = stream.get_data()
    info_stream = stream.info

    data_epochs = epochs.get_data()
    info_epochs = epochs.info

    print("-" * 80)
    print("Got streaming data with:")
    print("  - shape (n_channels, n_samples):", data_stream[0].shape)
    print("  - info:", info_stream)

    print("-" * 80)
    print("Got epoch data with:")
    print("  - shape (n_epochs, n_channels, n_samples):", data_epochs.shape)
    print("  - info:", info_epochs)

    # Streaming plot until plot is closed
    plot_streaming(stream, epochs)

    # close down
    epochs.disconnect()
    stream.disconnect()
    player.stop()


if __name__ == "__main__":
    main()
