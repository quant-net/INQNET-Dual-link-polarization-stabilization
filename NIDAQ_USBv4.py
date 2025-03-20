import nidaqmx
from nidaqmx.stream_readers import AnalogMultiChannelReader
from nidaqmx.constants import TerminalConfiguration, AcquisitionType
import numpy as np
import time
from datetime import datetime
import os


class NIDAQ_USB:
    def __init__(self, devID=None, ai_samples=1000, ai_rate=25e3, ai_channels=4, incr=1,
                 ao_samples=2, ao_rate=None, ao_channels=1, min_v=0, max_v=1.5, scale=None):
        """
        Initializes NIDAQ USB device for Analog Input (AI) and Analog Output (AO).

        Arguments:
        devID: Device identifier (e.g., 'Dev1').
        ai_samples: Total number of AI samples per collection run.
        ai_rate: AI sampling rate.
        ai_channels: Number of AI channels.
        incr: Number of AI samples per increment (default: 1).

        Optional:
        ao_samples: Total number of AO samples (default: None).
        ao_rate: AO sampling rate (default: None).
        ao_channels: Number of AO channels (default: 1).
        min_v, max_v: Voltage range for AO channels.

        Changes:
        Analog Input can be measured both synchronously and asynchronously
        Measured AI data no longer contains timestamps
        """
        self.devID = devID
        self.ai_samples = int(ai_samples)
        self.ai_rate = int(ai_rate)
        self.ai_channels = ai_channels
        self.incr = incr

        self.ao_samples = int(ao_samples)
        self.ao_rate = ao_rate
        self.ao_channels = ao_channels
        self.min_v = min_v
        self.max_v = max_v

        self.ai_task = None
        self.ao_task = None
        self.scale = scale

        self.data = {
            "AI Data": np.zeros((self.ai_channels + 1, self.ai_samples)),
            "AO Data": np.zeros((self.ao_channels + 1, self.ao_samples if self.ao_samples else 0))
        }

        self.ai_idx = 0
        self.callback_counter = 0

    def __str__(self):
        return (f"<NIDAQ_USB: {self.devID}>\n"
                f"AI - Samples: {self.ai_samples}, Rate: {self.ai_rate}, Channels: {self.ai_channels}\n"
                f"AO - Samples: {self.ao_samples}, Rate: {self.ao_rate}, Channels: {self.ao_channels}")

    # Setup Functions
    def setup_ai_task(self, acquisition_type=AcquisitionType.FINITE):
        """Setup Analog Input task."""
        self.ai_task = nidaqmx.Task()
        self.ai_task.ai_channels.add_ai_voltage_chan(
            f"{self.devID}/ai0:{self.ai_channels - 1}",
            terminal_config=TerminalConfiguration.RSE
        )
        self.ai_task.timing.cfg_samp_clk_timing(
            rate=self.ai_rate,
            sample_mode=acquisition_type,
            samps_per_chan=self.ai_samples
        )

    def setup_ao_task(self):
        """Setup Analog Output task."""
        if not self.ao_samples or not self.ao_rate:
            raise ValueError("AO samples and rate must be defined for AO task.")

        self.ao_task = nidaqmx.Task()
        self.ao_task.ao_channels.add_ao_voltage_chan(
            f"{self.devID}/ao0:{self.ao_channels - 1}",
            min_val=self.min_v, max_val=self.max_v
        )

    # Task Control Functions
    def start_task(self, task):
        if task:
            task.start()

    def stop_task(self, task):
        if task:
            task.stop()

    def close_task(self, task):
        if task:
            task.close()

    def close_all_tasks(self):
        try:
            self.close_task(self.ai_task)
            self.close_task(self.ao_task)
        except Exception as error:
            print("Could not close one or more tasks")

    # Synchronous AI Reading
    def ai_measurement(self, samples):
        """Perform AI measurement for a given number of samples."""
        if not self.ai_task:
            raise RuntimeError("AI task not set up. Call setup_ai_task() first.")
        if samples > self.ai_samples:
            raise ValueError("Number of samples exceeds the total AI samples configured.")

        # If the buffer is full, reset it to hold fresh data
        if self.ai_idx + samples > self.ai_samples:
            # print("AI buffer full. Resetting buffer for fresh data.")
            self.data["AI Data"] = np.zeros((self.ai_channels + 1, self.ai_samples))
            self.ai_idx = 0  # Reset the data index

        # Prepare for data acquisition
        data_values = np.zeros((self.ai_channels + 1, samples))
        ai_start = time.perf_counter()

        # Perform measurement
        self.start_task(self.ai_task)
        data_read = self.ai_task.read(number_of_samples_per_channel=samples)
        self.stop_task(self.ai_task)

        # Store timestamps and data
        data_values[1:, :] = np.array(data_read)
        data_values[0, :] = time.perf_counter() - ai_start

        # Store in buffer
        end_idx = self.ai_idx + samples
        self.data["AI Data"][:, self.ai_idx:end_idx] = data_values
        self.ai_idx = end_idx

        return data_values

    # Asynchronous AI Reading
    def ai_measurement_async(self, duration_sec, buffer_size, callback_interval):
        """
        Perform asynchronous AI measurement using AnalogMultiChannelReader.

        Arguments:
        duration_sec: Duration of the measurement in seconds.
        buffer_size: Size of the data buffer for each callback.
        callback_interval: Number of samples per channel per callback.
        """
        if not self.ai_task:
            raise RuntimeError("AI task not set up. Call setup_ai_task() first.")

        # Configure the task for continuous sampling
        self.ai_task.timing.cfg_samp_clk_timing(
            rate=self.ai_rate,
            sample_mode=AcquisitionType.CONTINUOUS
        )

        self.callback_counter = 0
        reader = AnalogMultiChannelReader(self.ai_task.in_stream)

        # Ensure the buffer matches the callback interval
        data_buffer = np.zeros((self.ai_channels+1, callback_interval), dtype=np.float64)
        ai_async_start = time.perf_counter()

        def callback(task_handle, event_type, num_samples, callback_data):
            nonlocal data_buffer
            if num_samples != callback_interval:
                raise RuntimeError(f"Unexpected number of samples: {num_samples}")

            # Adjust the buffer to read the exact number of samples
            reader.read_many_sample(data_buffer[1:,:], num_samples)
            self.callback_counter += 1

            # # Log data statistics (optional)
            # print("-" * 80)
            # print(f"Callback #{self.callback_counter}")
            # print(f"Mean: {np.mean(data_buffer, axis=1)}")
            # print(f"Min: {np.min(data_buffer, axis=1)}")
            # print(f"Max: {np.max(data_buffer, axis=1)}")
            # print("-" * 80)
            return 0  # Indicate success

        # Register the callback
        self.ai_task.register_every_n_samples_acquired_into_buffer_event(callback_interval, callback)

        # Start the task and run for the specified duration
        self.start_task(self.ai_task)
        time.sleep(duration_sec)
        data_buffer[0,:] = time.perf_counter() - ai_async_start # Maybe better to do timing for each?
        
        return data_buffer

    # AO Output
    def ao_output(self, data):
        """Send data to AO channels."""
        if not self.ao_task:
            raise RuntimeError("AO task not set up. Call setup_ao_task() first.")
        if len(data) != self.ao_samples:
            raise ValueError("AO data size must match ao_samples.")

        self.ao_task.write(data, auto_start=False)
        self.start_task(self.ao_task)
        self.stop_task(self.ao_task)

    # Data Handling Functions
    def save_data(self, filename):
        """Save AI data to a file."""
        date_str = datetime.now().strftime('%Y%m%d')
        folder_path = os.path.join(os.path.dirname(__file__), "NIDAQ Data " + date_str)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        file_path = os.path.join(folder_path, filename)
        np.savez(file_path, **self.data)
        print(f"Data saved to {file_path}")

    def load_data(self, filename):
        """Load data from a file."""
        self.data = np.load(filename)
        print(f"Data loaded from {filename}")


# Example usage (Replace with actual device parameters)
if __name__ == "__main__":
    TPM = 0.25# time for each data acquisition measurement of 1000 samples (rounded up from 0.248s)

    duration = 1
    ai_samples = int(duration * (1000/TPM))
    print(f"Number of samples: {ai_samples}")

    ai_rate = 25000
    device = NIDAQ_USB(devID="Dev1", ai_samples=ai_samples, ai_rate=ai_rate, ai_channels=4)

    print(device)

    try:
        # Synchronous Measurement
        tic = time.perf_counter()
        device.setup_ai_task()
        ai_data_sync = device.ai_measurement(ai_samples)
        time.sleep(duration)
        print(f"Synchronous Data:\n{ai_data_sync}")
        # device.close_task(device.ai_task)
        toc = time.perf_counter()
        print(f"Time elapsed is {toc - tic} seconds")
        device.save_data("Sync Measurement")

        # # Asynchronous Measurement
        # device.setup_ai_task()
        # ai_data_async = device.ai_measurement_async(duration_sec=1, buffer_size=1000, callback_interval=1000)
        # print(f"Asynchronous Data:\n{ai_data_async}")
        # device.save_data("Async Measurement")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        device.close_all_tasks()
