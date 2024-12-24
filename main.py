import numpy as np
import time
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def data_stream_simulator():
    """
    Generates a simulated real-time data stream with regular patterns, seasonal variations, and random noise.
    
    Yields:
        float: The next value in the simulated data stream.
    """
    t = 0
    while True:
        try:
            # Generate components of the signal
            base = 10 * np.sin(2 * np.pi * t / 50)  # regular pattern
            seasonality = 5 * np.sin(2 * np.pi * t / 200)  # seasonal component
            noise = np.random.normal(0, 1)  # random noise
            yield base + seasonality + noise
            t += 1
            time.sleep(0.1)  # simulate real-time streaming by adding delay
        except Exception as e:
            print(f"Error in data stream simulation: {e}")
            break

def detect_anomalies(data_stream, window_size=50, threshold=3):
    """
    Detects anomalies in a data stream using a sliding window approach.
    
    Args:
        data_stream (generator): The real-time data stream generator.
        window_size (int): The size of the sliding window to calculate statistics.
        threshold (float): Z-score threshold for anomaly detection.
        
    Yields:
        tuple: Current value in the data stream and list of detected anomalies.
    """
    data_window = deque(maxlen=window_size)  # Sliding window to keep recent values
    anomalies = []

    for value in data_stream:
        try:
            # Append value to the sliding window
            data_window.append(value)
            if len(data_window) == window_size:
                # Calculate mean and standard deviation for the window
                mean = np.mean(data_window)
                std = np.std(data_window)

                # Calculate z-score if std deviation is non-zero
                z_score = (value - mean) / std if std > 0 else 0
                if abs(z_score) > threshold:
                    anomalies.append(value)
                    print(f"Anomaly detected: {value}")

            # Yield the current value and any detected anomalies
            yield value, anomalies
        except ZeroDivisionError:
            print("Standard deviation is zero; cannot calculate z-score.")
            continue
        except Exception as e:
            print(f"Error in anomaly detection: {e}")
            continue

def live_plot(data_stream):
    """
    Plots the data stream in real-time and highlights detected anomalies.
    
    Args:
        data_stream (generator): The generator yielding data values and detected anomalies.
    """
    fig, ax = plt.subplots()
    data, anomalies = [], []
    x_data, y_data = [], []

    def update(frame):
        try:
            # Get the next value and anomaly list from the data stream
            value, anomaly_list = next(data_stream)
            data.append(value)
            x_data.append(len(data))
            y_data.append(value)

            # Check for anomalies and add to list for plotting
            if value in anomaly_list:
                anomalies.append((len(data), value))

            # Clear and plot the data stream and anomalies
            ax.clear()
            ax.plot(x_data, y_data, label='Data Stream')
            if anomalies:
                ax.scatter(*zip(*anomalies), color='red', label='Anomalies')
            ax.legend()
            ax.set_title("Real-Time Data Stream with Anomaly Detection")
            ax.set_xlabel("Data Point")
            ax.set_ylabel("Value")
        except StopIteration:
            print("Data stream ended.")
            plt.close(fig)
        except Exception as e:
            print(f"Error in plotting: {e}")

    ani = FuncAnimation(fig, update, interval=100)
    plt.show()

if __name__ == "__main__":
    # Create the data stream and pass it through the anomaly detector
    data_stream = detect_anomalies(data_stream_simulator())
    # Start the live plot to visualize the stream and anomalies
    live_plot(data_stream)
