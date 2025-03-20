import numpy as np
import matplotlib.pyplot as plt

def load_and_plot_BSM_tracking(npz_file):
    # Load the data from the .npz file
    data = np.load(npz_file)
    BSM_tracking_data = data['BSM_tracking_data']
    
    # Extracting columns
    time = BSM_tracking_data[:, 0]
    psi_plus = BSM_tracking_data[:, 1]
    psi_minus = BSM_tracking_data[:, 2]
    
    # Plot the data
    plt.figure(figsize=(8, 5))
    plt.scatter(time, psi_plus, label='psi+', color='blue', alpha=0.7)
    plt.scatter(time, psi_minus, label='psi-', color='red', alpha=0.7)
    
    plt.xlabel("Time (s)")
    plt.ylabel("Coin Counts Rate (1/s)")
    plt.title("BSM Tracking Data")
    plt.legend()
    plt.grid(True)
    plt.show()


def load_and_plot_pol_tracking(npz_file):

    colors = ["blue", "red", "green", "black", "navy", "orange"]
    
    # Load the data from the .npz file
    data = np.load(npz_file)
    cleaned_tracking_data = data['raw_tracking_data']
    print(cleaned_tracking_data)
    
    if True:
        fig2, ax2 = plt.subplots()
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Error")
        ax2.set_title("Stabilization Tracking Plot")
        
        ax2.scatter(cleaned_tracking_data[0, :], cleaned_tracking_data[1, :], label="Bob H1", color=colors[0])
        ax2.scatter(cleaned_tracking_data[0, :], cleaned_tracking_data[2, :], label="Bob D2", color=colors[1])
        ax2.scatter(cleaned_tracking_data[0, :], cleaned_tracking_data[3, :], label="Bob H2", color=colors[2])
        ax2.scatter(cleaned_tracking_data[0, :], cleaned_tracking_data[4, :], label="Alice H1", color=colors[3])
        ax2.scatter(cleaned_tracking_data[0, :], cleaned_tracking_data[5, :], label="Alice D2", color=colors[4])
        ax2.scatter(cleaned_tracking_data[0, :], cleaned_tracking_data[6, :], label="Alice H2", color=colors[5])
        
        ax2.set_ylim(-0.01, 0.1)
        ax2.legend()
        plt.show()




# Example usage
path = "C:/Users/QUANT-NET Admin/Documents/Python codes/QUANTNET/INQNET_DLPolarizationStabilization_Main/"
date='2025-03-18'
runner_idx='13'
npz_file = f"PSO Data/{date}_BSM_TrackingData_{runner_idx}.npz"  # Replace with the actual path
load_and_plot_BSM_tracking(npz_file)

npz_file = f"PSO Data/{date}_Polarization_Stabilization_TrackingData_{runner_idx}.npz"  # Replace with the actual path
load_and_plot_pol_tracking(npz_file)