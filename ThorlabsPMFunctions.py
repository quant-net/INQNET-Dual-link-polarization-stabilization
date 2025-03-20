import pyvisa
import usbtmc
import time
from ThorlabsPM100 import ThorlabsPM100

class PowerMeter:
    def __init__(self, resource_name_pyvisa='USB0::0x1313::0x8076::M00927710::INSTR', 
                 wavelength=1536, use_pyvisa=True):
        """Initialize the Thorlabs Power Meter, either with pyvisa or usbtmc."""
        self.use_pyvisa = use_pyvisa

        if self.use_pyvisa:
            # Initialize with PyVISA
            self.rm = pyvisa.ResourceManager()
            self.device = self.rm.open_resource(resource_name_pyvisa)
            self.power_meter = ThorlabsPM100(inst=self.device)
            
            # Automatically extract vendor and product IDs if needed for USBTMC fallback
            self.vendor_id, self.product_id = self._extract_ids_from_pyvisa_resource(resource_name_pyvisa)
        else:
            # Initialize with USBTMC using extracted vendor and product IDs
            self.device = usbtmc.Instrument(self.vendor_id, self.product_id)
            self.power_meter = ThorlabsPM100(inst=self.device)
        
        # Set the wavelength
        self.set_wavelength(wavelength)
        time.sleep(1)  # Allow time for the wavelength setting to apply

        # Confirm connection
        self.confirm_connection()

    def _extract_ids_from_pyvisa_resource(self, resource_name):
        """Extract the vendor and product IDs from the PyVISA resource name."""
        # PyVISA resource name format: 'USB0::0x1313::0x8076::M00927710::INSTR'
        try:
            parts = resource_name.split('::')
            vendor_id = int(parts[1], 16)  # Convert hex to int
            product_id = int(parts[2], 16)  # Convert hex to int
            return vendor_id, product_id
        except (IndexError, ValueError):
            print(f"Error extracting vendor/product IDs from {resource_name}")
            return None, None

    def set_wavelength(self, wavelength):
        """Set the wavelength for the power meter using the appropriate backend."""
        print(f"Setting wavelength to {wavelength} nm.")
        self._send_command(f"SENS:CORR:WAV {wavelength}NM")

    def confirm_connection(self):
        """Query the instrument identity to confirm the connection."""
        response = self._query_command("*IDN?")
        print(f"Connected to: {response}")

    def measure_power(self, N=1, delay=0.1):
        """Measure the power N times, with retries, and return the list of power values."""
        powers = []
        self.max_retries = 5
        for i in range(N):
            retries = 0
            while retries < self.max_retries:
                try:
                    power = self._query_command("MEAS:POW?")
                    powers.append(float(power))
                    # time.sleep(delay)  # Delay between each measurement
                    break  # Break the retry loop if successful
                except Exception as e:
                    retries += 1
                    print(f"Error reading power (attempt {retries}/{self.max_retries}): {e}")
                    if retries == self.max_retries:
                        print("Max retries reached. Continuing with next measurement...")
        return powers

    def _send_command(self, command):
        """Send a command to the power meter, adjusting for the correct backend."""
        if self.use_pyvisa:
            self.device.write(command)
        else:
            self.device.write(command)
    
    def _query_command(self, command):
        """Send a query command to the power meter and return the response, adjusting for the correct backend."""
        if self.use_pyvisa:
            return self.device.query(command)
        else:
            return self.device.ask(command)

# Example usage
if __name__ == "__main__":
    # Initialize the Thorlabs Power Meter with pyvisa or usbtmc depending on the use_pyvisa flag
    use_pyvisa = True  # Set to True to use pyvisa, or False to use usbtmc

    if use_pyvisa:
        power_meter = PowerMeter(resource_name_pyvisa='USB0::0x1313::0x8076::M00927710::INSTR', wavelength=1536, use_pyvisa=True)
    else:
        power_meter = PowerMeter(wavelength=1536, use_pyvisa=False)

    # Measure the power
    measured_power = power_meter.measure_power(N=10)
    print(f"Measured Power Samples: {measured_power}")
