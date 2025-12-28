# power_monitor.py
import time
import threading
from typing import List, Tuple

try:
    import pynvml
except ImportError:
    pynvml = None


class PowerMonitor:
    """
    Simple NVML-based GPU power logger.
    Samples power in Watts at fixed interval (ms) in a background thread.
    """

    def __init__(self, device_index: int = 0, interval_ms: float = 10.0):
        self.device_index = device_index
        self.interval_ms = interval_ms
        self._samples: List[Tuple[float, float]] = []
        self._thread = None
        self._stop_flag = threading.Event()

    def start(self):
        if pynvml is None:
            raise RuntimeError("pynvml not installed. Run: pip install nvidia-ml-py3")

        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        self._samples.clear()
        self._stop_flag.clear()

        def _worker():
            start_t = time.time()
            interval_s = self.interval_ms / 1000.0
            while not self._stop_flag.is_set():
                now = time.time()
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self._handle)  # milliwatts
                power_w = power_mw / 1000.0
                self._samples.append((now - start_t, power_w))
                time.sleep(interval_s)

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread is None:
            return
        self._stop_flag.set()
        self._thread.join()
        pynvml.nvmlShutdown()

    @property
    def samples(self) -> List[Tuple[float, float]]:
        """
        Returns list of (time_sec_since_start, power_watts).
        """
        return list(self._samples)

    def summary(self):
        if not self._samples:
            return None
        times, powers = zip(*self._samples)
        avg_power = sum(powers) / len(powers)
        max_power = max(powers)
        duration = max(times) - min(times)
        energy_j = sum(powers) * (self.interval_ms / 1000.0)  # rough Riemann sum

        return {
            "avg_power_w": avg_power,
            "max_power_w": max_power,
            "duration_s": duration,
            "energy_j": energy_j,
        }
