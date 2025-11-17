# signals.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class SignalConfig:
    """Configuration for the synthetic weather signal.

    length: number of time steps in the series
    years: span in 'years' used to scale seasonal effects
    noise_std: standard deviation of additive white noise
    seed: RNG seed for reproducibility
    include_components: if False, only (t, temp) are returned
    """

    length: int = 2000
    years: float = 4.0
    noise_std: float = 2.0
    seed: Optional[int] = 42
    include_components: bool = True


class SyntheticWeatherSignal:
    """Generate a simplified synthetic weather-like temperature series.

    Components:
      - annual seasonal cycle (slow sine over 'years')
      - higher-frequency cycle (daily-ish pattern)
      - linear warming trend
      - AR(1)-style stochastic 'weather system' component
      - Gaussian noise

    The final signal is roughly: temp = annual + daily + trend + ar_weather + noise
    """

    def __init__(self, config: SignalConfig | None = None) -> None:
        self.config = config or SignalConfig()
        self.rng = np.random.default_rng(self.config.seed)

    def generate(self) -> pd.DataFrame:
        cfg = self.config

        # Time axes
        t = np.arange(cfg.length, dtype=float)
        days = np.linspace(0.0, cfg.years * 365.0, cfg.length)

        # Annual cycle: ±10 degrees over each year
        annual = 10.0 * np.sin(2.0 * np.pi * days / 365.0)

        # Higher-frequency cycle: approximate daily oscillation
        # We treat each step as one "hour" here for simplicity.
        daily = 5.0 * np.sin(2.0 * np.pi * t / 24.0)

        # Slow warming trend, roughly a few degrees over the full span
        trend = 0.01 * days  # ~0.01 degree per day

        # Low-frequency stochastic component with memory: AR(1)-like
        ar_weather = np.zeros(cfg.length, dtype=float)
        for i in range(1, cfg.length):
            ar_weather[i] = 0.9 * ar_weather[i - 1] + self.rng.normal(0.0, 0.5)

        # White noise
        noise = self.rng.normal(0.0, cfg.noise_std, size=cfg.length)

        # Final temperature signal
        temp = annual + daily + trend + ar_weather + noise

        data = {
            "t": t,
            "day": days,
            "temp": temp,
            "annual": annual,
            "daily": daily,
            "trend": trend,
            "ar_weather": ar_weather,
            "noise": noise,
        }
        df = pd.DataFrame(data)

        if not cfg.include_components:
            df = df[["t", "temp"]]

        return df


if __name__ == "__main__":

    generator = SyntheticWeatherSignal()
    df = generator.generate()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["t"], df["temp"], linewidth=0.8)
    ax.set_title("Synthetic Weather Temperature Signal")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Temperature (arbitrary units)")
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    print(df.describe().T[["mean", "std", "min", "max"]])
