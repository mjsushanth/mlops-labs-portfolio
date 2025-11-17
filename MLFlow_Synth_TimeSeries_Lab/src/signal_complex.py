# Synthetic data creation:

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import seaborn as sns



def generate_complex_weather_series(length=2000, years=4, noise_factor=2, seed=None):
    """complex weather series with strong long-term dependencies.
    Multiple time scales, non-linear interactions, regime changes, and conditional volatility."""
    
    if seed is not None:
        np.random.seed(seed)

    # time axes setup
    days = np.linspace(0, years * 365, length)
    hours = np.linspace(0, years * 365 * 24, length)
    t = np.arange(length)

    ## Base components - annual, daily cycles, trend, basic weather.
    annual_cycle = 40 * np.sin(2 * np.pi * days / 365)  
    daily_cycle = 20 * np.sin(2 * np.pi * hours / 24)   
    
    # Non-linear trend: base trend modified by sine wave
    base_trend = 0.05 * days  # .. from 0.01
    trend_modifier = 1 + 0.7 * np.sin(0.2 * t)  
    trend = base_trend * trend_modifier
    
    # Multi-scale weather systems
    weather_systems = np.zeros_like(days)
    weather_periods = [30, 90, 180, 365]
    for period in weather_periods:
        weather_systems += np.cumsum(np.random.normal(0, 0.3/period, length)) * period/50  # Increased volatility

    ## Complex components - multiple sine waves with different frequencies
    sine_wave1 = 3 * np.sin(0.1 * t)    # .. from 1.82
    sine_wave2 = 4 * np.sin(1.3 * t)    # .. from 2.2
    sine_wave3 = 5 * np.sin(2.5 * t)    # .. from 2
    complex_signal = sine_wave1 + sine_wave2 + sine_wave3

    ## Long-term dependencies with multiple lags
        # Multiple shifted dependencies with decreasing weights for longer lags
    lag_steps = [100, 200, 300, 400]    # Much longer lags
    lag_weights = [0.9, 0.8, 0.7, 0.6]  # Stronger weights

    long_term_deps = np.zeros_like(complex_signal)
    for lag, weight in zip(lag_steps, lag_weights):
        shifted_signal = np.roll(complex_signal, lag)
        shifted_signal[:lag] = 0  # Zero out wrapped values
        long_term_deps += weight * shifted_signal

    ## Non-linear interactions and seasonal memory
    
    # Multiplicative interaction between current and past values
    interaction_lag = 150  # .. from 45
    interaction_signal = 2 * complex_signal * np.roll(complex_signal, interaction_lag)

    interaction_signal[:interaction_lag] = 0

    # Seasonal memory effect (quarter-year lag)
    seasonal_lag = 365 // 2  # Half-year memory instead of quarter
    seasonal_memory = 0.8 * np.roll(annual_cycle, seasonal_lag)  # .. from 0.5
    seasonal_memory[:seasonal_lag] = 0

    ## Combine base components
    series = (annual_cycle +  daily_cycle +  trend + weather_systems +  complex_signal + long_term_deps + interaction_signal + seasonal_memory)

    ## Add regime changes and volatility effects -->  # Regime changes based on historical values
    regime_threshold = np.mean(series)
    regime_lag = 50
    historical_values = np.roll(series, regime_lag)
    regime_effect = np.where(historical_values > regime_threshold, 
                           2.5 * series,    # .. from 1.5
                           0.3 * series)    # from 0.7
    regime_effect[:regime_lag] = series[:regime_lag]
    
    # apply regime effect with smoothing
    series = 0.7 * series + 0.3 * regime_effect


    # Conditional volatility based on past absolute values
    volatility_lag = 250  # .. from 75
    volatility = 1 + 1.5 * np.abs(np.roll(series, volatility_lag))  # .. from 0.5
    volatility[:volatility_lag] = 1
    series *= volatility

    # Add sudden jumps
    jump_points = np.random.choice(length, size=5)
    for point in jump_points:
        series[point:point+50] += np.random.uniform(-30, 30)

    # Add scaled noise -- final noise.
    series += noise_factor * np.random.normal(0, 1, length)
    
    return series, {
        'annual_cycle': annual_cycle,
        'daily_cycle': daily_cycle,
        'trend': trend,
        'weather_systems': weather_systems,
        'complex_signal': complex_signal,
        'long_term_deps': long_term_deps,
        'interaction_signal': interaction_signal,
        'seasonal_memory': seasonal_memory,
        'regime_effect': regime_effect,
        'volatility': volatility
    }

## --------------------------------------------------------------------------------------------------------------------------------------


def compsplots(length=2000, years=4, noise_factor=2):
    """Visualizes all components of the complex weather series"""
    
    # Generate series and get all components
    series, components = generate_complex_weather_series( length=length, years=years, noise_factor=noise_factor )
    
    # Setup time axes
    days = np.linspace(0, years * 365, length)
    hours = np.linspace(0, years * 365 * 24, length)
    t = np.arange(length)
    
    # Create figure with subplots
    fig, axes = plt.subplots(13, 1, figsize=(15, 40))
    fig.suptitle('Complex Weather Series Components Analysis', fontsize=16, y=0.92)
    
    # 1. Base Components
    axes[0].plot(days, components['annual_cycle'])
    axes[0].set_title('Annual Cycle (±20° seasonal variation)')
    axes[0].grid(True)
    
    # 2. Daily Cycle (first 100 points)
    axes[1].plot(hours[:100], components['daily_cycle'][:100])
    axes[1].set_title('Daily Cycle (±5° daily variation)')
    axes[1].grid(True)
    
    # 3. Non-linear Trend
    axes[2].plot(days, components['trend'])
    axes[2].set_title('Non-linear Warming Trend')
    axes[2].grid(True)
    
    # 4. Multi-scale Weather Systems
    axes[3].plot(days, components['weather_systems'])
    axes[3].set_title('Multi-scale Weather Systems')
    axes[3].grid(True)
    
    # 5. Complex Signal (Combined Sine Waves)
    axes[4].plot(t, components['complex_signal'])
    axes[4].set_title('Complex Signal (Multiple Frequencies)')
    axes[4].grid(True)
    
    # 6. Long-term Dependencies
    axes[5].plot(t, components['long_term_deps'])
    axes[5].set_title('Long-term Dependencies (Multiple Lags)')
    axes[5].grid(True)
    
    # 7. Non-linear Interactions
    axes[6].plot(t, components['interaction_signal'])
    axes[6].set_title('Non-linear Past-Present Interactions')
    axes[6].grid(True)
    
    # 8. Seasonal Memory Effect
    axes[7].plot(t, components['seasonal_memory'])
    axes[7].set_title('Seasonal Memory Effect')
    axes[7].grid(True)
    
    # 9. Regime Effects
    regime_impact = components['regime_effect'] - series
    axes[8].plot(t, regime_impact)
    axes[8].set_title('Regime Change Impact')
    axes[8].grid(True)
    
    # 10. Conditional Volatility
    axes[9].plot(t, components['volatility'])
    axes[9].set_title('Conditional Volatility')
    axes[9].grid(True)
    
    # 11. Clean Combined Signal
    clean_signal = (components['annual_cycle'] + components['daily_cycle'] + components['trend'] + components['weather_systems'] + 
                   components['complex_signal'] + components['long_term_deps'] + components['interaction_signal'] + components['seasonal_memory'])
    axes[10].plot(t, clean_signal)
    axes[10].set_title('Combined Signal (Before Regime/Volatility Effects)')
    axes[10].grid(True)
    
    # 12. Noise Component (first 200 points)
    noise = noise_factor * np.random.normal(0, 1, length)
    axes[11].plot(t[:200], noise[:200])
    axes[11].set_title('Random Noise Component')
    axes[11].grid(True)
    
    # 13. Final Combined Signal
    axes[12].plot(t, series)
    axes[12].set_title('Final Combined Signal')
    axes[12].grid(True)
    
    # Add analysis plots
    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))
    fig2.suptitle('Signal Analysis', fontsize=13)
    
    # Autocorrelation plot
    from statsmodels.tsa.stattools import acf
    
    # Distribution of values
    axes2[0].hist(series, bins=50, density=True)
    axes2[0].set_title('Value Distribution')
    axes2[0].grid(True)
    
    # Lag Plot (t vs t-30)
    lag = 30
    axes2[1].scatter(series[:-lag], series[lag:], alpha=0.1)
    axes2[1].set_title(f'Lag Plot (t vs t-{lag})')
    axes2[1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print component ranges for comparison
    print("\nComponent Ranges:")
    for name, component in components.items():
       print(f"{name}: {np.min(component):.2f} to {np.max(component):.2f}")
    print(f"Final Series: {np.min(series):.2f} to {np.max(series):.2f}")
    

compsplots()