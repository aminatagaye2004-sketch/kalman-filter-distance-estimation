# Kalman Filter for LiDAR Distance Estimation

## Overview

This project investigates the application of a **one-dimensional Kalman filter** to improve distance measurements obtained from a VL53L0X time-of-flight (ToF) sensor.

Using experimental data collected during an autonomous rover project, the performance of the Kalman filter is compared against raw sensor measurements and a three-sample moving average.

Two experiments are evaluated:

1. **Static Distance Experiment:** Evaluates filtering performance at four known distances: 200, 500, 800 and 1000 mm.
2. **Angular Velocity Experiment:** Investigates how servo-driven sensor rotation affects distance estimation at nominal angular velocities of 33.3, 100, 300 and 800°/s.

The project examines both the benefits and limitations of Kalman filtering, particularly its ability to reduce random measurement noise while retaining systematic measurement bias.

## Technologies Used

- Python
- NumPy — numerical calculations
- Pandas — experimental data processing
- Matplotlib — data visualisation
- Kalman filtering — recursive state estimation

## Methodology

### Kalman Filter Implementation

A scalar Kalman filter was implemented using a constant-distance state model.

**Prediction:**

$$
\hat{x}_k^- = \hat{x}_{k-1}
$$

$$
P_k^- = P_{k-1} + Q
$$

**Kalman gain:**

$$
K_k = \frac{P_k^-}{P_k^- + R}
$$

**Measurement update:**

$$
\hat{x}_k = \hat{x}_k^- + K_k(z_k - \hat{x}_k^-)
$$

**Covariance update:**

$$
P_k = (1-K_k)P_k^-
$$

**Where:**

- $\hat{x}_k$: Estimated distance
- $z_k$: Sensor measurement
- $P_k$: Estimation uncertainty
- $Q$: Process noise variance
- $R$: Measurement noise variance
- $K_k$: Kalman gain

The process noise variance was set to $Q = 1\ \mathrm{mm}^2$.
Measurement noise variance $R$ was estimated from calibration data for each experimental condition.




### Experiment 1: Static Distance Measurements

The VL53L0X sensor measured stationary targets at four known distances.

For each distance, three trials containing 20 measurements each were collected.

- Trial 1 was used to estimate measurement noise variance.
- Trials 2 and 3 were used for filter evaluation.
- Kalman filtering was compared against raw measurements and a causal three-sample moving average.

Performance was assessed using root mean square error (RMSE), mean bias and standard deviation.

### Experiment 2: Angular Velocity Measurements

The VL53L0X sensor was mounted on a servo motor rotating between 60° and 120°.

Measurements were recorded at the nominal 90° position, with the target stationary at 500 mm.

Four nominal angular velocities were evaluated: 33.3, 100, 300 and 800°/s.

Each condition included three trials of 20 measurements. Trial 1 was used for measurement noise estimation, while Trials 2 and 3 were used for evaluation.

The same Kalman filter implementation was applied to both experiments.

## Results

### Experiment 1: Static Distance

| Distance | Raw RMSE | Kalman RMSE | RMSE Reduction |
|---|---:|---:|---:|
| 200 mm | 2.474 mm | 1.447 mm | 41.5% |
| 500 mm | 7.315 mm | 5.565 mm | 23.9% |
| 800 mm | 6.217 mm | 2.612 mm | 58.0% |
| 1000 mm | 10.125 mm | 4.138 mm | 59.1% |

The Kalman filter reduced RMSE across all evaluated distances and outperformed the moving average in all eight evaluation trials.

### Experiment 2: Angular Velocity

| Angular Velocity | Raw RMSE | Kalman RMSE | RMSE Reduction |
|---|---:|---:|---:|
| 33.3°/s | 2.718 mm | 1.536 mm | 43.5% |
| 100°/s | 2.769 mm | 1.552 mm | 43.9% |
| 300°/s | 3.782 mm | 2.946 mm | 22.1% |
| 800°/s | 5.494 mm | 4.900 mm | 10.8% |

Filtering improved distance estimation at every tested angular velocity, but the relative improvement decreased at higher speeds.

The average Kalman-filtered bias increased from approximately +0.38 mm at 33.3°/s to +4.82 mm at 800°/s.

This demonstrates that reducing measurement fluctuations does not necessarily eliminate systematic estimation errors.

## Key Findings

- Kalman filtering reduced RMSE in all 16 evaluation trials across both experiments.
- The Kalman filter outperformed the three-sample moving average under the tested conditions.
- At higher angular velocities, systematic bias became a more significant component of the total estimation error.
- A constant-distance Kalman filter can improve measurement precision but does not inherently correct speed-dependent sensor bias.

## Limitations

The investigation uses a relatively small experimental dataset, with 20 measurements per trial.

The process noise variance was fixed at \(Q=1\text{ mm}^2\), and the moving-average window was fixed at three samples. Neither parameter was systematically optimised.

Angular velocities were based on programmed servo motion rather than independently measured sensor velocity.

Additionally, the constant-distance state model does not explicitly represent sensor rotation or potential angular-velocity-dependent measurement bias.

The calibration and evaluation trials were separated, but no additional untouched dataset was reserved for final validation.

## Future Improvements

- Investigate systematic bias compensation as a function of angular velocity.
- Evaluate different process noise values and moving-average window sizes.
- Collect additional measurements across repeated experimental sessions.
- Investigate adaptive Kalman filtering to account for changing measurement noise.
- Extend the state-estimation model to incorporate motion when appropriate.

## Running the Project

Clone the repository and install the required Python libraries:

```bash
pip install numpy pandas matplotlib
```

Place `kalman_filter.py` and `rover(Sheet1).csv` in the same directory.

Run:

```bash
python kalman_filter.py
```

The script generates comparison plots, experiment summaries and CSV files containing the calculated performance metrics.

## Project Background

The experimental data originated from an autonomous rocker-bogie rover engineering project involving sensor integration and distance measurement testing.

This follow-up project focuses specifically on applying and evaluating recursive estimation techniques using the previously collected sensor measurements.# kalman-filter-distance-estimation
