import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. LOAD EXPERIMENTAL DATA
# ==========================================

file_path = "rover(Sheet1).csv"
data = pd.read_csv(file_path, header=None)

# Static experiment: 12 rows (4 distances x 3 trials)
static_data = data.iloc[1:13, :].copy()


def get_static_trial(distance, trial):
    """Return 20 measurements for one static trial."""

    rows = static_data[
        (static_data.iloc[:, 0].astype(float) == distance) &
        (static_data.iloc[:, 1].astype(int) == trial)
    ]

    if len(rows) != 1:
        raise ValueError(
            f"Expected one trial for {distance} mm, Trial {trial}"
        )

    return rows.iloc[0, 2:22].astype(float).to_numpy()


# ==========================================
# 2. SCALAR KALMAN FILTER
# ==========================================

def kalman_filter(measurements, Q, R):
    """
    Estimate a constant distance using a 1D Kalman filter.

    Q: Process noise variance (mm^2)
    R: Measurement noise variance (mm^2)
    """

    x = float(measurements[0])
    P = float(R)

    estimates = [x]
    gains = [np.nan]

    for z in measurements[1:]:

        # Prediction
        x_pred = x
        P_pred = P + Q

        # Kalman gain
        K = P_pred / (P_pred + R)

        # Measurement update
        x = x_pred + K * (z - x_pred)

        # Uncertainty update
        P = (1 - K) * P_pred

        estimates.append(x)
        gains.append(K)

    return np.array(estimates), np.array(gains)


# ==========================================
# 3. MOVING AVERAGE FILTER
# ==========================================

def moving_average(measurements, window=3):
    """Causal moving average of recent measurements."""

    estimates = []

    for i in range(len(measurements)):

        start = max(0, i - window + 1)
        recent = measurements[start:i + 1]

        estimates.append(np.mean(recent))

    return np.array(estimates)


# ==========================================
# 4. PERFORMANCE METRICS
# ==========================================

def calculate_metrics(estimates, true_distance):
    """Calculate RMSE, bias and standard deviation."""

    errors = estimates - true_distance

    rmse = np.sqrt(np.mean(errors ** 2))
    bias = np.mean(errors)
    std = np.std(estimates, ddof=1)

    return rmse, bias, std


# ==========================================
# 5. EVALUATE STATIC EXPERIMENT
# ==========================================

distances = [200, 500, 800, 1000]

Q = 1.0
window = 3

results = []

for true_distance in distances:

    # Trial 1: Calibration
    calibration = get_static_trial(true_distance, 1)
    R = np.var(calibration, ddof=1)

    print(f"\n========== {true_distance} mm ==========")
    print(f"Calibration R: {R:.2f} mm²")

    for trial in [2, 3]:

        # Independent evaluation data
        measurements = get_static_trial(true_distance, trial)

        # Apply filters
        filtered, gains = kalman_filter(measurements, Q, R)

        ma_filtered = moving_average(
            measurements, window=window
        )

        # Calculate performance metrics
        raw_rmse, raw_bias, raw_std = calculate_metrics(
            measurements, true_distance
        )

        kalman_rmse, kalman_bias, kalman_std = calculate_metrics(
            filtered, true_distance
        )

        ma_rmse, ma_bias, ma_std = calculate_metrics(
            ma_filtered, true_distance
        )

        # RMSE improvement
        improvement = (
            (raw_rmse - kalman_rmse) / raw_rmse
        ) * 100 if raw_rmse != 0 else 0.0

        # Store results
        results.append({
            "Distance": true_distance,
            "Trial": trial,
            "R": R,
            "Raw RMSE": raw_rmse,
            "Kalman RMSE": kalman_rmse,
            "MA RMSE": ma_rmse,
            "Raw Bias": raw_bias,
            "Kalman Bias": kalman_bias,
            "MA Bias": ma_bias,
            "Raw Std": raw_std,
            "Kalman Std": kalman_std,
            "MA Std": ma_std,
            "Improvement (%)": improvement
        })

        # Print results
        print(f"\nTrial {trial}")

        print(f"Raw RMSE: {raw_rmse:.3f} mm")
        print(f"Kalman RMSE: {kalman_rmse:.3f} mm")
        print(f"Moving Average RMSE: {ma_rmse:.3f} mm")

        print(f"Raw bias: {raw_bias:.3f} mm")
        print(f"Kalman bias: {kalman_bias:.3f} mm")
        print(f"Moving Average bias: {ma_bias:.3f} mm")

        print(f"Raw std: {raw_std:.3f} mm")
        print(f"Kalman std: {kalman_std:.3f} mm")
        print(f"Moving Average std: {ma_std:.3f} mm")

        print(f"RMSE reduction: {improvement:.1f}%")

        # ==========================================
        # 6. PLOT FILTER COMPARISON
        # ==========================================

        samples = np.arange(1, len(measurements) + 1)

        plt.figure(figsize=(10, 5))

        plt.plot(
            samples, measurements, "o--",
            label="Raw sensor measurements"
        )

        plt.plot(
            samples, filtered, "o-",
            label="Kalman estimate"
        )

        plt.plot(
            samples, ma_filtered, "s-",
            label="Moving average (3 samples)"
        )

        plt.axhline(
            true_distance,
            linestyle=":",
            color="black",
            label="True distance"
        )

        plt.xlabel("Measurement number")
        plt.ylabel("Distance (mm)")

        plt.title(
            f"Filter Comparison - {true_distance} mm, Trial {trial}"
        )

        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save each graph
        plt.savefig(
            f"comparison_{true_distance}mm_trial{trial}.png",
            dpi=300
        )

        plt.close()


# ==========================================
# 7. RESULTS SUMMARY
# ==========================================

results_df = pd.DataFrame(results)

print("\n========== FULL RESULTS ==========")
print(results_df.round(3).to_string(index=False))

# Average performance by distance
summary = results_df.groupby("Distance")[
    ["Raw RMSE", "Kalman RMSE", "MA RMSE"]
].mean()

summary["Kalman Improvement (%)"] = (
    (summary["Raw RMSE"] - summary["Kalman RMSE"])
    / summary["Raw RMSE"]
) * 100

print("\n========== AVERAGE RESULTS ==========")
print(summary.round(3))

# Save results to CSV
results_df.to_csv("filter_results.csv", index=False)
summary.to_csv("filter_summary.csv")

print("\nResults and graphs saved successfully!")


# EXPERIMENT 2: ANGULAR VELOCITY

angular_data = data.iloc[26:38, :].copy()

speeds = ["Slow", "Medium", "Fast", "Very Fast"]
angular_velocities = [33.3, 100, 300, 800]

true_distance = 500

def get_angular_trial(speed, trial):
    rows = angular_data[
        (angular_data.iloc[:, 1] == speed) &
        (pd.to_numeric(
            angular_data.iloc[:, 4],
            errors="coerce"
        ) == trial)
    ]

    if len(rows) != 1:
        raise ValueError(f"Missing data: {speed}, Trial {trial}")

    return rows.iloc[0, 5:25].astype(float).to_numpy()

angular_results = []

for speed, velocity in zip(speeds, angular_velocities):

    calibration = get_angular_trial(speed, 1)
    R = np.var(calibration, ddof=1)

    for trial in [2, 3]:

        measurements = get_angular_trial(speed, trial)

        filtered, gains = kalman_filter(measurements, Q, R)
        ma_filtered = moving_average(measurements, window=3)

        raw_rmse, raw_bias, raw_std = calculate_metrics(
            measurements, true_distance
        )

        kalman_rmse, kalman_bias, kalman_std = calculate_metrics(
            filtered, true_distance
        )

        ma_rmse, ma_bias, ma_std = calculate_metrics(
            ma_filtered, true_distance
        )

        improvement = (
            (raw_rmse - kalman_rmse) / raw_rmse
        ) * 100 if raw_rmse != 0 else 0.0

        angular_results.append({
            "Speed": speed,
            "Angular Velocity": velocity,
            "Trial": trial,
            "R": R,
            "Q": Q,
            "Raw RMSE": raw_rmse,
            "Kalman RMSE": kalman_rmse,
            "MA RMSE": ma_rmse,
            "Raw Bias": raw_bias,
            "Kalman Bias": kalman_bias,
            "MA Bias": ma_bias,
            "Raw Std": raw_std,
            "Kalman Std": kalman_std,
            "MA Std": ma_std,
            "Improvement (%)": improvement 
        })

        samples = np.arange(1, len(measurements) + 1)

        plt.figure(figsize=(10, 5))

        plt.plot(
            samples, measurements, "o--",
            label="Raw measurements"
        )

        plt.plot(
            samples, filtered, "o-",
            label="Kalman estimate"
        )

        plt.plot(
            samples, ma_filtered, "s-",
            label="Moving average"
        )

        plt.axhline(
            true_distance,
            linestyle=":",
            color="black",
            label="True distance"
        )

        plt.xlabel("Measurement number")
        plt.ylabel("Distance (mm)")
        plt.title(
            f"Angular Velocity: {velocity}°/s - Trial {trial}"
        )

        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            f"angular_{velocity}deg_trial{trial}.png",
            dpi=300
        )

        plt.close()

angular_results_df = pd.DataFrame(angular_results)

angular_results_df.to_csv(
    "experiment2_results.csv", index=False
)

print("\n========== EXPERIMENT 2 FULL RESULTS ==========")
print(angular_results_df.round(3).to_string(index=False))

angular_summary = angular_results_df.groupby("Angular Velocity")[
    ["Raw RMSE", "Kalman RMSE", "MA RMSE",
     "Raw Bias", "Kalman Bias", "MA Bias"]
].mean()

angular_summary["Kalman Improvement (%)"] = (
    (angular_summary["Raw RMSE"] - angular_summary["Kalman RMSE"])
    / angular_summary["Raw RMSE"]
) * 100

print("\n========== EXPERIMENT 2 SUMMARY ==========")
print(angular_summary.round(3).to_string())

angular_summary.to_csv("experiment2_summary.csv")


# ==========================================
# 8. EXPERIMENT 2 SUMMARY GRAPHS
# ==========================================

velocities = angular_summary.index.to_numpy()

plt.figure(figsize=(9, 5))

plt.plot(
    velocities, angular_summary["Raw RMSE"],
    "o--", label="Raw measurements"
)

plt.plot(
    velocities, angular_summary["Kalman RMSE"],
    "o-", label="Kalman filter"
)

plt.plot(
    velocities, angular_summary["MA RMSE"],
    "s-", label="Moving average"
)

plt.xlabel("Angular velocity (°/s)")
plt.ylabel("RMSE (mm)")
plt.title("Effect of Angular Velocity on Distance Estimation")

plt.xticks(velocities)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("angular_velocity_rmse.png", dpi=300)
plt.close()






plt.figure(figsize=(9, 5))

plt.plot(
    velocities, angular_summary["Raw Bias"],
    "o--", label="Raw measurements"
)

plt.plot(
    velocities, angular_summary["Kalman Bias"],
    "o-", label="Kalman filter"
)

plt.plot(
    velocities, angular_summary["MA Bias"],
    "s-", label="Moving average"
)

plt.axhline(0, color="black", linestyle=":")

plt.xlabel("Angular velocity (°/s)")
plt.ylabel("Bias (mm)")
plt.title("Effect of Angular Velocity on Measurement Bias")

plt.xticks(velocities)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("angular_velocity_bias.png", dpi=300)
plt.close()

print("\nExperiment 2 summary graphs saved successfully!")
