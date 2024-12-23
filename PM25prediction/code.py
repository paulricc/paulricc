import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from pykalman import KalmanFilter
from scipy.optimize import minimize
from geopy.distance import geodesic
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.api as sm

"""Import and Preprocess data"""

data = pd.read_csv('/content/ts_epa_2020_west_sept_fill.csv')
data.info()
data.head()

station_96 = data[data['station_id'] == 96].copy()
station_96['datetime'] = pd.to_datetime(station_96['datetime'])

"""Descriptive Statistics"""

descr_stats = station_96[["pm25", 'temp', 'wind']]
descr_stats_summary = descr_stats.describe().T
descr_stats_summary["range"] = descr_stats_summary['max'] - descr_stats_summary['min']

plt.figure(figsize=(12,6))

plt.fill_between(
    station_96["datetime"],
    25,
    station_96["pm25"].max(),
    color='red',
    alpha=0.2,
    label = 'Dangerous PM 2.5 levels'
)

plt.plot(station_96["datetime"], station_96["pm25"], label='PM 2.5 levels')
plt.axhline(y=25, color="red", linestyle="--")
plt.legend()
plt.title('PM 2.5 levels at station 96')
plt.xlabel('Time')
plt.ylabel('PM 2.5 levels')
plt.show()

# Daily observations

variables = ["pm25", "temp", "wind"]
labels_facet = {
  "pm25": "PM 2.5",
  "temp": "Temperature",
  "wind": "Wind"
  }

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

for ax, var in zip(axes, variables):
  ax.plot(station_96["datetime"], station_96[var], label=labels_facet[var])
  ax.set_title(labels_facet[var])
  ax.set_ylabel(var)
  ax.legend()

axes[-1].set_xlabel("Datetime")

plt.tight_layout()

plt.show()

# Observations every 2 days

station_96["date"] = station_96["datetime"].dt.date
station_96_filtered = station_96[station_96["date"].apply(lambda x: x.toordinal() % 2 == 0)]

variables = ["pm25", "temp", "wind"]
labels_facet = {
  "pm25": "PM 2.5",
  "temp": "Temperature",
  "wind": "Wind"
  }

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

for ax, var in zip(axes, variables):
  ax.plot(station_96_filtered["datetime"], station_96_filtered[var], label=labels_facet[var])
  ax.set_title(labels_facet[var])
  ax.set_ylabel(var)
  ax.legend()

axes[-1].set_xlabel("Datetime")

plt.tight_layout()

plt.show()


"""Hidden Markov Model"""

model = GaussianHMM(n_components=3, random_state=29)
model.fit(station_96[['pm25']].values)
station_96['state'] = model.predict(station_96[['pm25']].values)

hmm_state_means = model.means_.flatten()
station_96["hmm_fitted"] = hmm_state_means[station_96["state"]]
hmm_state_variances = np.sqrt(model.covars_).flatten()
transition_matrix = model.transmat_
initial_prob = model.startprob_

print("Transition Matrix:\n", transition_matrix)
print("Initial Probabilities:\n", initial_prob)
print("State Means:\n", hmm_state_means)
print("State Standard Deviations:\n", hmm_state_variances)



""" Random Walk plus noise"""

def log_likelihood(params, data):
    transition_covariance, observation_covariance = params
    kf = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        initial_state_mean=data[0],
        initial_state_covariance=1,
        observation_covariance=observation_covariance,
        transition_covariance=transition_covariance
    )
    loglikelihood = kf.loglikelihood(data)
    return -loglikelihood


observations = station_96["pm25"].values
result = minimize(
    fun=log_likelihood,
    x0=[1, 1],
    args=(observations,),
    bounds=[(1e-5, None), (1e-5, None)]
)


optimized_transition_covariance, optimized_observation_covariance = result.x

print("Estimated Variances:")
print(f"Transition Variance (Process Noise): {optimized_transition_covariance}")
print(f"Observation Variance (Observation Noise): {optimized_observation_covariance}")

kf = KalmanFilter(
    transition_matrices=[1],
    observation_matrices=[1],
    initial_state_mean=observations[0],
    initial_state_covariance=1,
    observation_covariance=optimized_observation_covariance,
    transition_covariance=optimized_transition_covariance
)

filtered_state_means, filtered_state_covariances = kf.filter(observations)
observation_means, smoothed_state_covariances = kf.smooth(observations)

def calculate_error_metrics(observations, predictions):

  mae = mean_absolute_error(observations, predictions)
  msfe = mean_squared_error(observations, predictions)
  mape = np.mean(np.abs((observations - predictions)/observations)) * 100

  return {'MAE': mae, 'MSFE': msfe, 'MAPE': mape}


error_metrics = calculate_error_metrics(observations, observation_means)

print("Error Metrics:")
print(f"MAE: {error_metrics['MAE']:.4f}")
print(f"MAPE: {error_metrics['MAPE']:.2f}%")
print(f"MSFE: {error_metrics['MSFE']:.4f}")

filtered_std_dev = np.sqrt(np.diagonal(filtered_state_covariances))
lower_bound = filtered_state_means - 1.96 * filtered_std_dev
upper_bound = filtered_state_means + 1.96 * filtered_std_dev

plt.figure(figsize=(12, 9))
plt.plot(observations, label='Actual Observations', color='black', linewidth=8)
plt.plot(filtered_state_means, label='Forecasted Observations', color='blue', linewidth=1)
plt.fill_between(
    np.arange(len(observations)),
    lower_bound.flatten(),
    upper_bound.flatten(),
    color="blue",
    alpha=0.2,
    label="95% Confidence Interval",
)
plt.title("Forecasts with 95% Confidence Intervals")
plt.xlabel("Time")
plt.ylabel("PM2.5")
plt.legend()
plt.show()


"""Dynamic Linear Model with regressive component"""

observations = station_96["pm25"].values
regressor = station_96["wind"].values.reshape(-1, 1)

model = sm.tsa.SARIMAX(endog=observations, exog=regressor, order=(0, 0, 0), trend='n')
result = model.fit()

print(result.summary())

predicted = result.fittedvalues

mae = mean_absolute_error(observations, predicted)
mse = mean_squared_error(observations, predicted)
mape = np.mean(np.abs((observations - predicted) / observations)) * 100

print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"MAPE: {mape:.2f}%")



"""Multivariate model"""

station_96['intercept'] = 1
out_var = station_96['pm25'].values
regressors = station_96[['intercept', 'wind', 'temp']].values

model = sm.tsa.SARIMAX(
    endog = out_var,
    exog = regressors,
    order = (0,0,0),
    trend='n',
    time_varying_regression = True,
    mle_regression = False
)

result = model.fit()

print(result.summary())

dynamic_coefficients

dynamic_coefficients = result.smoothed_state.T

plt.figure(figsize=(12, 6))
plt.plot(station_96['datetime'], dynamic_coefficients[:, 0], label = 'Intercept', color = 'red')
plt.plot(station_96['datetime'], dynamic_coefficients[:, 1], label = 'Wind coefficients', color = 'blue')
plt.plot(station_96['datetime'], dynamic_coefficients[:, 2], label = 'Temperature coefficients', color = 'green')
plt.title('Dynamic coefficients over time')
plt.xlabel('Time')
plt.ylabel('Coefficients')
plt.legend()
plt.show()
