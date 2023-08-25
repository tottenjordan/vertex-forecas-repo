"""Visualizes uplift performance given a CSV file of uplift modeling results.

This binary serves as a quick way to perform analysis on the predictions of an
uplift model. This currently only works for the single treatment case, with each
experiment yielding its own page of analysis plots. The CSV file must have the
following format:

uplift,response,weight,group,experiment
0.8,0,1.0,0,0
...

Each row represents data for an individual:
- "uplift" is the model's predicted uplift for an individual.
- "response" is the individual's observed response.
- "weight" is the statistical weight assigned to the individual. This is
  optional and defaults to weighting all individuals equally.
- "group" is a {0,1} value representing if the individual belongs to the control
  (0) group or the treatment (1) group.
- "experiment" indicates which treatment this individual belongs to in the case
  of multiple treatments and is an optional column in the case where there is
  just one treatment.

A sample output could look something like:
https://screenshot.googleplex.com/5EkVHWQXmB5CyU7
"""

import collections
import csv
from typing import Any, List, Mapping, Optional, Sequence, TextIO, Tuple, Union

from absl import app
from absl import flags
from absl import logging
import matplotlib as mpl
from matplotlib.backends import backend_pdf

mpl.use('Agg')
import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
import numpy as np

from google3.learning.brain.research.causality.cate_metrics import calibration
from google3.learning.tfx.models.uplift.utils import uplift_utilities
from google3.pyglib import gfile

_INPUT_FILENAME = flags.DEFINE_string(
    'input_filename',
    None,
    (
        'Name of the input CSV file name. '
        'Can be a CNS, Placer, X20, or local file.'
    ),
)
_OUTPUT_FILENAME = flags.DEFINE_string(
    'output_filename',
    None,
    'Name of the (local) output file name to which the plots are saved.',
)
_HISTOGRAM_LOG_Y_SCALE = flags.DEFINE_bool(
    'histogram_log_y_scale',
    False,
    'Whether to scale the y-axis in the histograms logarithmically.',
)
_SCAN_POINTS = flags.DEFINE_integer(
    'scan_points',
    250,
    (
        'An integer specifying the number of thresholds to explore '
        '(on a linear scale).'
    ),
)


def interpolated_difference(x1: np.ndarray, y1: np.ndarray, x2: np.ndarray,
                            y2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """Computes the difference (curve2 - curve1) using interpolation."""

  if len(x1) == len(x2) and np.all(x1 == x2):
    return x1, y2 - y1

  # Assumes x1 and x2 are sorted in the same order.
  if np.all(np.diff(x1) >= 0.0):
    common_x = np.unique(np.concatenate((x1, x2)))
    interp1 = np.interp(common_x, x1, y1)
    interp2 = np.interp(common_x, x2, y2)
  else:
    common_x = np.unique(np.concatenate((x1, x2)))
    interp1 = np.interp(common_x, x1[::-1], y1[::-1])[::-1]
    interp2 = np.interp(common_x, x2[::-1], y2[::-1])[::-1]
  return common_x, interp2 - interp1


def _gather_data_from_csv_file(  # pylint: disable=missing-function-docstring
    csvfile: Any) -> Mapping[Optional[str], Mapping[int, List[float]]]:
  defaultdict = collections.defaultdict
  experiments = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
  uplift_index = None
  response_index = None
  weight_index = None
  group_index = None
  experiment_index = None
  csvreader = csv.reader(csvfile)
  for row in csvreader:
    if uplift_index is None:
      try:
        uplift_index = row.index('uplift')
        response_index = row.index('response')
        group_index = row.index('group')
        if 'weight' in row:
          weight_index = row.index('weight')
        if 'experiment' in row:
          experiment_index = row.index('experiment')
      except ValueError:
        logging.error('The first row in the CSV file must '
                      'contain the column labels.')
        raise
      continue

    uplift = float(row[uplift_index])
    response = float(row[response_index])
    group = int(float(row[group_index]))
    weight = None
    if weight_index is not None:
      weight = float(row[weight_index])
    experiment = None
    if experiment_index is not None:
      experiment = str(row[experiment_index]).strip()

    experiments[experiment][group]['uplift'].append(uplift)
    experiments[experiment][group]['response'].append(response)
    if weight is not None:
      experiments[experiment][group]['weight'].append(weight)

  return experiments


def expected_response_rate(uplift_threshold: float,
                           treatment: uplift_utilities.UpliftCurveInput,
                           control: uplift_utilities.UpliftCurveInput) -> float:
  """Computes the expected response rate given an uplift threshold.

  This computes the expected response rate (ERR) by selecting treatment subjects
  at or above the given uplift threshold and control subjects below the
  threshold.

  Args:
    uplift_threshold: A threshold value for the uplift distributions.
    treatment: A UpliftCurveInput containing the information regarding the
      treatment group.
    control: A UpliftCurveInput containing the information regarding the control
      group.

  Returns:
    The ERR value.
  """
  treatment_indices = treatment.uplift >= uplift_threshold
  control_indices = control.uplift < uplift_threshold

  treatment_weights = treatment.weights[treatment_indices]
  control_weights = control.weights[control_indices]

  return ((np.sum(treatment_weights * treatment.outcomes[treatment_indices]) +
           np.sum(control_weights * control.outcomes[control_indices])) /
          (np.sum(treatment_weights) + np.sum(control_weights)))


def random_response_rate(uplift_threshold: float,
                         treatment: uplift_utilities.UpliftCurveInput,
                         control: uplift_utilities.UpliftCurveInput) -> float:
  """Computes the response rate under random uplift assignment.

  This computes the random response rate (RRR) by assigning the average
  response rate in the treatment group to the treated subjects and likewise for
  the subjects in the control group. This yields what one would expect on
  average by randomly assigning uplift values and then applying the given
  threshold.

  Args:
    uplift_threshold: A threshold value for the uplift distributions.
    treatment: A UpliftCurveInput containing the information regarding the
      treatment group.
    control: A UpliftCurveInput containing the information regarding the control
      group.

  Returns:
    The RRR value.
  """
  n_treatment = np.sum(treatment.uplift >= uplift_threshold)
  n_control = np.sum(control.uplift < uplift_threshold)

  treatment_response_rate = (
      np.sum(treatment.weights * treatment.outcomes) /
      np.sum(treatment.weights))
  control_response_rate = (
      np.sum(control.weights * control.outcomes) / np.sum(control.weights))

  return (n_treatment * treatment_response_rate +
          n_control * control_response_rate) / (
              n_treatment + n_control)


def generate_metrics_and_plots_for_experiment(
    treatment: uplift_utilities.UpliftCurveInput,
    control: uplift_utilities.UpliftCurveInput,
    histogram_log_y_scale: bool = False,
    histogram_bins: Union[str, int, List[float]] = 'auto',
    scan_points: int = 250,
    experiment: Optional[str] = None,
    calibration_bins: int = 25,
    calibration_confidence_interval: int = 90,
) -> Tuple[mpl.figure.Figure, Sequence[mpl.axes.Axes]]:
  """Generates accuracy and calibration metrics.

  Args:
    treatment: uplift estimate, outcome and weights associated with treatment.
    control: uplift estimate, outcome and weights associated with control.
    histogram_log_y_scale: If histogram should be transformed to log scale.
    histogram_bins: Number of bins in the histogram plot.
    scan_points: Number of scan points for histogram threshold.
    experiment: Optional name for the experiment. This is displayed in the graph
      title.
    calibration_bins: Number of bins for the calibration plot.
    calibration_confidence_interval: Confidence interval width for calibration
      error.

  Returns:
    Plots containing accuracy and calibration metrics

  """
  fig, axes = plt.subplots(
      nrows=3,
      ncols=3,
      figsize=[18, 20],
      gridspec_kw={'width_ratios': [12, 12, 1]})

  # Uplift distribution histograms colored by response rate per bin.
  def compute_bar_contents_and_rr(uplift, outcomes, weights):
    bins = np.histogram_bin_edges(uplift, bins=histogram_bins)
    bin_widths = np.diff(bins)
    bin_assignment = np.digitize(uplift, bins=bins, right=True)
    unique_bins = np.unique(bin_assignment)
    bar_content = {
        'x': [],
        'height': [],
        'width': [],
    }
    bin_response_rates = {}
    weights_sum = np.sum(weights)
    for i in unique_bins:
      indices = bin_assignment == i
      bar_content['x'].append(bins[i])
      bar_content['height'].append(np.sum(weights[indices]) / weights_sum)
      bar_content['width'].append(bin_widths[i if i < len(bin_widths) else -1])
      bin_response_rates[i] = np.sum(
          outcomes[indices] * weights[indices]) / np.sum(weights[indices])
    return bar_content, bin_response_rates

  bar_content, treatment_bin_response_rates = compute_bar_contents_and_rr(
      treatment.uplift, treatment.outcomes, treatment.weights)

  treatment_bars = axes[0, 0].bar(
      **bar_content, align='center', log=histogram_log_y_scale)
  axes[0, 0].set_ylabel('normalized count', fontsize=16)
  axes[0, 0].set_xlabel(r'predicted uplift ($u$)', fontsize=16)
  axes[0, 0].tick_params(labelsize=12)
  axes[0, 0].set_title('Treatment', fontsize=18)
  # Reverse uplift axis limits to be in descending order.
  axes[0, 0].set_xlim(*axes[0, 0].get_xlim()[::-1])

  bar_content, control_bin_response_rates = compute_bar_contents_and_rr(
      control.uplift, control.outcomes, control.weights)

  control_bars = axes[0, 1].bar(
      **bar_content, align='center', log=histogram_log_y_scale)
  axes[0, 1].set_xlabel(r'predicted uplift ($u$)', fontsize=16)
  axes[0, 1].tick_params(labelsize=12)
  axes[0, 1].set_title('Control', fontsize=18)
  # Reverse uplift axis limits to be in descending order.
  axes[0, 1].set_xlim(*axes[0, 1].get_xlim()[::-1])

  max_xlim = max(axes[0, 0].get_xlim()[0], axes[0, 1].get_xlim()[0])
  min_xlim = min(axes[0, 0].get_xlim()[1], axes[0, 1].get_xlim()[1])
  axes[0, 0].set_xlim(max_xlim, min_xlim)
  axes[0, 1].set_xlim(max_xlim, min_xlim)
  # Only enforce the same y-scale if the two are within 150% of each other.
  if (abs(axes[0, 1].get_ylim()[1] - axes[0, 0].get_ylim()[1]) /
      min(axes[0, 1].get_ylim()[1], axes[0, 0].get_ylim()[1])) <= 1.5:
    min_ylim = min(axes[0, 0].get_ylim()[0], axes[0, 1].get_ylim()[0])
    max_ylim = max(axes[0, 0].get_ylim()[1], axes[0, 1].get_ylim()[1])
    axes[0, 0].set_ylim(min_ylim, max_ylim)
    axes[0, 1].set_ylim(min_ylim, max_ylim)

  # Per-bin response rate custom color bar.
  cmap = mpl.cm.viridis
  min_rate = None
  max_rate = None
  for i in treatment_bin_response_rates:
    min_rate = treatment_bin_response_rates[i] if not min_rate else min(
        treatment_bin_response_rates[i], min_rate)
    max_rate = treatment_bin_response_rates[i] if not max_rate else max(
        treatment_bin_response_rates[i], max_rate)
  for i in control_bin_response_rates:
    min_rate = min(control_bin_response_rates[i], min_rate)
    max_rate = max(control_bin_response_rates[i], max_rate)
  rate_range = max_rate - min_rate
  for i, b in enumerate(treatment_bin_response_rates):
    treatment_bars.patches[i].set_facecolor(
        cmap((treatment_bin_response_rates[b] - min_rate) / rate_range))
  for i, b in enumerate(control_bin_response_rates):
    control_bars.patches[i].set_facecolor(
        cmap((control_bin_response_rates[b] - min_rate) / rate_range))
  norm = mpl.colors.Normalize(vmin=min_rate, vmax=max_rate)
  cb = mpl.colorbar.ColorbarBase(axes[0, 2], cmap=cmap, norm=norm)
  cb.set_label('Response Rate', fontsize=14)
  cb.ax.tick_params(labelsize=12)
  axes[0, 2].yaxis.set_ticks_position('left')

  # Cumulative distribution plots.
  uplift_threshold, threshold_uplift, threshold_err = (
      uplift_utilities.optimize_uplift_threshold(
          treatment.uplift,
          treatment.outcomes,
          control.uplift,
          control.outcomes,
          treatment_weights=treatment.weights,
          control_weights=control.weights,
          scan_points=scan_points,
          return_err_versus_threshold_curves=True))

  _, expected_responses, expected_weights, treatment_fraction = (
      uplift_utilities.apply_threshold(
          uplift_threshold,
          treatment.uplift,
          treatment.outcomes,
          control.uplift,
          control.outcomes,
          treatment_weights=treatment.weights,
          control_weights=control.weights))

  treatment_distribution = (
      np.cumsum(treatment.weights) / np.sum(treatment.weights))
  treatment_cumulative_responses = np.cumsum(
      treatment.outcomes * treatment.weights) / np.sum(treatment.weights)
  control_distribution = np.cumsum(control.weights) / np.sum(control.weights)
  control_cumulative_responses = np.cumsum(
      control.outcomes * control.weights) / np.sum(control.weights)
  reversed_control_cumulative_responses = np.cumsum(
      (control.outcomes * control.weights)[::-1])[::-1] / np.sum(
          control.weights)

  # Threshold plot.
  axes[1, 0].plot(
      treatment.uplift,
      treatment_cumulative_responses,
      label='treatment',
      linestyle='-.')
  axes[1, 0].plot(
      control.uplift,
      reversed_control_cumulative_responses,
      label='control',
      linestyle=':')
  axes[1, 0].plot(threshold_uplift, threshold_err, label='ERR @ threshold')

  axes[1, 0].vlines(
      uplift_threshold, *axes[1, 0].get_ylim(), linestyles='dashed', alpha=0.5)

  axes[1, 0].set_title(r'Uplift Threshold Plot', fontsize=18)
  axes[1, 0].set_ylabel('response rate', fontsize=16)
  axes[1, 0].set_xlabel(r'predicted uplift ($u$)', fontsize=16)
  axes[1, 0].tick_params(labelsize=12)
  axes[1, 0].set_xlim(*axes[1, 0].get_xlim()[::-1])
  axes[1, 0].legend(fontsize=12)

  # Cumulative responses versus population fraction.
  axes[1, 1].plot(
      treatment_distribution,
      treatment_cumulative_responses,
      label='treatment',
      linestyle='-.')
  axes[1, 1].plot(
      control_distribution,
      control_cumulative_responses,
      label='control',
      linestyle=':')
  axes[1, 1].plot(
      np.cumsum(expected_weights) / np.sum(expected_weights),
      np.cumsum(expected_responses * expected_weights) /
      np.sum(expected_weights),
      label='expected @ optimal threshold')

  axes[1, 1].vlines(
      treatment_fraction,
      *axes[1, 1].get_ylim(),
      linestyles='dashed',
      alpha=0.5)

  axes[1, 1].set_title(r'Response Rate vs $x$', fontsize=18)
  axes[1, 1].set_xlabel(r'population fraction ($x$)', fontsize=16)
  axes[1, 1].tick_params(labelsize=12)
  axes[1, 1].legend(fontsize=12)

  axes[1, 2].set_visible(False)

  # Show key analysis results.
  axes[1, 1].text(
      *(1.05, 0.9), ('Key Analysis Results:'),
      fontsize=14,
      transform=axes[1, 1].transAxes)
  axes[1, 1].text(
      *(1.05, 0.75), ('Uplift Threshold: {:.5}\n'.format(uplift_threshold) +
                      'Treated Fraction: {:.5}'.format(treatment_fraction)),
      fontsize=14,
      transform=axes[1, 1].transAxes)
  axes[1, 1].text(
      *(1.05, 0.6),
      ('Expected Response Rate: {:.3}\n'.format(
          expected_response_rate(uplift_threshold, treatment, control)) +
       'Random Response Rate: {:.3}'.format(
           random_response_rate(uplift_threshold, treatment, control))),
      fontsize=14,
      transform=axes[1, 1].transAxes)

  curves, metrics = uplift_utilities.compute_uplift_curves_and_metric(
      treatment,
      control,
      label_type=uplift_utilities.UpliftLabelType.BINARY,
      metric_types=[
          uplift_utilities.UpliftMetricType.AUUC,
          uplift_utilities.UpliftMetricType.QINI_CONTINUOUS,
      ])
  colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

  # Qini curves.
  x, y = interpolated_difference(curves.obs_auuc.control.covariate,
                                 curves.obs_auuc.control.values,
                                 curves.obs_auuc.treatment.covariate,
                                 curves.obs_auuc.treatment.values)
  axes[2, 0].plot(x, y, label='observed', color=colors[0])
  x, y = interpolated_difference(curves.random_auuc.control.covariate,
                                 curves.random_auuc.control.values,
                                 curves.random_auuc.treatment.covariate,
                                 curves.random_auuc.treatment.values)
  axes[2, 0].plot(x, y, label='random', linestyle=':', color=colors[1])
  axes[2, 0].vlines(
      treatment_fraction,
      *axes[2, 0].get_ylim(),
      linestyles='dashed',
      alpha=0.5)
  axes[2, 0].set_title('Uplift Curve', fontsize=18)
  axes[2, 0].set_ylabel('cumulative uplift', fontsize=16)
  axes[2, 0].set_xlabel(r'treatment fraction ($x$)', fontsize=16)
  axes[2, 0].tick_params(labelsize=12)
  axes[2, 0].legend(fontsize=12)

  # Calibration metrics
  # TODO(sharatsharat): Add test_covariates as optional parameters and switch to
  # AIPW when present.
  calibration_results = calibration.ipw_calibration(
      preds_cate_sub=np.concatenate((treatment.uplift, control.uplift)),
      test_dataset_label=np.concatenate((treatment.outcomes, control.outcomes)),
      test_dataset_treatment=np.concatenate(
          (np.ones_like(treatment.uplift), np.zeros_like(control.uplift))),
      return_curve=True,
      bins=scan_points,
      plot_bins=calibration_bins)
  conf_alpha = (1.0 - 0.01 * calibration_confidence_interval) * 0.5
  p_alpha, p_one_minus_alpha = np.quantile(
      calibration_results['bootstrap_samples'], [conf_alpha, 1.0 - conf_alpha])
  axes[2, 1].set_title('Calibration curve', fontsize=18)
  axes[2, 1].set_ylabel('Observed uplift', fontsize=16)
  axes[2, 1].set_xlabel('Predicted uplift', fontsize=16)
  axes[2, 1].plot(calibration_results['curve_pred'],
                  calibration_results['curve_obs'])
  axes[2, 1].text(
      *(1.05, 0.8),
      ('Qini coefficient: {:.2} \n\n'.format(metrics.qini_continuous) +
       'Calibration error: {:.2e} \n' +
       'Calibration interval ({}%): [{:.2e},{:.2e}]').format(
           calibration_results['estimate'], calibration_confidence_interval,
           p_alpha, p_one_minus_alpha),
      fontsize=14,
      transform=axes[2, 1].transAxes)

  axes[2, 2].set_visible(False)

  if experiment is not None:
    fig.suptitle(
        f'Uplift Performance Plots: {experiment}',
        y=.925,
        fontweight=650,
        fontsize=20)

  plt.subplots_adjust(hspace=0.25)

  return fig, axes


def run_analysis(
    csvfile: TextIO,
    histogram_log_y_scale: bool = False,
    histogram_bins: Union[str, int, List[float]] = 'auto',
    scan_points: int = 250
) -> Mapping[Optional[str], Tuple[mpl.figure.Figure, Sequence[mpl.axes.Axes]]]:
  """Compute uplift plots and metrics given a CSV file.

  Args:
    csvfile: An openned IO buffer containing the CSV file contents.
    histogram_log_y_scale: Flag to control whether the histograms of the uplift
      should be logarithmically scaled. Useful in cases when there are sparsely
      populated outliers.
    histogram_bins: Argument passed to the Numpy histogram function specifying
      how to bin the histogram.
    scan_points: An integer specifying the number of uplift thresholds to
      explore (on a linear scale) when finding the optimal threshold.

  Returns:
    A mapping between the experiment IDs present in the CSV file (or `None` if
    not specified) and the figures and axes containing the analysis plots.
  """
  experiments = _gather_data_from_csv_file(csvfile)

  data = {}
  for experiment in experiments:
    if experiment is not None:
      logging.info('Computing analysis for experiment: %s.', experiment)
    else:
      logging.info('Computing analysis.')

    control_uplift = np.array(experiments[experiment][0]['uplift'])
    control_response = np.array(experiments[experiment][0]['response'])
    control_weight = None
    if 'weight' in experiments[experiment][0]:
      control_weight = np.array(experiments[experiment][0]['weight'])
    treatment_uplift = np.array(experiments[experiment][1]['uplift'])
    treatment_response = np.array(experiments[experiment][1]['response'])
    treatment_weight = None
    if 'weight' in experiments[experiment][1]:
      treatment_weight = np.array(experiments[experiment][1]['weight'])

    # This automatically sorts the distributions by uplift which is useful for
    # the cumulative distributions.
    control_input = uplift_utilities.UpliftCurveInput(
        control_uplift, control_response, weights=control_weight)
    treatment_input = uplift_utilities.UpliftCurveInput(
        treatment_uplift, treatment_response, weights=treatment_weight)

    data[experiment] = generate_metrics_and_plots_for_experiment(
        treatment_input, control_input, histogram_log_y_scale, histogram_bins,
        scan_points, experiment)

  return data


def save_analysis(  # pylint: disable=missing-function-docstring
    output_filename: str,
    data: Mapping[Optional[str], Tuple[mpl.figure.Figure,
                                       Sequence[mpl.axes.Axes]]]) -> None:
  with backend_pdf.PdfPages(output_filename) as pdf:
    for experiment in data:
      fig, _, = data[experiment]
      pdf.savefig(fig, bbox_inches='tight')
  logging.info('Saved plots to %s.', output_filename)


def main(argv):
  del argv

  with gfile.Open(_INPUT_FILENAME.value) as csvfile:
    data = run_analysis(
        csvfile, _HISTOGRAM_LOG_Y_SCALE.value, scan_points=_SCAN_POINTS.value
    )
  save_analysis(_OUTPUT_FILENAME.value, data)


if __name__ == '__main__':
  flags.mark_flag_as_required('input_filename')
  flags.mark_flag_as_required('output_filename')
  app.run(main)