"""Functions for computing Qini metrics.

These functions are used to produce both uplift curves and compute metrics from
those curves. Detailed descriptions of the methodology used here can be found at
(go/uplift-metrics)
"""

import enum
from typing import Optional, Sequence, Set, Text, Tuple, Union

from absl import logging

import attr
import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds


class UpliftMetricType(enum.Enum):
  # go/uplift-concepts#auuc
  AUUC = 'area_under_uplift_curve'
  # go/uplift-concepts#a-continuous-metric-q-c
  QINI_CONTINUOUS = 'qini_continuous'
  # go/uplift-concepts#a-bounded-metric-for-binary-responses-q
  QINI_MIN_SURE_THING = 'qini_minimum_sure_things_ideal'
  # go/uplift-concepts#a-binary-and-continuous-metric-q-0
  QINI_MAX_SURE_THING = 'qini_maximum_sure_things_ideal'


class UpliftLabelType(enum.Enum):
    CONTINUOUS = 'continuous'
    BINARY = 'binary'


@attr.s(slots=True, frozen=True)
class Curve(object):
    """Pair of arrays representing the covariates and values of an uplift curve.

    Attributes:
      covariate: The covariates.
      values: The values of the uplift curve. Both arrays are of equal length.
    """
    covariate = attr.ib(type=np.ndarray)
    values = attr.ib(type=np.ndarray)


@attr.s(slots=True, frozen=True)
class UpliftCurve(object):
    """Represents the treatment and control grouping of Curve objects.

  Attributes:
    treatment: The treatment set of uplift curves.
    control: The control set of uplift curves.
  """
    treatment = attr.ib(type=Curve)
    control = attr.ib(type=Curve)


@attr.s(slots=True, frozen=True)
class UpliftCurves(object):
    """A set of uplift curves.

  Attributes:
    obs_auuc: Curves needed to compute the AUUC metric.
    random_auuc: Curves needed to compute the AUUC for the random model.
    min_sure_thing_qini: Curves needed to compute the Qini metric given a
      minimum-sure-things hypothesis.
    max_sure_thing_qini: Curves needed to compute the Qini metric given a
      maximum-sure-things hypothesis.
    """
    obs_auuc = attr.ib(type=UpliftCurve)
    random_auuc = attr.ib(type=Optional[UpliftCurve], default=None)
    min_sure_thing_qini = attr.ib(type=Optional[UpliftCurve], default=None)
    max_sure_thing_qini = attr.ib(type=Optional[UpliftCurve], default=None)


@attr.s(slots=True, frozen=True)
class UpliftMetrics(object):
    """A set of uplift metrics.

  Attributes:
    auuc: The AUUC metric.
    qini_continuous: The "continuous" Qini metric.
    qini_max_sure_thing: The Qini metric assuming the maximum number of
      sure-things.
    qini_min_sure_thing: The Qini metric assuming the minimum number of
      sure-things.
  """
    auuc = attr.ib(type=Optional[float], default=None)
    qini_continuous = attr.ib(type=Optional[float], default=None)
    qini_min_sure_thing = attr.ib(type=Optional[float], default=None)
    qini_max_sure_thing = attr.ib(type=Optional[float], default=None)


@attr.s(slots=True, frozen=True)
class UpliftCurveInput(object):
    """
    A data class used for grouping inputs to compute uplift curves.

    Attributes:
      uplift: The predicted uplift for either a treatment or control group.
        All arrays will be sorted w.r.t. this array in descending order.
      outcomes: The observed outcomes. Entries should correspond to entries in the
        `uplift` array.
      weights: The weight for each subject. The default is to weigh all subjects
        equally. If provided, entries should correspond to entries in the `uplift`
        array.

    Raises:
      ValueError: Any of the weights are negative.
      ValueError: All arrays are not of equal length.
    """
    uplift = attr.ib(type=np.ndarray)
    outcomes = attr.ib(type=np.ndarray)
    weights = attr.ib(type=Optional[np.ndarray], default=None)

  @outcomes.validator
  def outcomes_check(self, attribute, value):
        if len(value) != len(self.uplift):
            raise ValueError('All arrays must be of equal length.')

  @weights.validator
  def weights_check(self, attribute, value):
        if value is not None and len(value) != len(self.uplift):
            raise ValueError('All arrays must be of equal length.')
        if value is not None and not np.all(value >= 0.0):
            raise ValueError('Elements in weights must be non-negative.')

  def __attrs_post_init__(self):
    outcomes = self.outcomes
    uplift = self.uplift
    weights = self.weights

    if weights is None:
        weights = np.ones_like(outcomes)

    outcomes = outcomes.flatten()
    uplift = uplift.flatten()
    weights = weights.flatten()

    # Sort in descending order of uplift (argsort returns ascending).
    # Reversing the arrays preserves the ordering after the stable sort.
    uplift_sorted_indices = np.argsort(uplift[::-1], kind='stable')[::-1]
    outcomes = outcomes[::-1][uplift_sorted_indices]
    uplift = uplift[::-1][uplift_sorted_indices]
    weights = weights[::-1][uplift_sorted_indices]

    # Using https://www.attrs.org/en/stable/init.html#post-init-hook.
    object.__setattr__(self, 'outcomes', outcomes)
    object.__setattr__(self, 'uplift', uplift)
    object.__setattr__(self, 'weights', weights)


def _compute_ideal_curve(
    treatment_fraction: float, control_fraction: float,
    sure_things_fraction: float, x: np.ndarray,
    control: bool = False
) -> np.ndarray:
    """Computes an ideal Qini curve.

      Args:
        treatment_fraction: The fraction of positive responses in the treatment
          group.
        control_fraction: The fraction of positive responses in the control group.
        sure_things_fraction: The desired fraction of sure-things. No check is made
          here to ensure consistency between treatment_fraction, control_fraction,
          and sure_things_fraction.
        x: An array of population fractions at which to evaluate the ideal curve.
        control: A flag indicating whether to compute the treatment or control ideal
          curves. Defaults to computing the ideal treatment curve.
      Returns:
        An ndarray containing the ideal uplift curve given the specified rates.
      """
    persuadables_fraction = treatment_fraction - sure_things_fraction
    sleeping_dogs_fraction = control_fraction - sure_things_fraction
    lost_causes_fraction = (1.0 - treatment_fraction - control_fraction + sure_things_fraction)

    if control:
        # Ideal control.
        return np.where(
            x < persuadables_fraction,
            # Persuadables.
            np.zeros_like(x),
            np.where(
                x < 1.0 - sleeping_dogs_fraction,
                # Even mixture of sure-things and lost-causes.
                (sure_things_fraction / (sure_things_fraction + lost_causes_fraction)) * (x - persuadables_fraction),
                # Sleeping-dogs.
                (x - 1.0) + control_fraction
            )
        )
    # Ideal treatment.
    return np.where(
        x < persuadables_fraction,
        # Persuadables.
        x,
        np.where(
            x < 1.0 - sleeping_dogs_fraction,
            # Even mixture of sure-things and lost-causes.
            (sure_things_fraction / (sure_things_fraction + lost_causes_fraction)) * (x - persuadables_fraction) + persuadables_fraction,
            # Sleeping-dogs.
            treatment_fraction * np.ones_like(x)
        )
    )


def _compute_curves_and_metric(
    treatment_distribution: np.ndarray,
    weighted_normed_treatment_cumulative_outcomes: np.ndarray,
    control_distribution: np.ndarray,
    weighted_normed_control_cumulative_outcomes: np.ndarray,
    label_type: UpliftLabelType, metric_types: Set[UpliftMetricType]
    ) -> Tuple[UpliftCurves, UpliftMetrics]:
    """Computes various AUUC-based metrics.

  Args:
    treatment_distribution: An np.ndarray of the distribution of treatment
      subjects.
    weighted_normed_treatment_cumulative_outcomes: An np.ndarray of cumulative
      weighted responses for the treatment group.
    control_distribution: An np.ndarray of the distribution of control subjects.
    weighted_normed_control_cumulative_outcomes: An np.ndarray of cumulative
      weighted responses for the control group.
    label_type: An UpliftLabelType enum indicating how to interpret the outcomes
      when computing the metric.
    metric_types: A set of desired UpliftMetricType enums indicating the types
      of uplift metrics to return. Defaults to computing the AUUC.

  Returns:
    A UpliftCurves object of requisite curves needed to compute the metrics
    along with an UpliftMetrics object containing the values of the specified
    metrics.

  Raises:
    ValueError: If the metric_types and label_type are incompatible.
  """
    return_auuc = UpliftMetricType.AUUC in metric_types
    return_qini_continuous = UpliftMetricType.QINI_CONTINUOUS in metric_types
    return_binary_qini_min_sure_thing = (
        UpliftMetricType.QINI_MIN_SURE_THING in metric_types and
        label_type == UpliftLabelType.BINARY
    )
    return_binary_qini_max_sure_thing = (
        UpliftMetricType.QINI_MAX_SURE_THING in metric_types and
        label_type == UpliftLabelType.BINARY
    )

    metric_names_list = ','.join(
        metric_type.value for metric_type in metric_types)
    if (not return_auuc and
        not return_qini_continuous and
        not return_binary_qini_min_sure_thing and
        not return_binary_qini_max_sure_thing):
        raise ValueError(
            'No available uplift metric for {} '.format(metric_names_list) + 'and label type {}.'.format(label_type.value))
    else:
        logging.info(
            'Computing metrics %s with label type %s.',
            metric_names_list, label_type.value
        )

    treatment_auuc = np.trapz(
        x=treatment_distribution,
        y=weighted_normed_treatment_cumulative_outcomes)
    control_auuc = np.trapz(
        x=control_distribution,
        y=weighted_normed_control_cumulative_outcomes)
    obs_auuc = treatment_auuc - control_auuc
    obs_auuc_group = UpliftCurve(
        Curve(
            treatment_distribution,
            weighted_normed_treatment_cumulative_outcomes),
        Curve(
            control_distribution, weighted_normed_control_cumulative_outcomes))

    requisite_curves = {'obs_auuc': obs_auuc_group}
    computed_metrics = {}

    if return_auuc:
        computed_metrics['auuc'] = obs_auuc
        metric_types.discard(UpliftMetricType.AUUC)

    if not metric_types:
        return UpliftCurves(**requisite_curves), UpliftMetrics(**computed_metrics)

    treatment_rate = weighted_normed_treatment_cumulative_outcomes[-1]
    control_rate = weighted_normed_control_cumulative_outcomes[-1]

    # Random model in treatment coordinates.
    treatment_random_auuc = np.trapz(
        x=treatment_distribution,
        y=(treatment_rate * treatment_distribution))
    # Random model in control coordinates.
    control_random_auuc = np.trapz(
        x=control_distribution,
      y=(control_rate * control_distribution))
    random_auuc = treatment_random_auuc - control_random_auuc
    random_auuc_group = UpliftCurve(
      Curve(
          treatment_distribution, treatment_rate * treatment_distribution),
      Curve(
          control_distribution, control_rate * control_distribution))
    requisite_curves['random_auuc'] = random_auuc_group

    obs_minus_random = obs_auuc - random_auuc

    if return_qini_continuous:
        computed_metrics['qini_continuous'] = 2 * obs_minus_random
        metric_types.discard(UpliftMetricType.QINI_CONTINUOUS)

    def compute_ideal_denom(f_st):
        ideal_distribution = np.array([0.0, treatment_rate - f_st,
                                   1.0 - (control_rate - f_st), 1.0])

        ideal_treatment_curve = _compute_ideal_curve(
            treatment_rate, control_rate, f_st, ideal_distribution,
            control=False
        )
        treatment_ideal_auuc = np.trapz(ideal_treatment_curve, ideal_distribution)
        ideal_control_curve = _compute_ideal_curve(
            treatment_rate, control_rate, f_st, ideal_distribution,
            control=True)
        control_ideal_auuc = np.trapz(ideal_control_curve, ideal_distribution)
        ideal_auuc = treatment_ideal_auuc - control_ideal_auuc
        ideal_auuc_group = UpliftCurve(
            Curve(
                ideal_distribution, ideal_treatment_curve),
            Curve(
                ideal_distribution, ideal_control_curve))

        return ideal_auuc_group, ideal_auuc - random_auuc

    if return_binary_qini_min_sure_thing:
        # Minimum fraction of sure-things.
        if treatment_rate + control_rate < 1.0:
            f_st = 0.0
        else:
            f_st = treatment_rate + control_rate - 1.0

        ideal_auuc_group, ideal_minus_random = compute_ideal_denom(f_st)
        requisite_curves['min_sure_thing_qini'] = ideal_auuc_group
        computed_metrics['qini_min_sure_thing'] = (obs_minus_random / ideal_minus_random)
        metric_types.discard(UpliftMetricType.QINI_MIN_SURE_THING)

        if return_binary_qini_max_sure_thing:
            # Maximum fraction of sure-things.
            f_st = np.min([treatment_rate, control_rate])

            ideal_auuc_group, ideal_minus_random = compute_ideal_denom(f_st)
            requisite_curves['max_sure_thing_qini'] = ideal_auuc_group
            computed_metrics['qini_max_sure_thing'] = (
                obs_minus_random / ideal_minus_random)
            metric_types.discard(UpliftMetricType.QINI_MAX_SURE_THING)

    return UpliftCurves(**requisite_curves), UpliftMetrics(**computed_metrics)


def apply_threshold(
    uplift_threshold: float,
    treatment_uplift: np.ndarray,
    treatment_outcomes: np.ndarray,
    control_uplift: np.ndarray,
    control_outcomes: np.ndarray,
    treatment_weights: Optional[np.ndarray] = None,
    control_weights: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Computes expected distributions after applying an uplift threshold.

  Args:
    uplift_threshold: A threshold value on the uplift distributions.
    treatment_uplift: An np.ndarray containing the treatment group's uplift
      values.
    treatment_outcomes: An np.ndarray containing the treatment group's outcome
      values.
    control_uplift: An np.ndarray containing the control group's uplift values.
    control_outcomes: An np.ndarray containing the control group's outcome
      values.
    treatment_weights: An np.ndarray containing the treatment group's weight
      values.
    control_weights: An np.ndarray containing the control group's weight values.

  Returns:
    The uplift, outcomes, and weight distributions after applying the threshold
    value to the uplift distributions as well as the fraction of subjects
    treated.
  """

    if treatment_weights is None:
        treatment_weights = np.ones_like(treatment_uplift)
    if control_weights is None:
        control_weights = np.ones_like(control_uplift)

    treatment_indices = treatment_uplift >= uplift_threshold
    treatment_uplift = treatment_uplift[treatment_indices]
    treatment_weights = treatment_weights[treatment_indices]
    treatment_outcomes = treatment_outcomes[treatment_indices]

    control_indices = control_uplift < uplift_threshold
    control_uplift = control_uplift[control_indices]
    control_weights = control_weights[control_indices]
    control_outcomes = control_outcomes[control_indices]

    uplift = np.concatenate((treatment_uplift, control_uplift))
    outcomes = np.concatenate((treatment_outcomes, control_outcomes))
    weights = np.concatenate((treatment_weights, control_weights))

    treated_fraction = (np.sum(treatment_weights) / (np.sum(treatment_weights) + np.sum(control_weights)))

    return uplift, outcomes, weights, treated_fraction


def optimize_uplift_threshold(
    treatment_uplift: np.ndarray, treatment_outcomes: np.ndarray,
    control_uplift: np.ndarray, control_outcomes: np.ndarray,
    treatment_weights: Optional[np.ndarray] = None,
    control_weights: Optional[np.ndarray] = None,
    scan_points: int = 250,
    return_err_versus_threshold_curves: bool = False
    ) -> Union[float, Tuple[float, np.ndarray, np.ndarray]]:
    """Determines the optimal uplift threshold based on ERR.

  This function computes an optimal uplift threshold by maximizing the expected
  response rate (ERR). This function does not assume the uplift is sorted and
  performs a simple linear scan over various uplift values to find an optimal
  threshold. Most importantly the number of points to scan over should be far
  fewer than the numbers of elements in the uplift and outcomes arrays. This is
  because the primary computational assumption is that the outcomes are evenly
  distributed such that an arbitrary slice on uplift yields a smoothly varying
  cumulative distribution of responses. This is rarely the case in practice and
  scanning too finely will yield artificially noisy estimates of the ERR.

  Args:
    treatment_uplift: An np.ndarray containing the treatment group's uplift
      values.
    treatment_outcomes: An np.ndarray containing the treatment group's outcome
      (or response) values.
    control_uplift: An np.ndarray containing the control group's uplift values.
    control_outcomes: An np.ndarray containing the control group's outcome
      (or response) values.
    treatment_weights: An np.ndarray containing the treatment group's weight
      values.
    control_weights: An np.ndarray containing the control group's weight values.
    scan_points: An integer specifying the number of thresholds to explore (on
      a linear scale).
    return_err_versus_threshold_curves: Whether or not to return the uplift and
      ERR arrays.

  Returns:
    A threshold value that when applied to the uplift distributions in the
    treatment and control groups yields the maximum expected response rate. If
    `return_err_versus_threshold_curves` is `True` then also returns the uplift
    and ERR arrays.
    """

    if treatment_weights is None:
        treatment_weights = np.ones_like(treatment_uplift)
    if control_weights is None:
        control_weights = np.ones_like(control_uplift)

    uplift_thresholds = np.linspace(
        min(np.min(treatment_uplift), np.min(control_uplift)),
        max(np.max(treatment_uplift), np.max(control_uplift)),
        num=scan_points)

    def _apply_threshold(uplift_threshold):
        treatment_indices = treatment_uplift >= uplift_threshold
        control_indices = control_uplift < uplift_threshold

        return (
            treatment_uplift[treatment_indices],
            treatment_outcomes[treatment_indices],
            treatment_weights[treatment_indices],
            control_uplift[control_indices],
            control_outcomes[control_indices],
            control_weights[control_indices]
        )

    def compute_expected_response_rate(uplift_threshold):
        _, to, tw, _, co, cw = _apply_threshold(uplift_threshold)
        return (np.sum(to * tw) + np.sum(co * cw)) / (np.sum(tw) + np.sum(cw))

    v_compute_expected_response_rate = np.vectorize(
        compute_expected_response_rate
    )

    expected_response_rates = v_compute_expected_response_rate(uplift_thresholds)
    # NOTE: np.argmax will return the _first_ occurrence of the maximum. This will
    #       correspond to a higher uplift threshold and lower treated fraction.
    max_index = np.argmax(expected_response_rates)

    if return_err_versus_threshold_curves:
        return (
            uplift_thresholds[max_index], 
            uplift_thresholds,
            expected_response_rates
        )
    return uplift_thresholds[max_index]


def compute_uplift_curves_and_metric_on_keras_model(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    treatment_key: Text,
    label_type: UpliftLabelType = UpliftLabelType.BINARY,
    metric_types: Optional[Set[UpliftMetricType]] = None
) -> Tuple[UpliftCurves, UpliftMetrics]:
    """Computes various AUUC-based metrics with given `model` and `dataset`.

  Args:
    model: A Uplift keras model, which prediction is the uplift scores.
    dataset: The data used from metric computation.
    treatment_key: The name of the treatment column.
    label_type: An UpliftLabelType enum indicating how to interpret the outcomes
      when computing the metric.
    metric_types: A sequence of desired UpliftMetricType enums indicating the
      types of uplift metrics to return. Defaults to computing the AUUC.

  Returns:
    A UpliftCurves object of requisite curves needed to compute the metrics
    along with an UpliftMetrics object containing the values of the specified
    metrics.
    """
    predictions = model.predict(dataset)
    if not isinstance(predictions, np.ndarray):
        predictions = predictions.to_numpy()
    data_iter = tfds.as_numpy(dataset)
    labels = np.array([l for _, l in data_iter])
    treatment_mask = np.array([f[treatment_key] for f, _ in data_iter]).astype(bool)
    treatment = UpliftCurveInput(
        predictions[treatment_mask],
        labels[treatment_mask]
    )
    control_mask = np.logical_not(treatment_mask)
    control = UpliftCurveInput(predictions[control_mask], labels[control_mask])
    return compute_uplift_curves_and_metric(
        treatment, 
        control, 
        label_type,
        metric_types
    )


def compute_uplift_curves_and_metric(
    treatment: UpliftCurveInput,
    control: UpliftCurveInput,
    label_type: UpliftLabelType = UpliftLabelType.BINARY,
    metric_types: Optional[Sequence[UpliftMetricType]] = None
    ) -> Tuple[UpliftCurves, UpliftMetrics]:
    """Computes various AUUC-based metrics.

  Args:
    treatment: A UpliftCurveInput containing the information regarding the
      treatment group.
    control: A UpliftCurveInput containing the information regarding the
      control group.
    label_type: An UpliftLabelType enum indicating how to interpret the outcomes
      when computing the metric.
    metric_types: A sequence of desired UpliftMetricType enums indicating the
      types of uplift metrics to return. Defaults to computing the AUUC.

  Returns:
    A UpliftCurves object of requisite curves needed to compute the metrics
    along with an UpliftMetrics object containing the values of the specified
    metrics.
    """
    if metric_types is None:
        metric_types = set([UpliftMetricType.AUUC])
    else:
        metric_types = set(metric_types)

    def compute_cumulative_responses(uplift, outcomes, weights):
        """Computes the distribution and weighted cumulative responses."""
        # Bucketize identical predictions to ensure sorting-order invariance.
        unique_uplift, counts = np.unique(uplift, return_counts=True)
        # np.unique returns the counts of the sorted array in ascending order.
        # However, we want the reverse ordering as uplift is sorted in descending
        # order.
        unique_uplift = unique_uplift[::-1]
        accumated_indices = np.cumsum(counts[::-1]) - 1

        def accumulate_and_norm(x, norm):
            cumulative_x = np.cumsum(x)
            cumulative_x = cumulative_x[accumated_indices]
            normed_cumulative_x = cumulative_x / norm
            normed_cumulative_x = np.concatenate(([0], normed_cumulative_x))
            return normed_cumulative_x

        norm = np.sum(weights)
        weighted_normed_cumulative_outcomes = accumulate_and_norm(
            outcomes * weights, norm
        )
        distribution = accumulate_and_norm(weights, norm)

        return unique_uplift, distribution, weighted_normed_cumulative_outcomes

    (_, treatment_distribution,
     weighted_normed_treatment_cumulative_outcomes) = compute_cumulative_responses(
          treatment.uplift, treatment.outcomes, treatment.weights
    )

    (_, control_distribution,
     weighted_normed_control_cumulative_outcomes) = compute_cumulative_responses(
        control.uplift, control.outcomes, control.weights
    )

    return _compute_curves_and_metric(
        treatment_distribution,
        weighted_normed_treatment_cumulative_outcomes,
        control_distribution,
        weighted_normed_control_cumulative_outcomes,
        label_type, metric_types
    )