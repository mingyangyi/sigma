'''
Randomized Smoothing: Hard-RS and Soft-RS
Soft-RS uses empirical Bernstein bound

MACER: Attack-free and Scalable Robust Training via Maximizing Certified Radius
ICLR 2020 Submission

References:
[1] J. Cohen, E. Rosenfeld and Z. Kolter. 
Certified Adversarial Robustness via Randomized Smoothing. In ICML, 2019.

Acknowledgements:
[1] https://github.com/locuslab/smoothing/blob/master/code/core.py
'''

from math import ceil

import numpy as np
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint
import torch
import torch.nn.functional as F


class Smooth(object):
  '''
  Smoothed classifier
  mode can be hard, soft or both
  beta is the inverse of softmax temperature
  '''

  # to abstain, Smooth returns this int
  ABSTAIN = -1

  def __init__(self, base_classifier: torch.nn.Module, sigma_net: torch.nn.Module, num_classes: int,
               sigma: list, device, mode='hard', beta=1.0):
    self.base_classifier = base_classifier
    self.sigma_net = sigma_net
    self.num_classes = num_classes
    self.sigma = sigma
    self.device = device
    self.mode = mode
    self.square_sum = None
    self.ss = 0
    self.beta = beta

  def certify(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int) -> (int, float):
    self.base_classifier.eval()
    if self.mode == 'both':
      c_hard, c_soft = self.predict(x, n0, batch_size)
      o_hard, o_soft = self._sample_noise(x, n, batch_size)
      na_hard, na_soft = o_hard[c_hard].item(), o_soft[c_soft].item()
      self.ss = self.square_sum[c_soft]
      pa_hard = self._lower_confidence_bound(na_hard, n, alpha, 'hard')
      pa_soft = self._lower_confidence_bound(na_soft, n, alpha, 'soft')
      r_hard = 0.0
      r_soft = 0.0
      if pa_hard < 0.5:
        c_hard = Smooth.ABSTAIN
      else:
        if self.sigma_net is not None:
          r_hard = self.sigma_net(x=x.unsqueeze(0), mean=False) * norm.ppf(pa_hard)
        else:
          p_value = ceil(F.softmax(self.base_classifier(x=x.unsqueeze(0)), dim=1).max(1)[0] * len(self.sigma))
          r_hard = self.sigma.sort()[1][p_value] * norm.ppf(pa_hard)
      if pa_soft < 0.5:
        c_soft = Smooth.ABSTAIN
      else:
        if self.sigma_net is not None:
          r_soft = self.sigma_net(x=x.unsqueeze(0), mean=False) * norm.ppf(pa_soft)
        else:
          p_value = ceil(F.softmax(self.base_classifier(x=x.unsqueeze(0)), dim=1).max(1)[0] * len(self.sigma))
          r_soft = self.sigma.sort()[1][p_value] * norm.ppf(pa_soft)
      return c_hard, r_hard, c_soft, r_soft
    else:
      # make an initial prediction of the label
      cAHat = self.predict(x, n0, batch_size)
      # draw more samples of f(x + epsilon)
      observation = self._sample_noise(x, n, batch_size)
      # use these samples to estimate a lower bound on pA
      nA = observation[cAHat].item()
      if self.mode == 'soft':
        self.ss = self.square_sum[cAHat]
      pABar = self._lower_confidence_bound(nA, n, alpha, self.mode)
      if pABar < 0.5:
        return Smooth.ABSTAIN, 0.0
      else:
        if self.sigma_net is not None:
          radius = self.sigma_net(x=x.unsqueeze(0), mean=False) * norm.ppf(pABar)
        else:
          p_value = ceil(F.softmax(self.base_classifier(x=x.unsqueeze(0)), dim=1).max(1)[0] * len(self.sigma))
          radius = self.sigma.sort()[1][p_value] * norm.ppf(pABar)
        return cAHat, radius

  def predict(self, x: torch.tensor, n: int, batch_size: int) -> int:
    self.base_classifier.eval()
    if self.mode == 'both':
      result_hard, result_soft = self._sample_noise(x, n, batch_size)
      return result_hard.argsort()[::-1][0], result_soft.argsort()[::-1][0]
    else:
      result = self._sample_noise(x, n, batch_size)
      return result.argsort()[::-1][0]

  def _sample_noise(self, x: torch.tensor, num: int, batch_size) -> np.ndarray:
    self.base_classifier.eval()
    with torch.no_grad():
      result_hard = np.zeros(self.num_classes, dtype=int)
      result_soft = np.zeros(self.num_classes, dtype=float)
      self.square_sum = np.zeros(self.num_classes, dtype=float)
      for _ in range(ceil(num / batch_size)):
        this_batch_size = min(batch_size, num)
        num -= this_batch_size

        batch = x.repeat((this_batch_size, 1, 1, 1))
        if self.sigma_net is not None:
          noise = torch.randn_like(batch, device=self.device) * self.sigma_net(batch, mean=False).view(this_batch_size, 1, 1, 1)
        else:
          p_value = ceil(F.softmax(self.base_classifier(x=x.unsqueeze(0)), dim=1).max(1)[0] * len(self.sigma))
          noise = torch.randn_like(batch, device=self.device) * self.sigma.sort()[0][p_value]
        predictions = self.base_classifier(batch + noise)
        predictions *= self.beta
        if self.mode == 'hard' or self.mode == 'both':
          p_hard = predictions.argmax(1)
          result_hard += self._count_arr(p_hard.cpu().numpy(),
                                         self.num_classes)
        if self.mode == 'soft' or self.mode == 'both':
          p_soft = F.softmax(predictions, 1)
          p_soft_square = p_soft ** 2
          p_soft = p_soft.sum(0)
          p_soft_square = p_soft_square.sum(0)
          result_soft += p_soft.cpu().numpy()
          self.square_sum += p_soft_square.cpu().numpy()
      if self.mode == 'hard':
        return result_hard
      if self.mode == 'soft':
        return result_soft
      else:
        return result_hard, result_soft

  def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
    counts = np.zeros(length, dtype=int)
    for idx in arr:
      counts[idx] += 1
    return counts

  def _lower_confidence_bound(self, NA, N, alpha: float, mode) -> float:
    if mode == 'hard':
      return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]
    else:
      sample_variance = (self.ss - NA * NA / N) / (N - 1)
      if sample_variance < 0:
        sample_variance = 0
      t = np.log(2 / alpha)
      return NA / N - np.sqrt(2 * sample_variance * t / N) - 7 * t / 3 / (N - 1)
