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
               sigma: list, device, mode='hard', beta=1.0, distribute='False', sigma_base=0.25, certify_robustness=0.5):
    self.base_classifier = base_classifier
    self.sigma_net = sigma_net
    self.num_classes = num_classes
    self.sigma = sigma
    self.device = device
    self.mode = mode
    self.square_sum = None
    self.ss = 0
    self.beta = beta
    self.distribute = distribute
    self.sigma_base = sigma_base
    self.certify_robustness = certify_robustness

  def certify(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int) -> (int, float):
    self.base_classifier.eval()
    if self.mode == 'both':
      c_hard, c_soft = self.predict(x, n0, batch_size, False)
      o_hard, o_soft = self._sample_noise(x, n, batch_size, False)
      na_hard, na_soft = o_hard[c_hard].item(), o_soft[c_soft].item()
      self.ss = self.square_sum[c_soft]
      pa_hard = self._lower_confidence_bound(na_hard, n, alpha, 'hard')
      pa_soft = self._lower_confidence_bound(na_soft, n, alpha, 'soft')

      c_hard_base, c_soft_base = self.predict(x, n0, batch_size, True)
      o_hard_base, o_soft_base = self._sample_noise(x, n, batch_size, True)
      na_hard_base, na_soft_base = o_hard_base[c_hard_base].item(), o_soft_base[c_soft_base].item()
      self.ss = self.square_sum[c_soft_base]
      pa_hard_base = self._lower_confidence_bound(na_hard_base, n, alpha, 'hard')
      pa_soft_base = self._lower_confidence_bound(na_soft_base, n, alpha, 'soft')

      r_hard_output, r_soft_output = 0.0, 0.0

      if pa_hard < 0.5 and pa_hard_base < 0.5:
        c_hard_output = Smooth.ABSTAIN

      elif pa_hard >= 0.5 and pa_hard_base >= 0.5:
        if self.sigma_net is not None:
          sigma_tmp = self.sigma_net(x=x.unsqueeze(0), mean=self.distribute)
        else:
          if self.distribute == 'False':
            sigma_tmp = self.sigma.mean()
          else:
            p_value = ceil(F.softmax(self.base_classifier(x=x.unsqueeze(0)), dim=1).max(1)[0] * (len(self.sigma) - 1))
            sigma_tmp = self.sigma.sort()[0][p_value]

        r_hard = sigma_tmp * norm.ppf(pa_hard)
        r_hard_base = self.sigma_base * norm.ppf(pa_hard_base)

        if r_hard > self.certify_robustness and r_hard_base > self.certify_robustness:
          if sigma_tmp < self.sigma_base:
            r_hard_output = r_hard
            c_hard_output = c_hard
          else:
            r_hard_output = r_hard_base
            c_hard_output = c_hard_base

        elif r_hard > self.certify_robustness and r_hard_base <= self.certify_robustness:
          r_hard_output = r_hard
          c_hard_output = c_hard

        elif r_hard > self.certify_robustness and r_hard_base <= self.certify_robustness:
          r_hard_output = r_hard_base
          c_hard_output = c_hard_base

        else:
          r_hard_output = torch.max(r_hard, r_hard_base)
          if r_hard > r_hard_base:
            c_hard_output = c_hard
          else:
            c_hard_output = c_hard_base

      elif pa_hard >= 0.5 and pa_hard_base < 0.5:
        if self.sigma_net is not None:
          sigma_tmp = self.sigma_net(x=x.unsqueeze(0), mean=self.distribute)
        else:
          if self.distribute == 'False':
            sigma_tmp = self.sigma.mean()
          else:
            p_value = ceil(F.softmax(self.base_classifier(x=x.unsqueeze(0)), dim=1).max(1)[0] * (len(self.sigma) - 1))
            sigma_tmp = self.sigma.sort()[0][p_value]

        r_hard = sigma_tmp * norm.ppf(pa_hard)
        r_hard_output = r_hard
        c_hard_output = c_hard

      else:
        r_hard_base = self.sigma_base * norm.ppf(pa_hard_base)
        r_hard_output = r_hard_base
        c_hard_output = c_hard_base

      if pa_soft < 0.5 and pa_soft_base < 0.5:
        c_soft_output = Smooth.ABSTAIN

      elif pa_soft >= 0.5 and pa_soft_base >= 0.5:
        if self.sigma_net is not None:
          sigma_tmp = self.sigma_net(x=x.unsqueeze(0), mean=self.distribute)
        else:
          if self.distribute == 'False':
            sigma_tmp = self.sigma.mean()
          else:
            p_value = ceil(F.softmax(self.base_classifier(x=x.unsqueeze(0)), dim=1).max(1)[0] * (len(self.sigma) - 1))
            sigma_tmp = self.sigma.sort()[0][p_value]

        r_soft = sigma_tmp * norm.ppf(pa_soft)
        r_soft_base = self.sigma_base * norm.ppf(pa_soft_base)

        if r_soft > self.certify_robustness and r_soft_base > self.certify_robustness:
          if sigma_tmp < self.sigma_base:
            r_soft_output = r_soft
            c_soft_output = c_soft
          else:
            r_soft_output = r_soft_base
            c_soft_output = c_soft_base

        elif r_soft > self.certify_robustness and r_soft_base <= self.certify_robustness:
          r_soft_output = r_soft
          c_soft_output = c_soft

        elif r_soft > self.certify_robustness and r_soft_base <= self.certify_robustness:
          r_soft_output = r_soft_base
          c_soft_output = c_soft_base

        else:
          r_soft_output = torch.max(r_soft, r_soft_base)
          if r_soft > r_soft_base:
            c_soft_output = c_soft
          else:
            c_soft_output = c_soft_base

      elif pa_soft >= 0.5 and pa_soft_base < 0.5:
        if self.sigma_net is not None:
          sigma_tmp = self.sigma_net(x=x.unsqueeze(0), mean=self.distribute)

        else:
          if self.distribute == 'False':
            sigma_tmp = self.sigma.mean()
          else:
            p_value = ceil(F.softmax(self.base_classifier(x=x.unsqueeze(0)), dim=1).max(1)[0] * (len(self.sigma) - 1))
            sigma_tmp = self.sigma.sort()[0][p_value]
        r_soft = sigma_tmp * norm.ppf(pa_soft)
        r_soft_output = r_soft
        c_soft_output = c_soft

      else:
        r_soft_base = self.sigma_base * norm.ppf(pa_soft_base)
        r_soft_output = r_soft_base
        c_soft_output = c_soft_base
                  
      return c_hard_output, r_hard_output, c_soft_output, r_soft_output

    elif self.mode == 'soft':
      c_soft = self.predict(x, n0, batch_size, False)
      o_soft = self._sample_noise(x, n, batch_size, False)
      na_soft = o_soft[c_soft].item()
      self.ss = self.square_sum[c_soft]
      pa_soft = self._lower_confidence_bound(na_soft, n, alpha, 'soft')

      c_soft_base = self.predict(x, n0, batch_size, True)
      o_soft_base = self._sample_noise(x, n, batch_size, True)
      na_soft_base = o_soft_base[c_soft_base].item()
      self.ss = self.square_sum[c_soft_base]
      pa_soft_base = self._lower_confidence_bound(na_soft_base, n, alpha, 'soft')

      r_soft_output = 0.0

      if pa_soft < 0.5 and pa_soft_base < 0.5:
        c_soft_output = Smooth.ABSTAIN

      elif pa_soft >= 0.5 and pa_soft_base >= 0.5:
        if self.sigma_net is not None:
          sigma_tmp = self.sigma_net(x=x.unsqueeze(0), mean=self.distribute)
        else:
          if self.distribute == 'False':
            sigma_tmp = self.sigma.mean()
          else:
            p_value = ceil(F.softmax(self.base_classifier(x=x.unsqueeze(0)), dim=1).max(1)[0] * (len(self.sigma) - 1))
            sigma_tmp = self.sigma.sort()[0][p_value]

        r_soft = sigma_tmp * norm.ppf(pa_soft)
        r_soft_base = self.sigma_base * norm.ppf(pa_soft_base)

        if r_soft > self.certify_robustness and r_soft_base > self.certify_robustness:
          if sigma_tmp < self.sigma_base:
            r_soft_output = r_soft
            c_soft_output = c_soft
          else:
            r_soft_output = r_soft_base
            c_soft_output = c_soft_base

        elif r_soft > self.certify_robustness and r_soft_base <= self.certify_robustness:
          r_soft_output = r_soft
          c_soft_output = c_soft

        elif r_soft > self.certify_robustness and r_soft_base <= self.certify_robustness:
          r_soft_output = r_soft_base
          c_soft_output = c_soft_base

        else:
          r_soft_output = torch.max(r_soft, r_soft_base)
          if r_soft > r_soft_base:
            c_soft_output = c_soft
          else:
            c_soft_output = c_soft_base

      elif pa_soft >= 0.5 and pa_soft_base < 0.5:
        if self.sigma_net is not None:
          sigma_tmp = self.sigma_net(x=x.unsqueeze(0), mean=self.distribute)

        else:
          if self.distribute == 'False':
            sigma_tmp = self.sigma.mean()
          else:
            p_value = ceil(F.softmax(self.base_classifier(x=x.unsqueeze(0)), dim=1).max(1)[0] * (len(self.sigma) - 1))
            sigma_tmp = self.sigma.sort()[0][p_value]
        r_soft = sigma_tmp * norm.ppf(pa_soft)
        r_soft_output = r_soft
        c_soft_output = c_soft

      else:
        r_soft_base = self.sigma_base * norm.ppf(pa_soft_base)
        r_soft_output = r_soft_base
        c_soft_output = c_soft_base

      return c_soft_output, r_soft_output

    else:
      c_hard = self.predict(x, n0, batch_size, False)
      o_hard = self._sample_noise(x, n, batch_size, False)
      na_hard = o_hard[c_hard].item()
      self.ss = self.square_sum[c_hard]
      pa_hard = self._lower_confidence_bound(na_hard, n, alpha, 'hard')

      c_hard_base = self.predict(x, n0, batch_size, True)
      o_hard_base = self._sample_noise(x, n, batch_size, True)
      na_hard_base = o_hard_base[c_hard_base].item()
      self.ss = self.square_sum[c_hard_base]
      pa_hard_base = self._lower_confidence_bound(na_hard_base, n, alpha, 'hard')

      r_hard_output = 0.0

      if pa_hard < 0.5 and pa_hard_base < 0.5:
        c_hard_output = Smooth.ABSTAIN

      elif pa_hard >= 0.5 and pa_hard_base >= 0.5:
        if self.sigma_net is not None:
          sigma_tmp = self.sigma_net(x=x.unsqueeze(0), mean=self.distribute)
        else:
          if self.distribute == 'False':
            sigma_tmp = self.sigma.mean()
          else:
            p_value = ceil(F.softmax(self.base_classifier(x=x.unsqueeze(0)), dim=1).max(1)[0] * (len(self.sigma) - 1))
            sigma_tmp = self.sigma.sort()[0][p_value]

        r_hard = sigma_tmp * norm.ppf(pa_hard)
        r_hard_base = self.sigma_base * norm.ppf(pa_hard_base)

        if r_hard > self.certify_robustness and r_hard_base > self.certify_robustness:
          if sigma_tmp < self.sigma_base:
            r_hard_output = r_hard
            c_hard_output = c_hard
          else:
            r_hard_output = r_hard_base
            c_hard_output = c_hard_base

        elif r_hard > self.certify_robustness and r_hard_base <= self.certify_robustness:
          r_hard_output = r_hard
          c_hard_output = c_hard

        elif r_hard > self.certify_robustness and r_hard_base <= self.certify_robustness:
          r_hard_output = r_hard_base
          c_hard_output = c_hard_base

        else:
          r_hard_output = torch.max(torch.tensor(r_hard), torch.tensor(r_hard_base))
          if r_hard > r_hard_base:
            c_hard_output = c_hard
          else:
            c_hard_output = c_hard_base

      elif pa_hard >= 0.5 and pa_hard_base < 0.5:
        if self.sigma_net is not None:
          sigma_tmp = self.sigma_net(x=x.unsqueeze(0), mean=self.distribute)
        else:
          if self.distribute == 'False':
            sigma_tmp = self.sigma.mean()
          else:
            p_value = ceil(F.softmax(self.base_classifier(x=x.unsqueeze(0)), dim=1).max(1)[0] * (len(self.sigma) - 1))
            sigma_tmp = self.sigma.sort()[0][p_value]

        r_hard = sigma_tmp * norm.ppf(pa_hard)
        r_hard_output = r_hard
        c_hard_output = c_hard

      else:
        r_hard_base = self.sigma_base * norm.ppf(pa_hard_base)
        r_hard_output = r_hard_base
        c_hard_output = c_hard_base

      return c_hard_output, r_hard_output

  def predict(self, x: torch.tensor, n: int, batch_size: int, sigma_base=False) -> int:
    self.base_classifier.eval()
    if self.mode == 'both':
      result_hard, result_soft = self._sample_noise(x, n, batch_size, sigma_base)
      return result_hard.argsort()[::-1][0], result_soft.argsort()[::-1][0]
    else:
      result = self._sample_noise(x, n, batch_size, sigma_base)
      return result.argsort()[::-1][0]

  def _sample_noise(self, x: torch.tensor, num: int, batch_size, sigma_base=False) -> np.ndarray:
    self.base_classifier.eval()
    with torch.no_grad():
      result_hard = np.zeros(self.num_classes, dtype=int)
      result_soft = np.zeros(self.num_classes, dtype=float)
      self.square_sum = np.zeros(self.num_classes, dtype=float)
      for _ in range(ceil(num / batch_size)):
        this_batch_size = min(batch_size, num)
        num -= this_batch_size
        batch = x.repeat((this_batch_size, 1, 1, 1))

        if sigma_base:
          noise = torch.randn_like(batch, device=self.device) * self.sigma_base

        else:
          if self.sigma_net is not None:
            noise = torch.randn_like(batch, device=self.device) * self.sigma_net(batch, mean=self.distribute).view(this_batch_size, 1, 1, 1)
          else:
            if self.distribute == 'False':
              noise = torch.randn_like(batch, device=self.device) * self.sigma.mean()
            else:
              p_value = ceil(F.softmax(self.base_classifier(x=x.unsqueeze(0)), dim=1).max(1)[0] * (len(self.sigma) - 1))
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
