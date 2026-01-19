import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

class ABTestCalculator:
    def __init__(self, alpha=0.05, bootstrap_iter=5000, bayes_iter=10000, alternative='two-sided', delta=0.0, method='z_test'):
        self.alpha = alpha
        self.bootstrap_iter = bootstrap_iter
        self.bayes_iter = bayes_iter
        self.alternative = alternative
        self.delta = delta
        self.method = method
        self.hypotheses = []

    def register_hypothesis(self, name, expectation='greater', metric='conversion_rate'):
        self.hypotheses.append({
            'name': name,
            'expectation': expectation,
            'metric': metric
        })

    def analyze(self, n_A, conv_A, n_B, conv_B):
        results = {}

        cr_A = conv_A / n_A
        cr_B = conv_B / n_B
        uplift = cr_B - cr_A
        p_pool = (conv_A + conv_B) / (n_A + n_B)
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n_A + 1/n_B))
        z_score = uplift / se

        if self.alternative == 'two-sided':
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        elif self.alternative == 'greater':
            p_value = 1 - stats.norm.cdf(z_score)
        elif self.alternative == 'less':
            p_value = stats.norm.cdf(z_score)
        else:
            raise ValueError("Invalid alternative hypothesis")

        z_crit = stats.norm.ppf(1 - self.alpha / 2)
        ci_z = [uplift - z_crit * se, uplift + z_crit * se]
        significant = (p_value < self.alpha) and (abs(uplift) >= self.delta)

        results['z_test'] = {
            'cr_A': cr_A,
            'cr_B': cr_B,
            'uplift': uplift,
            'z_score': z_score,
            'p_value': p_value,
            'ci': ci_z,
            'significant': significant
        }

        group_A = np.concatenate([np.ones(conv_A), np.zeros(n_A - conv_A)])
        group_B = np.concatenate([np.ones(conv_B), np.zeros(n_B - conv_B)])
        bs_diffs = [
            np.random.choice(group_B, size=n_B, replace=True).mean() -
            np.random.choice(group_A, size=n_A, replace=True).mean()
            for _ in range(self.bootstrap_iter)
        ]
        ci_bs = np.percentile(bs_diffs, [100 * self.alpha / 2, 100 * (1 - self.alpha / 2)])
        results['bootstrap'] = {
            'mean_diff': np.mean(bs_diffs),
            'ci': ci_bs.tolist(),
            'significant': not (ci_bs[0] <= self.delta <= ci_bs[1])
        }

        alpha_A, beta_A = conv_A + 1, n_A - conv_A + 1
        alpha_B, beta_B = conv_B + 1, n_B - conv_B + 1
        samples_A = np.random.beta(alpha_A, beta_A, self.bayes_iter)
        samples_B = np.random.beta(alpha_B, beta_B, self.bayes_iter)
        lift_samples = samples_B - samples_A
        prob_b_better = np.mean(lift_samples > self.delta)
        ci_bayes = np.percentile(lift_samples, [2.5, 97.5])
        results['bayesian'] = {
            'prob_B_better': prob_b_better,
            'mean_diff': np.mean(lift_samples),
            'ci': ci_bayes.tolist()
        }

        sd_pooled = np.sqrt(p_pool * (1 - p_pool))
        results['effect_size'] = {
            'cohens_d': uplift / sd_pooled
        }

        self.results = results
        self.bs_diffs = bs_diffs
        return results

    def summarize(self):
        r = self.results
        h_line = self.hypotheses[-1]['name'] if self.hypotheses else 'H₀: No difference'

        method_label = {
            'z_test': 'Z-test',
            'bootstrap': 'Bootstrap',
            'bayesian': 'Bayesian'
        }.get(self.method, 'Z-test')

        significance = r[self.method]['significant'] if self.method in ['z_test', 'bootstrap'] else r['bayesian'][
                                                                                                        'prob_B_better'] > 0.95

        summary = f"""
    === Hypothesis: {h_line} ===

    → Method used: {method_label}
    → Observed uplift: {r['z_test']['uplift']:.2%}
    → p-value (Z-test): {r['z_test']['p_value']:.4f}
    → Confidence Interval (Z): {r['z_test']['ci'][0]:.2%} — {r['z_test']['ci'][1]:.2%}
    → Confidence Interval (Bootstrap): {r['bootstrap']['ci'][0]:.2%} — {r['bootstrap']['ci'][1]:.2%}
    → Bayesian P(B > A + δ): {r['bayesian']['prob_B_better']:.2%}
    → Significant ({method_label})? {'✅ YES' if significance else '❌ NO'}
    → Cohen's d: {r['effect_size']['cohens_d']:.3f}
    """
        return summary

    def plot_bootstrap(self):
        if not hasattr(self, 'bs_diffs'):
            raise ValueError("Run analyze() before plotting.")
        ci = self.results['bootstrap']['ci']
        plt.hist(self.bs_diffs, bins=50, color='skyblue', edgecolor='black')
        plt.axvline(ci[0], color='red', linestyle='--', label='CI Lower')
        plt.axvline(ci[1], color='green', linestyle='--', label='CI Upper')
        plt.axvline(0, color='black', linestyle=':', label='Zero Effect')
        plt.title('Bootstrap Distribution')
        plt.legend()
        plt.grid(True)
        plt.show()
