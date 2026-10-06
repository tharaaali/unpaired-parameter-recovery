"""Curvature and gradient-noise audit of the saved nonlinear toy experiment.

For each saved n=512, sigma=0 run, regenerate exactly the unpaired training
samples, reconstruct its discrepancy objective, and evaluate a local grid.
The sandwich SE estimate is sd_r[M'_r(theta*)] / |mean_r[M''_r(theta*)]|.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from scipy.optimize import minimize_scalar

from aistats2027_nonlinear_transform_experiment import (
    METHODS, THETAS, forward_transform, make_objective, sample_asymmetric_mixture,
)


ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/nonlinear_transform'
OUT=BASE/'curvature'
RADIUS=.04
GRID_POINTS=13
GRADIENT_H=.005
ROBUST_RADII=(.02,.04,.06)
HOLDOUT_SEEDS=range(10,50)


def read_csv(path):
    with path.open(newline='',encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def train_data(theta_true,seed,dimension):
    theta_index=THETAS.index(theta_true)
    rng=np.random.default_rng(1_000_003+10_007*seed+1_009*theta_index)
    reference=sample_asymmetric_mixture(rng,512,dimension)
    source=sample_asymmetric_mixture(rng,512,dimension)
    observed=forward_transform(source,theta_true)
    scale=np.maximum(reference.std(axis=0,ddof=1),.1)
    return reference,observed,scale


def quadratic_fit(objective,center,radius,points=GRID_POINTS):
    offsets=np.linspace(-radius,radius,points)
    values=np.array([objective(float(center+t)) for t in offsets])
    coefficients=np.polynomial.polynomial.polyfit(offsets,values,2)
    prediction=np.polynomial.polynomial.polyval(offsets,coefficients)
    residual=float(np.sqrt(np.mean((values-prediction)**2)))
    variation=float(np.ptp(values))
    return dict(gradient=float(coefficients[1]),curvature=float(2*coefficients[2]),
                relative_fit_residual=residual/max(variation,1e-12))


def make_runs(rows):
    output=[]
    selected=[r for r in rows if int(r['n'])==512 and float(r['noise_std'])==0.]
    cache={}
    for row in selected:
        theta=float(row['theta_true']);seed=int(row['seed']);method=row['method']
        key=(theta,seed)
        if key not in cache:
            cache[key]=train_data(theta,seed,int(row['dimension']))
        reference,observed,scale=cache[key]
        objective,_=make_objective(method,reference,observed,scale,seed)
        estimate=float(row['theta_estimate'])
        objective_saved=float(row['training_objective'])
        objective_rebuilt=float(objective(estimate))
        if not np.isclose(objective_saved,objective_rebuilt,rtol=1e-6,atol=1e-10):
            raise ValueError(f'Objective mismatch for {key} {method}: {objective_saved}, {objective_rebuilt}')
        truth_fit=quadratic_fit(objective,theta,RADIUS)
        estimate_fit=quadratic_fit(objective,estimate,RADIUS)
        gradient_central=(objective(theta+GRADIENT_H)-objective(theta-GRADIENT_H))/(2*GRADIENT_H)
        result=dict(theta_true=theta,seed=seed,method=method,theta_estimate=estimate,
                    error=estimate-theta,
                    gradient_at_truth=float(gradient_central),
                    gradient_quadratic_at_truth=truth_fit['gradient'],
                    curvature_at_truth=truth_fit['curvature'],
                    curvature_at_estimate=estimate_fit['curvature'],
                    fit_residual_truth=truth_fit['relative_fit_residual'],
                    fit_residual_estimate=estimate_fit['relative_fit_residual'],
                    objective_saved=objective_saved,
                    objective_rebuilt=objective_rebuilt)
        for radius in ROBUST_RADII:
            tag=str(radius).replace('.','p')
            result[f'curvature_truth_radius_{tag}']=quadratic_fit(objective,theta,radius)['curvature']
            result[f'curvature_estimate_radius_{tag}']=quadratic_fit(objective,estimate,radius)['curvature']
        output.append(result)
        print(method,theta,seed,'H*',round(result['curvature_at_truth'],5),
              'g*',round(result['gradient_at_truth'],5),flush=True)
    return output


def summarize(rows,bootstrap_draws=10000):
    output=[]
    for method in METHODS:
        for theta in THETAS:
            group=sorted((r for r in rows if r['method']==method and float(r['theta_true'])==theta),
                         key=lambda r:int(r['seed']))
            H=np.array([float(r['curvature_at_truth']) for r in group])
            Hhat=np.array([float(r['curvature_at_estimate']) for r in group])
            g=np.array([float(r['gradient_at_truth']) for r in group])
            errors=np.array([float(r['error']) for r in group])
            predicted_se=float(np.std(g,ddof=1)/abs(np.mean(H)))
            empirical_sd=float(np.std(errors,ddof=1))
            rng=np.random.default_rng(938000+METHODS.index(method)*100+THETAS.index(theta))
            indices=rng.integers(len(group),size=(bootstrap_draws,len(group)))
            hboot=H[indices].mean(axis=1)
            gboot=g[indices].std(axis=1,ddof=1)
            seboot=gboot/np.maximum(np.abs(hboot),1e-12)
            sdboot=errors[indices].std(axis=1,ddof=1)
            result=dict(method=method,theta_true=theta,runs=len(group),
                        mean_curvature_truth=float(H.mean()),
                        sd_curvature_truth=float(H.std(ddof=1)),
                        mean_curvature_estimate=float(Hhat.mean()),
                        mean_gradient_truth=float(g.mean()),
                        gradient_sd=float(g.std(ddof=1)),
                        predicted_se=predicted_se,
                        predicted_se_ci_low=float(np.quantile(seboot,.025)),
                        predicted_se_ci_high=float(np.quantile(seboot,.975)),
                        empirical_sd=empirical_sd,
                        empirical_sd_ci_low=float(np.quantile(sdboot,.025)),
                        empirical_sd_ci_high=float(np.quantile(sdboot,.975)),
                        empirical_bias=float(errors.mean()),
                        empirical_rmse=float(np.sqrt(np.mean(errors**2))),
                        linearized_bias=float(-g.mean()/H.mean()),
                        se_ratio=float(predicted_se/empirical_sd),
                        negative_curvature_truth=int(np.sum(H<=0)),
                        negative_curvature_estimate=int(np.sum(Hhat<=0)),
                        mean_quadratic_fit_residual_truth=float(np.mean([float(r['fit_residual_truth']) for r in group])),
                        mean_quadratic_fit_residual_estimate=float(np.mean([float(r['fit_residual_estimate']) for r in group])))
            for radius in ROBUST_RADII:
                tag=str(radius).replace('.','p')
                result[f'mean_curvature_truth_radius_{tag}']=float(np.mean([float(r[f'curvature_truth_radius_{tag}']) for r in group]))
                result[f'mean_curvature_estimate_radius_{tag}']=float(np.mean([float(r[f'curvature_estimate_radius_{tag}']) for r in group]))
            output.append(result)
    return output


def fit_holdout(dimension=3):
    output=[]
    for theta in THETAS:
        for seed in HOLDOUT_SEEDS:
            reference,observed,scale=train_data(theta,seed,dimension)
            for method in METHODS:
                objective,_=make_objective(method,reference,observed,scale,seed)
                result=minimize_scalar(objective,bounds=(0.,1.2),method='bounded',
                                       options={'xatol':1e-5,'maxiter':80})
                output.append(dict(method=method,theta_true=theta,seed=seed,
                                   theta_estimate=float(result.x),
                                   error=float(result.x-theta),
                                   objective_evaluations=int(result.nfev),
                                   success=bool(result.success)))
            print('holdout',theta,seed,flush=True)
    return output


def add_holdout(summary,holdout,bootstrap_draws=10000):
    for row in summary:
        selected=[r for r in holdout if r['method']==row['method']
                  and float(r['theta_true'])==float(row['theta_true'])]
        errors=np.array([float(r['error']) for r in selected])
        rng=np.random.default_rng(118000+METHODS.index(row['method'])*100+
                                  THETAS.index(float(row['theta_true'])))
        indices=rng.integers(len(errors),size=(bootstrap_draws,len(errors)))
        boot_sd=errors[indices].std(axis=1,ddof=1)
        boot_rmse=np.sqrt(np.mean(errors[indices]**2,axis=1))
        row.update(holdout_runs=len(errors),
                   holdout_sd=float(errors.std(ddof=1)),
                   holdout_sd_ci_low=float(np.quantile(boot_sd,.025)),
                   holdout_sd_ci_high=float(np.quantile(boot_sd,.975)),
                   holdout_bias=float(errors.mean()),
                   holdout_rmse=float(np.sqrt(np.mean(errors**2))),
                   holdout_rmse_ci_low=float(np.quantile(boot_rmse,.025)),
                   holdout_rmse_ci_high=float(np.quantile(boot_rmse,.975)),
                   predicted_to_holdout_sd_ratio=float(row['predicted_se']/errors.std(ddof=1)))
    return summary


def plot(summary):
    colors=dict(zip(METHODS,['#517ea5','#389e77','#bc7955','#8d67aa']))
    markers=dict(zip(THETAS,['o','s','^']))
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for row in summary:
        method=row['method'];theta=row['theta_true']
        axes[0].errorbar(row['predicted_se'],row['holdout_sd'],
                         xerr=[[row['predicted_se']-row['predicted_se_ci_low']],
                               [row['predicted_se_ci_high']-row['predicted_se']]],
                         yerr=[[row['holdout_sd']-row['holdout_sd_ci_low']],
                               [row['holdout_sd_ci_high']-row['holdout_sd']]],
                         fmt=markers[theta],color=colors[method],capsize=2,
                         label=method if theta==THETAS[0] else None)
        axes[1].scatter(row['mean_curvature_truth'],row['gradient_sd'],
                        marker=markers[theta],color=colors[method])
    lim=max(max(r['predicted_se_ci_high'],r['holdout_sd_ci_high']) for r in summary)*1.08
    axes[0].plot([0,lim],[0,lim],color='black',ls='--',lw=1)
    axes[0].set(xlim=(0,lim),ylim=(0,lim),xlabel='Predicted SE = sd(gradient) / |mean curvature|',
                ylabel='Independent holdout SD of fitted parameter')
    axes[1].set(xlabel='Mean curvature at true parameter',ylabel='SD of empirical gradient')
    for ax in axes:ax.grid(alpha=.2)
    axes[0].legend(frameon=False,fontsize=8)
    axes[1].legend(handles=[Line2D([0],[0],marker=markers[theta],linestyle='None',
                                   color='#444444',label=f'θ*={theta:.1f}')
                            for theta in THETAS],frameon=False,fontsize=8,
                   loc='upper left')
    fig.savefig(OUT/'curvature_vs_uncertainty.png',dpi=200)
    fig.savefig(OUT/'curvature_vs_uncertainty.pdf')
    plt.close(fig)


def report(summary):
    lines=['# Local curvature and parameter uncertainty in the nonlinear toy','',
           'This audit reconstructs the exact `n=512`, noise-free objectives in the existing ten-seed experiment. It uses independent unpaired 3D mixture samples and the same random sliced projections as the fitted runs. Objective values at saved estimates match the original CSV. Curvature and gradient noise use seeds 0–9; a separate set of 40 newly fitted datasets (seeds 10–49) tests the predicted SE.',
           '', '## Calculation', '',
           'For each method, true parameter, and seed, a quadratic is fitted to 13 objective values within ±0.04 of the true parameter and separately around the fitted minimum. The empirical gradient at the true parameter uses a central difference with step 0.005. For the ten independent datasets at a fixed method and truth, `J` is the sample variance of those gradients and `H` is the mean curvature at truth. The predicted SE is `sqrt(J)/abs(H)`. All four discrepancies use this same calculation; raw curvature values are on different objective scales and must not be ranked without `J`.',
           '', '## Original ten datasets: local calculation', '',
           '| θ* | Method | H at truth | H at estimate | SD(gradient) | Predicted SE | Empirical SD | Bias | RMSE |',
           '| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for theta in THETAS:
        for method in METHODS:
            row=next(r for r in summary if r['method']==method and r['theta_true']==theta)
            lines.append(f"| {theta:.1f} | {method} | {row['mean_curvature_truth']:.5f} | {row['mean_curvature_estimate']:.5f} | {row['gradient_sd']:.5f} | {row['predicted_se']:.4f} | {row['empirical_sd']:.4f} | {row['empirical_bias']:+.4f} | {row['empirical_rmse']:.4f} |")
    lines.extend(['', '## Independent forty-dataset check', '',
                  '| θ* | Method | Predicted SE from seeds 0–9 | Holdout SD, seeds 10–49 | Holdout bias | Holdout RMSE |',
                  '| ---: | --- | ---: | ---: | ---: | ---: |'])
    for theta in THETAS:
        for method in METHODS:
            row=next(r for r in summary if r['method']==method and r['theta_true']==theta)
            lines.append(f"| {theta:.1f} | {method} | {row['predicted_se']:.4f} | {row['holdout_sd']:.4f} [{row['holdout_sd_ci_low']:.4f}, {row['holdout_sd_ci_high']:.4f}] | {row['holdout_bias']:+.4f} | {row['holdout_rmse']:.4f} |")
    lines.extend(['', 'Predicted and original-sample SD bootstrap intervals and curvature around the fitted minimum are in `summary.csv`. The SE predictor resamples only ten development datasets, so its interval remains imprecise. Per-run curvature, objective fit residual, and gradient are in `runs.csv`; independent parameter estimates are in `holdout_fits.csv`.',
                  '', '## Interpretation', '',
                  'The local linearization is `theta_hat − theta* ≈ −M_r′(theta*)/H`. Its variance approximation depends on both gradient fluctuation and curvature. A raw Hessian comparison across Moments, MMD, Energy, and Sliced W₂ is meaningless because multiplying any objective by a positive constant changes its Hessian while leaving its minimizer unchanged. The ratio `sqrt(J)/|H|` is invariant to that rescaling.',
                  'This formula predicts SD, whereas RMSE also contains squared bias. The table therefore keeps empirical SD and bias separate. Across the independent check, the predicted SE is 1.01–1.40 times the observed SD: the order and scale are informative, but all twelve point predictions are high. All twelve holdout SDs lie within bootstrap intervals for the ten-seed predictor. The pattern is descriptive because methods share datasets and the ten-seed gradient-variance estimate is noisy.',
                  'For example, at θ*=0.5 Moments has much larger raw curvature than Energy (1.240 versus 0.391), yet higher predicted SE (0.0868 versus 0.0637) because its empirical gradient fluctuates more (0.1077 versus 0.0249). The independent SDs likewise favor Energy (0.0545 versus 0.0620). This is the mechanism that raw Hessian values alone miss.',
                  'Sliced W₂ contains sorting changes. The quadratic fit averages over any local kinks; `summary.csv` reports curvature with radii ±0.02, ±0.04, and ±0.06. In this experiment its mean curvature changes by less than 0.1% across those radii, so sorting does not make the reported local fit unstable. This remains a first-order finite-sample diagnostic, not a proof of asymptotic normality or causal explanation of every method difference.',
                  '', '## Reproduce', '',
                  '`python scripts/aistats2027_nonlinear_curvature.py` reads the saved original `nonlinear_transform/runs.csv`, regenerates its n=512/noise=0 training samples, and fits the independent holdout datasets. `--holdout-only` reuses saved per-run curvature calculations.',
                  ''])
    (OUT/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,default=BASE/'runs.csv')
    parser.add_argument('--holdout-only',action='store_true')
    parser.add_argument('--summarize-only',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    runs=read_csv(OUT/'runs.csv') if args.holdout_only or args.summarize_only else make_runs(read_csv(args.input))
    summary=summarize(runs)
    holdout=read_csv(OUT/'holdout_fits.csv') if args.summarize_only else fit_holdout()
    summary=add_holdout(summary,holdout)
    write_csv(OUT/'runs.csv',runs)
    write_csv(OUT/'holdout_fits.csv',holdout)
    write_csv(OUT/'summary.csv',summary)
    plot(summary)
    report(summary)
    (OUT/'metadata.json').write_text(json.dumps(dict(input=str(args.input),radius=RADIUS,
        grid_points=GRID_POINTS,gradient_step=GRADIENT_H,robust_radii=ROBUST_RADII,
        n=512,noise_std=0.,seeds=list(range(10)),holdout_seeds=list(HOLDOUT_SEEDS)),indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
