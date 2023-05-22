import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import shap

#from Utils import Utils

def plot_roc(
        clf: xgb.XGBClassifier, 
        y_train: np.ndarray, 
        y_pred_train: np.ndarray, 
        y_test: np.ndarray,
        y_pred_test: np.ndarray
    ) -> None:

    loss_eff_train, win_eff_train, _ = roc_curve(y_train, y_pred_train)
    loss_eff_test, win_eff_test, _ = roc_curve(y_test, y_pred_test)
    
    fig = plt.figure()
    axes = fig.gca()

    axes.plot(
        loss_eff_train, 
        win_eff_train, 
        color='red', 
        label='Train set'
        )

    axes.plot(
        loss_eff_test, 
        win_eff_test, 
        color='royalblue', 
        label='Test set'
        )

    axes.plot(
        loss_eff_test, 
        win_eff_test, 
        color='royalblue', 
        label='Test set'
        )
    
    axes.plot(
        np.linspace(0,1,100),
        np.linspace(0,1,100), 
        linestyle="--", 
        color='black', 
        zorder=0, 
        label="Random classifier"
    )

    axes.set_xlabel('False positive rate', ha='right', x=1, size=13)
    axes.set_xlim((0,1))
    axes.set_ylabel('True positive rate', ha='right', y=1, size=13)
    axes.set_ylim((0,1))
    axes.legend(bbox_to_anchor=(0.97,0.28))
    axes.grid(True, 'major', linestyle='solid', color='grey', alpha=0.5)
    fig.savefig('plots/ROC_curve.pdf')
    plt.close() 


def plot_confusion_matrix(
        y_test: np.ndarray, 
        y_pred_test_class: np.ndarray
    ) -> None:

    trues_preds_test = pd.DataFrame({'actual':y_test, 
                                     'prediction':y_pred_test_class}
                                   )
    c_matrix = pd.crosstab(
        index=trues_preds_test['actual'], 
        columns=trues_preds_test['prediction'], 
        normalize='columns'
    )

    fig = sns.heatmap(c_matrix, 
        vmin=0, 
        vmax=1, 
        annot=True, 
        cmap='viridis'
    )

    fig.get_figure().savefig('plots/confusion_matrix.pdf')
    plt.close() 


def plot_input(
        var: str, 
        y_true: np.ndarray, 
        normalise: bool=False, 
        log: bool=False
    ) -> None:

    fig  = plt.figure()
    axes = fig.gca()

    var_sig = df[var][y_true==1]
    var_bkg = df[var][y_true==0]

    axes.hist(var_sig, bins=30, label='True wins', histtype='step', color='forestgreen', density=normalise)
    axes.hist(var_bkg, bins=30, label='True Losses/Draw', histtype='step', color='firebrick', density=normalise)

    axes.legend(bbox_to_anchor=(0.9,0.97))

    #change axes limits
    if log:
        axes.set_yscale('log', nonposy='clip')

    fig.savefig(f'plots/input_{var}.pdf')
    plt.close()


def plot_output_score(
        y_test: np.ndarray, 
        y_pred_test: np.ndarray, 
        normalise: bool=False, 
        log: bool=False
    ) -> None:

    fig  = plt.figure()
    axes = fig.gca()

    bins = np.linspace(0,1,31)

    sig_scores = y_pred_test[y_test==1]
    bkg_scores = y_pred_test[y_test==0]

    axes.hist(sig_scores, bins=bins, label='True Wins', histtype='step', color='forestgreen', density=normalise)
    axes.hist(bkg_scores, bins=bins, label='True Losses/Draw', histtype='step', color='firebrick', density=normalise)

    axes.legend(bbox_to_anchor=(0.9,0.97))
    if normalise: axes.set_ylabel('Arbitrary Units', ha='right', y=1)
    else: 
        x_err = abs(bins[-1] - bins[-2])
        axes.set_ylabel('Events / {0:.2g}'.format(x_err) , size=14, ha='right', y=1)

    axes.set_xlabel('BDT Score', ha='right', x=1)

    if log: 
        axes.set_yscale('log', nonposy='clip')

    fig.savefig('plots/output_score.pdf')
    plt.close() 


def plot_shaps(
    clf: xgb.XGBClassifier, 
    x_test: np.ndarray, 
    train_vars: list[str]
    ) -> None:

    #shap plots
    explainer = shap.Explainer(clf)
    shap_values = explainer(pd.DataFrame(x_test, columns=train_vars))

    plt.rcParams.update({'text.usetex':'false'})
    plt.figure()
    shap.plots.beeswarm(shap_values, show=False)
    plt.tight_layout()
    plt.savefig('plots/shapley_beeswarm.pdf', bbox_inches="tight")
    print(' --> Saved plot: plots/shapley_beeswarm.pdf')
    plt.close()

    plt.figure()
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    plt.savefig('plots/shapley_bar_chart.pdf', bbox_inches="tight")
    print(' --> Saved plot: plots/shapley_bar_chart.pdf')
    plt.close() 

