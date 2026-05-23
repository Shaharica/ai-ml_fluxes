import os
import shap
import matplotlib.pyplot as plt


import shap


def compute_shap_values(model,model_name,X_train,X_test):

    # Tree models (RF, XGBoost)
    if model_name.lower() in ["rf", "xgb"]:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_test)
    # ANN model
    elif model_name.lower() == "ann":
        background = shap.kmeans(X_train, 50)
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(X_test)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return explainer, shap_values

def plot_shap_summary( shap_values, X_test, config, model_name):

    shap.summary_plot( shap_values.values, X_test, show=False)

    path = (
        f"{config.plot_dir}/"
        f"{config.experiment_name}_"
        f"{model_name}_shap_summary.svg"
    )

    plt.savefig(
        path,
        bbox_inches="tight"
    )

    plt.close()

    print(f"[INFO] Saved: {path}")

def plot_shap_beeswarm(
    shap_values,
    config,
    model_name
):

    plt.figure()
    shap.plots.beeswarm(
        shap_values,
        show=False
    )

    path = (
        f"{config.plot_dir}/"
        f"{config.experiment_name}_"
        f"{model_name}_shap_beeswarm.svg"
    )

    plt.savefig(
        path,
        bbox_inches="tight"
    )

    plt.close()

    print(f"[INFO] Saved: {path}")


def plot_shap_dependence(
    shap_values,
    X_test,
    feature_name,
    config,
    model_name
):

    plt.figure()

    shap.dependence_plot(
        feature_name,
        shap_values.values,
        X_test,
        show=False
    )

    path = (
        f"{config.plot_dir}/"
        f"{config.experiment_name}_"
        f"{model_name}_dependence_{feature_name}.svg"
    )

    plt.savefig(
        path,
        bbox_inches="tight"
    )

    plt.close()

    print(f"[INFO] Saved: {path}")

def plot_shap_waterfall(
    shap_values,
    sample_index,
    config,
    model_name
):

    plt.figure()

    shap.plots.waterfall(
        shap_values[sample_index],
        show=False
    )

    path = (
        f"{config.plot_dir}/"
        f"{config.experiment_name}_"
        f"{model_name}_waterfall.svg"
    )

    plt.savefig(
        path,
        bbox_inches="tight"
    )

    plt.close()

    print(f"[INFO] Saved: {path}")

def plot_shap_force(
    explainer,
    shap_values,
    X_test,
    sample_index,
    config,
    model_name
):

    shap.force_plot(
        explainer.expected_value,
        shap_values.values[sample_index],
        X_test.iloc[sample_index],
        matplotlib=True,
        show=False
    )

    path = (
        f"{config.plot_dir}/"
        f"{config.experiment_name}_"
        f"{model_name}_force.svg"
    )

    plt.savefig(
        path,
        bbox_inches="tight"
    )

    plt.close()

    print(f"[INFO] Saved: {path}")


def plot_shap_interaction(
    shap_values,
    X_test,
    feature_name,
    config,
    model_name
):

    plt.figure()

    shap.dependence_plot(
        feature_name,
        shap_values.values,
        X_test,
        interaction_index="auto",
        show=False
    )

    path = (
        f"{config.plot_dir}/"
        f"{config.experiment_name}_"
        f"{model_name}_interaction_{feature_name}.svg"
    )

    plt.savefig(
        path,
        bbox_inches="tight"
    )

    plt.close()
    print(f"[INFO] Saved: {path}")