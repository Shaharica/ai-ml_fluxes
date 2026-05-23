import os
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

def create_lime_explainer(X_train):
    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        mode="regression" )
    return explainer


def compute_lime_explanation( explainer, model, X_test,sample_index):

    explanation = explainer.explain_instance( X_test.iloc[sample_index].values,model.predict )
    return explanation


def save_lime_html( explanation, config, model_name):


    path = (
        f"{config.plot_dir}/"
        f"{config.experiment_name}_"
        f"{model_name}_lime.html"
    )

    explanation.save_to_file(path)

    print(f"[INFO] Saved: {path}")

def save_lime_feature_table( explanation,config, model_name):

    contributions = explanation.as_list()

    df = pd.DataFrame(
        contributions,
        columns=[
            "feature",
            "contribution"
        ]
    )

    path = (
        f"{config.save_dir}/"
        f"{config.experiment_name}_"
        f"{model_name}_lime_feature_table.csv"
    )

    df.to_csv(path, index=False )

    print(f"[INFO] Saved: {path}")
