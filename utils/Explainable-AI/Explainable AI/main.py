from config import Config
from preprocessing import (load_data,preprocess_data)
from train import ( train_rf, train_xgb, train_ann)
from evaluate import evaluate_model
from LIME import ( create_lime_explainer,compute_lime_explanation,save_lime_html,save_lime_feature_table)
from SHAP import  (compute_shap_values,plot_shap_summary,plot_shap_beeswarm,plot_shap_dependence,plot_shap_waterfall,plot_shap_force,plot_shap_interaction)
from feature_importance import(feature_imp,plot_feature_importance,save_feature_importance)
config = Config()

X, y = load_data("data/data.csv", config)

(X_train,X_test,y_train,y_test) = preprocess_data(X, y, config)

for model_name in config.models:

    print(f"\nTraining {model_name}")

    # RF
    if model_name == "RF":

        model = train_rf(
            X_train,
            y_train
        )

        metrics = evaluate_model(
            model,
            X_test,
            y_test
        )

    # XGB
    elif model_name == "XGB":
        model = train_xgb( X_train, y_train)
        metrics = evaluate_model(model,X_test,y_test)

    # ANN
    elif model_name == "ANN":
        model = train_ann(X_train, y_train, config )
        metrics = evaluate_model(
            model,
            X_test,
            y_test
        )

    print(metrics)

    # LIME
    lime_explainer = create_lime_explainer( X_train)
    lime_explanation = compute_lime_explanation(lime_explainer,model,X_test,sample_index=10)
    save_lime_html(lime_explanation,config,model_name)
    save_lime_feature_table(lime_explanation,config,model_name)

    #feature importance
    df = feature_imp(model,X_test,y_test,model_name,config)
    save_feature_importance(df,model_name,config)
    plot_feature_importance(df,model_name,config)

    #shap
    print("before shap")
    shap_explainer, shap_values = compute_shap_values(model=model,model_name=model_name,X_train=X_train,X_test=X_test)
    print("after")
    plot_shap_summary(shap_values=shap_values, X_test=X_test, config=config, model_name=model_name)
    plot_shap_beeswarm(shap_values=shap_values,config=config,model_name=model_name)
    plot_shap_dependence(shap_values=shap_values,X_test=X_test,feature_name=X_test.columns[0],config=config,model_name=model_name)
    plot_shap_waterfall(shap_values=shap_values,sample_index=0,config=config,model_name=model_name)
    plot_shap_force( explainer=shap_explainer, shap_values=shap_values, X_test=X_test, sample_index=0, config=config, model_name=model_name)
    plot_shap_interaction(shap_values=shap_values,X_test=X_test,feature_name=X_test.columns[0],config=config, model_name=model_name)



