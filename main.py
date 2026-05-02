import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from data.data_generator import generate_market_data
from utils.preprocessor import Preprocessor
from utils.metrics import evaluate
from utils.visualizer import (
    plot_raw_data,
    plot_forecasts,
    plot_feature_importance,
    plot_model_comparison,
    plot_nbeats_decomposition,
)

from models.linear_regression_model import LRModel
from models.random_forest_model import RFModel
from models.nbeats_model import NBEATSModel

from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model_bert = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embeddings(text_series):
    embeddings = []
    for text in text_series:
        inputs = tokenizer(str(text), return_tensors='pt', truncation=True, padding=True)
        outputs = model_bert(**inputs)
        emb = outputs.last_hidden_state[:, 0, :].detach().numpy()
        embeddings.append(emb[0])
    return np.array(embeddings)


def main():
    print("\n" + "═" * 65)
    print("   HYBRID FORECASTING — AUTO DATA + BERT + N-BEATS")
    print("═" * 65 + "\n")

    try:
        df_raw = pd.read_csv("data/my_dataset.csv")
        if df_raw.empty:
            raise ValueError("Dataset is empty")
        print("External dataset loaded")
        df_raw["date"] = pd.to_datetime(df_raw["date"])
        df_raw.set_index("date", inplace=True)
    except:
        df_raw = generate_market_data(n_days=730, save=True)

    if "sentiment" not in df_raw.columns:
        df_raw["sentiment"] = np.random.uniform(-1, 1, len(df_raw))

    if "promo" not in df_raw.columns:
        df_raw["promo"] = 0

    if "price" not in df_raw.columns:
        df_raw["price"] = 50

    plot_raw_data(df_raw)

    prep = Preprocessor(target_col="demand", test_size=0.1)
    df = prep.build_features(df_raw)

    num_cols = prep.num_feature_cols()
    hybrid_cols = prep.hybrid_feature_cols()

    num_cols = [c for c in num_cols if c in df.columns]
    hybrid_cols = [c for c in hybrid_cols if c in df.columns]

    if "news" not in df.columns:
        df["news"] = df["sentiment"].apply(
            lambda x: "market is growing" if x > 0 else "market is declining"
        )

    bert_features = get_bert_embeddings(df["news"])
    bert_features = np.array(bert_features)

    if len(bert_features.shape) == 1:
        bert_features = bert_features.reshape(1, -1)

    if bert_features.shape[0] != len(df):
        bert_features = np.tile(bert_features, (len(df), 1))

    bert_df = pd.DataFrame(
        bert_features,
        index=df.index,
        columns=[f"bert_{i}" for i in range(bert_features.shape[1])]
    )

    df = pd.concat([df, bert_df], axis=1)
    hybrid_cols.extend(bert_df.columns.tolist())

    X_train_num, X_test_num, y_train, y_test = prep.split(df, num_cols)
    X_train_hyb, X_test_hyb, _, _ = prep.split(df, hybrid_cols)

    test_index = df.index[-len(y_test):]

    results = []

    lr_model = LRModel()
    lr_model.fit(X_train_hyb, y_train, feature_names=hybrid_cols)
    y_pred_lr = lr_model.predict(X_test_hyb)
    results.append(evaluate(y_test, y_pred_lr, "Linear Regression"))

    rf_base = RFModel()
    rf_base.fit(X_train_num, y_train, feature_names=num_cols)
    y_pred_rf_base = rf_base.predict(X_test_num)
    results.append(evaluate(y_test, y_pred_rf_base, "RF Baseline"))

    rf_hyb = RFModel()
    rf_hyb.fit(X_train_hyb, y_train, feature_names=hybrid_cols)
    y_pred_rf_hyb = rf_hyb.predict(X_test_hyb)
    results.append(evaluate(y_test, y_pred_rf_hyb, "RF Hybrid (BERT)"))

    HORIZON = 30
    series = df["demand_scaled"].values
    dates_all = df.index

    nbeats = NBEATSModel(horizon=HORIZON, max_steps=200)
    nbeats.fit(series, dates=dates_all)

    y_pred_nbeats = nbeats.predict()
    y_true_nbeats = nbeats.ground_truth()

    results.append(evaluate(y_true_nbeats, y_pred_nbeats, "N-BEATS"))

    plot_nbeats_decomposition(series, nbeats, dates_all)

    plot_forecasts(
        test_index=test_index,
        y_test=y_test,
        predictions={
            "Linear Regression": y_pred_lr,
            "RF Baseline": y_pred_rf_base,
            "RF Hybrid (BERT)": y_pred_rf_hyb,
        },
        title="Model Comparison",
        filename="forecast_comparison.png",
    )

    plot_model_comparison(results)

    print("\nDONE")


if __name__ == "__main__":
    main()