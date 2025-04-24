

import marimo

__generated_with = "0.13.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    return mo, pd, plt, sns, train_test_split


@app.cell
def _(mo):
    mo.md(
        r"""
        # US Domestic Flights Delay Prediction (2013-2018)
        Analysis by Dimitrios Haralampopoulos, Emma Hodor, Mihir Savalia, and Samuel Miller

        ### Dataset Description
        You work for a travel booking website that wants to improve the customer experience for flights that were delayed. The company wants to create a feature to let customers know if the flight will be delayed because of weather when they book a flight to or from the busiest airports for domestic travel in the US.

        You are tasked with solving part of this problem by using machine learning (ML) to identify whether the flight will be delayed because of weather. You have been given access to the a dataset about the on-time performance of domestic flights that were operated by large air carriers. You can use this data to train an ML model to predict if the flight is going to be delayed for the busiest airports.

        ## Division of Labor
        * Dimitri - ANN and Naive Bayes
        * Emma - C5.0 Decision Tree and KNN
        * Mihir - EDA and Random Forest
        * Sam - ??? and ???
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""# Loading Data""")
    return


@app.cell
def _(pd):
    df = pd.read_csv("./flight_delay_predict.csv")
    df
    return (df,)


@app.cell
def _(mo):
    mo.md(r"""# EDA""")
    return


@app.cell
def _(mo):
    mo.md(r"""# ANN""")
    return


@app.cell
def _():
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import datetime
    return (
        MLPClassifier,
        StandardScaler,
        accuracy_score,
        datetime,
        f1_score,
        precision_score,
        recall_score,
    )


@app.cell
def _(df, pd):
    target = df["is_delay"]
    ann_df = df.iloc[:, 1::]
    ann_df = pd.get_dummies(data=ann_df, columns=["Year", "Reporting_Airline", "Origin", "OriginState", "Dest", "DestState"])
    dummy_cols = list(ann_df.columns[13::])

    def bool2int(b):
        return 1 if b else 0

    ann_df[dummy_cols] = ann_df[dummy_cols].map(bool2int)

    target, ann_df
    return ann_df, target


@app.cell
def _(ann_df, datetime):
    def to_integer(dt_time):
        dt_time = datetime.date.fromisoformat(dt_time)
        return 10000*dt_time.year + 100*dt_time.month + dt_time.day

    ann_df["FlightDate"] = ann_df["FlightDate"].map(to_integer)

    ann_df
    return


@app.cell
def _(StandardScaler, ann_df, pd, target, train_test_split):
    scaler = StandardScaler()
    ann_df_scaled = pd.DataFrame(scaler.fit_transform(ann_df), columns=ann_df.columns)

    trainX, testX, trainY, testY = train_test_split(ann_df_scaled, target, random_state=42, test_size=0.2)
    print(trainX.shape)
    print(trainY.shape)
    return testX, testY, trainX, trainY


@app.cell
def _(
    MLPClassifier,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    testX,
    testY,
    trainX,
    trainY,
):
    ann_model = MLPClassifier(hidden_layer_sizes=(trainX.shape[1], ), max_iter=10000)
    ann_model.fit(trainX, trainY)
    ann_preds = ann_model.predict(testX)

    accuracy = accuracy_score(ann_preds, testY)
    precision = precision_score(y_pred=ann_preds, y_true=testY, pos_label=1)
    recall = recall_score(y_pred=ann_preds, y_true=testY, pos_label=1)
    f1 = f1_score(y_pred=ann_preds, y_true=testY, pos_label=1)

    print(f"Accuracy: {accuracy:.2%}\nPrecision: {precision:.2%}\nRecall: {recall:.2%}\nF1: {f1:.2%}")
    return (ann_model,)


@app.cell
def _(ann_model, plt, sns):
    sns.lineplot(x=range(len(ann_model.loss_curve_)), y=ann_model.loss_curve_)
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""# Naive Bayes""")
    return


@app.cell
def _(mo):
    mo.md(r"""# C5.0 Decision Tree""")
    return


@app.cell
def _(mo):
    mo.md(r"""# KNN""")
    return


@app.cell
def _(mo):
    mo.md(r"""# Random Forest""")
    return


if __name__ == "__main__":
    app.run()
