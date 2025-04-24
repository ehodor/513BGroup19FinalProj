

import marimo

__generated_with = "0.13.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from sklearn.model_selection import train_test_split
    return mo, np, pd, plt, sns, train_test_split


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
        * Emma - CART and KNN
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
    #print(ann_df)
    return (to_integer,)


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
    plotted = sns.lineplot(x=range(len(ann_model.loss_curve_)), y=ann_model.loss_curve_)
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    plotted
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Naive Bayes
        Thank you, [StackOverflow](https://stackoverflow.com/questions/14254203/mixing-categorial-and-continuous-data-in-naive-bayes-classifier-using-scikit-lea/14255284#14255284)...
        """
    )
    return


@app.cell
def _():
    from sklearn.naive_bayes import CategoricalNB, GaussianNB
    from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    return (
        CategoricalNB,
        ColumnTransformer,
        GaussianNB,
        OneHotEncoder,
        Pipeline,
    )


@app.cell
def _(
    CategoricalNB,
    ColumnTransformer,
    GaussianNB,
    MinMaxScaler,
    OneHotEncoder,
    Pipeline,
    df,
    pd,
    to_integer,
):
    cnb = CategoricalNB()
    gnb = GaussianNB()


    gnb_scaler = MinMaxScaler()

    categorical_cols = ["Reporting_Airline", "Origin", "OriginState", "Dest", "DestState"] # separate categorical and continuous data
    cnb_categorical = df[categorical_cols]
    preprocessor = ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(sparse_output=False), categorical_cols)
    ])

    gnb_df = df.drop(columns=categorical_cols)
    gnb_df["FlightDate"] = gnb_df["FlightDate"].map(to_integer)
    gnb_df = pd.DataFrame(gnb_scaler.fit_transform(gnb_df), columns=gnb_df.columns)

    cnb_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', cnb)
    ])
    return cnb_categorical, cnb_pipeline, gnb, gnb_df


@app.cell
def _(cnb_categorical, cnb_pipeline, gnb, gnb_df, target, train_test_split):
    cnb_trainX, cnb_testX, cnb_trainY, cnb_testY = train_test_split(cnb_categorical, target, random_state=42, test_size=0.2)
    gnb_trainX, gnb_testX, gnb_trainY, gnb_testY = train_test_split(gnb_df, target, random_state=42, test_size=0.2)

    cnb_pipeline.fit(X=cnb_trainX, y=cnb_trainY) # cnb for categorical data
    cnb_preds = cnb_pipeline.predict(cnb_testX)

    gnb.fit(X=gnb_trainX, y=gnb_trainY) # gnb for continuous
    gnb_preds = gnb.predict(gnb_testX)
    return


@app.cell
def _(
    GaussianNB,
    cnb_categorical,
    cnb_pipeline,
    gnb,
    gnb_df,
    np,
    target,
    train_test_split,
):
    cnb_probabilities = cnb_pipeline.predict_proba(cnb_categorical) # get probabilities for both
    gnb_probabilities = gnb.predict_proba(gnb_df)

    new_features = np.hstack((cnb_probabilities, gnb_probabilities)) # concatenate them
    gnb2 = GaussianNB() # create new Gaussian Model

    mixed_trainX, mixed_testX, mixed_trainY, mixed_testY = train_test_split(new_features, target, random_state=42, test_size=0.2)
    return gnb2, mixed_testX, mixed_testY, mixed_trainX, mixed_trainY


@app.cell
def _(gnb2, mixed_testX, mixed_trainX, mixed_trainY):
    gnb2.fit(X=mixed_trainX, y=mixed_trainY)
    gnb2_preds = gnb2.predict(mixed_testX)
    return (gnb2_preds,)


@app.cell
def _(
    accuracy_score,
    f1_score,
    gnb2_preds,
    mixed_testY,
    precision_score,
    recall_score,
):
    gnb_accuracy = accuracy_score(gnb2_preds, mixed_testY)
    gnb_precision = precision_score(y_pred=gnb2_preds, y_true=mixed_testY, pos_label=1)
    gnb_recall = recall_score(y_pred=gnb2_preds, y_true=mixed_testY, pos_label=1)
    gnb_f1 = f1_score(y_pred=gnb2_preds, y_true=mixed_testY, pos_label=1)
    print(f"Accuracy: {gnb_accuracy:.2%}\nPrecision: {gnb_precision:.2%}\nRecall: {gnb_recall:.2%}\nF1: {gnb_f1:.2%}")
    return


@app.cell
def _(confusion_matrix, gnb2_preds, mixed_testY):
    confusion_matrix(y_true=mixed_testY, y_pred=gnb2_preds)
    return


@app.cell
def _(mo):
    mo.md(r"""# CART""")
    return


@app.cell
def _():
    from sklearn.tree import plot_tree
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import confusion_matrix, classification_report
    return (
        DecisionTreeClassifier,
        classification_report,
        confusion_matrix,
        plot_tree,
    )


@app.cell
def _(ann_df, target, train_test_split):

    trainX_CART, testX_CART, trainY_CART, testY_CART = train_test_split(ann_df, target, random_state=42, test_size=0.2)
    print(trainX_CART.shape)
    print(trainY_CART.shape)
    return testX_CART, testY_CART, trainX_CART, trainY_CART


@app.cell
def _(DecisionTreeClassifier, testX_CART, trainX_CART, trainY_CART):
    model_CART = DecisionTreeClassifier()
    model_CART.fit(trainX_CART,trainY_CART)
    target_pred_CART = model_CART.predict(testX_CART)
    return model_CART, target_pred_CART


@app.cell
def _(
    classification_report,
    confusion_matrix,
    mo,
    target_pred_CART,
    testY_CART,
):
    print(confusion_matrix(testY_CART,target_pred_CART))
    mo.md(f"{confusion_matrix(testY_CART,target_pred_CART)}")
    print(classification_report(testY_CART,target_pred_CART))
    mo.md(f"{classification_report(testY_CART,target_pred_CART)}")
    return


@app.cell
def _(ann_df, model_CART, plot_tree, plt):
    plt.figure(figsize=(20,10), dpi=100)
    tree_res = plot_tree(model_CART,fontsize=20,filled=True,feature_names=ann_df.columns);
    plt.show()
    tree_res
    return


@app.cell
def _(mo):
    mo.md(r"""# KNN""")
    return


@app.cell
def _():
    from sklearn.neighbors import KNeighborsClassifier 
    from sklearn.preprocessing import MinMaxScaler
    return KNeighborsClassifier, MinMaxScaler


@app.cell
def _(MinMaxScaler, ann_df, pd, target, train_test_split):

    scaler_minmax = MinMaxScaler()
    # Fit and transform the data
    attr = ann_df.head(80000)
    target_knn = target.head(80000)
    print(attr)
    attr = pd.DataFrame(scaler_minmax.fit_transform(attr), columns=attr.columns)
    attr.head()
    trainX_knn, testX_knn, trainY_knn, testY_knn = train_test_split(attr, target_knn, random_state=42, test_size=0.2)
    print(trainX_knn.shape)
    print(trainY_knn.shape)
    return testX_knn, testY_knn, trainX_knn, trainY_knn


@app.cell
def _(
    KNeighborsClassifier,
    accuracy_score,
    testX_knn,
    testY_knn,
    trainX_knn,
    trainY_knn,
):
    k_values = [3, 5, 10]
    for k in k_values:
        print(k)
        knn = KNeighborsClassifier(n_neighbors = k)
        knn.fit(trainX_knn, trainY_knn)
        target_pred_KNN = knn.predict(testX_knn)
        accuracy_KNN = accuracy_score(testY_knn,target_pred_KNN ) 
        print(f'Accuracy of model with k = {k}: {accuracy_KNN}')
        print('')
    return (target_pred_KNN,)


@app.cell
def _(
    accuracy_score,
    classification_report,
    confusion_matrix,
    target_pred_KNN,
    testY_knn,
):
    cm=confusion_matrix(testY_knn, target_pred_KNN)
    print('Confusion Matrix')
    print(confusion_matrix(testY_knn, target_pred_KNN))
    print()
    print('Accuracy score')
    print(accuracy_score(testY_knn, target_pred_KNN))
    print()
    print('Classification Report')
    print(classification_report(testY_knn, target_pred_KNN))
    return (cm,)


@app.cell
def _(target_pred_KNN, testX_knn, testY_knn):
    test_actual=testX_knn
    test_actual['target_pred']=target_pred_KNN
    test_actual['test_actual']=testY_knn
    test_actual.head(10)
    return (test_actual,)


@app.cell
def _(pd, test_actual):
    freq_table =pd.crosstab( test_actual['test_actual'], test_actual['target_pred'])
    print("Confusion Matrix")
    print(freq_table)
    return


@app.cell
def _(cm, plt, sns):
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax); 

    # labels, title and ticks
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(['Not delayed', 'Delayed'])
    ax.yaxis.set_ticklabels(['Not delayed', 'Delayed'])
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""# Random Forest""")
    return


if __name__ == "__main__":
    app.run()
