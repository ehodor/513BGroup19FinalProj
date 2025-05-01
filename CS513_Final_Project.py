

import marimo

__generated_with = "0.13.2"
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
        * Mihir - Logistic Regression and Random Forest
        * Sam - Hierarchical Clustering and K-Means Clustering
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
def _(df):
    df.info()
    return


@app.cell
def _(df):
    df.dropna(how="any", axis=0, inplace=True)
    return


@app.cell
def _(df):
    df.isnull().sum()
    return


@app.cell
def _(df):
    df.describe(include="all")
    return


@app.cell
def _(df, plt):
    edafig, eda_axes = plt.subplots(4, 5, figsize=(35, 25))
    for col, eda_axis in zip(df.columns, eda_axes.ravel()):
        eda_axis.hist(df[col])
        eda_axis.set_title(f"{col}")

    plt.show()
    return


@app.cell
def _(datetime, df):
    def to_integer(dt_time):
        if not isinstance(dt_time, str):
            dt_time = str(dt_time)
        dt_time = datetime.date.fromisoformat(dt_time)
        return 10000*dt_time.year + 100*dt_time.month + dt_time.day
    
    df["FlightDate"] = df["FlightDate"].map(to_integer)
    df
    return to_integer


@app.cell
def _(df, np, plt, sns):
    matrix = df.corr(numeric_only=True)
    plt.tight_layout()
    plt.figure(figsize=(10, 10))
    sns.heatmap(matrix, cmap=sns.color_palette("flare", as_cmap=True), annot=True, fmt=".2f", mask=np.triu(np.ones_like(matrix)))
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
def _(StandardScaler, ann_df, pd, target, train_test_split):
    scaler = StandardScaler()
    ann_df_scaled = pd.DataFrame(scaler.fit_transform(ann_df), columns=ann_df.columns)

    trainX, testX, trainY, testY = train_test_split(ann_df_scaled, target, random_state=42, test_size=0.3)
    print(trainX.shape)
    print(trainY.shape)
    return scaler, testX, testY, trainX, trainY


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
    ann_model = MLPClassifier(hidden_layer_sizes=(trainX.shape[1]*2, trainX.shape[1]//2), max_iter=1000, learning_rate='adaptive', early_stopping=True, alpha=0.001)
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
def _(ann_model, plt):
    # use global min / max to ensure all weights are shown on the same scale
    fig, axes = plt.subplots(1, 3)

    for coef, ax1 in zip(ann_model.coefs_, axes.ravel()):
        vmin, vmax = min([min(i) for i in coef]), max([max(i) for i in coef])
        ax1.matshow(coef, cmap=plt.cm.hsv, vmin=0.5 * vmin, vmax=0.5 * vmax)
        ax1.set_title(f"{coef.shape}")
        ax1.set_xticks(())
        ax1.set_yticks(())

    plt.show()
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
):
    cnb = CategoricalNB(alpha=0.001)
    gnb = GaussianNB(var_smoothing=0.0001)


    gnb_scaler = MinMaxScaler()

    categorical_cols = ["Reporting_Airline", "Origin", "OriginState", "Dest", "DestState"] # separate categorical and continuous data
    cnb_categorical = df[categorical_cols]
    preprocessor = ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(sparse_output=False), categorical_cols)
    ])

    gnb_df = df.drop(columns=categorical_cols)
    gnb_df = pd.DataFrame(gnb_scaler.fit_transform(gnb_df), columns=gnb_df.columns)

    cnb_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', cnb)
    ])
    return cnb_categorical, cnb_pipeline, gnb, gnb_df


@app.cell
def _(cnb_categorical, cnb_pipeline, gnb, gnb_df, target, train_test_split):
    cnb_trainX, cnb_testX, cnb_trainY, cnb_testY = train_test_split(cnb_categorical, target, random_state=42, test_size=0.4)
    gnb_trainX, gnb_testX, gnb_trainY, gnb_testY = train_test_split(gnb_df, target, random_state=42, test_size=0.4)

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
    gnb2 = GaussianNB(var_smoothing=0.0001) # create new Gaussian Model

    mixed_trainX, mixed_testX, mixed_trainY, mixed_testY = train_test_split(new_features, target, random_state=42, test_size=0.4)
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
    trainX_CART, testX_CART, trainY_CART, testY_CART = train_test_split(ann_df, target, random_state=42,  test_size=0.2)
    print(trainX_CART.shape)
    print(trainY_CART.shape)
    trainX_CART, trainY_CART
    return testX_CART, testY_CART, trainX_CART, trainY_CART


@app.cell
def _(DecisionTreeClassifier, testX_CART, trainX_CART, trainY_CART):
    model_CART = DecisionTreeClassifier(max_depth=7, max_leaf_nodes=50, max_features='sqrt')
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
    #mo.md(f"""Confusion matrix:""")
    print(classification_report(testY_CART,target_pred_CART))
    #mo.md(f"""Classification report: """)
    confusion_matrix(testY_CART,target_pred_CART)
    return

@app.cell
def _(classification_report,
      target_pred_CART,
    testY_CART,
      ):
     classification_report(testY_CART,target_pred_CART)
     return
    


@app.cell
def _(ann_df, model_CART, plot_tree, plt):
    plt.figure(figsize=(20,10), dpi=50)
    tree_res = plot_tree(model_CART,fontsize=15,filled=True,feature_names=ann_df.columns);
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
    from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
    from sklearn.pipeline import FeatureUnion, make_pipeline
    return (
        FeatureUnion,
        FunctionTransformer,
        KNeighborsClassifier,
        MinMaxScaler,
        make_pipeline,
    )


@app.cell
def _(df, target, train_test_split):

    attr = df.head(80000)
    target_knn = target.head(80000)
    #print(attr)
    trainX_knn, testX_knn, trainY_knn, testY_knn = train_test_split(attr, target_knn, random_state=42, test_size=0.2)
    print(trainX_knn.shape)
    print(trainY_knn.shape)
    trainX_knn, trainY_knn
    return testX_knn, testY_knn, trainX_knn, trainY_knn


@app.cell
def _(
    FeatureUnion,
    FunctionTransformer,
    KNeighborsClassifier,
    MinMaxScaler,
    OneHotEncoder,
    Pipeline,
    accuracy_score,
    make_pipeline,
    testX_knn,
    testY_knn,
    trainX_knn,
    trainY_knn,
):
    minmax_scaler = MinMaxScaler()

    categorical_cols_knn = ["Reporting_Airline", "Origin", "OriginState", "Dest", "DestState"] # separate categorical and continuous data
    def numeric(d):
        numeric_cols = d.drop(columns=categorical_cols_knn)
        return numeric_cols

    def categ(d):
        return d[categorical_cols_knn]

    k_values = [3, 5, 10]
    for k in k_values:
        print(k)
        knn_pipeline = Pipeline([
    ("features", FeatureUnion([
        ('numeric', make_pipeline(FunctionTransformer(numeric),minmax_scaler)),
        ('categorical', make_pipeline(FunctionTransformer(categ),OneHotEncoder(sparse_output=False)))
    ])),
    ('model', KNeighborsClassifier(n_neighbors= k))
    ])
        knn_pipeline.fit(trainX_knn, trainY_knn)
        target_pred_KNN = knn_pipeline.predict(testX_knn)
        accuracy_KNN = accuracy_score(testY_knn,target_pred_KNN ) 
        print(f"Accuracy of model with k = {k}: {accuracy_KNN}")
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
    confusion_matrix(testY_knn, target_pred_KNN), accuracy_score(testY_knn, target_pred_KNN), classification_report(testY_knn, target_pred_KNN)
    return (cm,)


@app.cell
def _(target_pred_KNN, testX_knn, testY_knn):
    test_actual=testX_knn
    test_actual['target_pred']=target_pred_KNN
    test_actual['test_actual']=testY_knn
    test_actual.head(10)
    test_actual
    return (test_actual,)


@app.cell
def _(pd, test_actual):
    freq_table =pd.crosstab( test_actual['test_actual'], test_actual['target_pred'])
    print("Confusion Matrix")
    print(freq_table)
    freq_table
    return


@app.cell
def _(cm, plt, sns):
    ax= plt.subplot()
    heatmap = sns.heatmap(cm, annot=True, fmt='g', ax=ax); 

    # labels, title and ticks
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(['Not delayed', 'Delayed'])
    ax.yaxis.set_ticklabels(['Not delayed', 'Delayed'])
    plt.show()
    heatmap
    return


@app.cell
def _(mo):
    mo.md(r"""# Random Forest""")
    return

@app.cell
def _():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import set_config
    return (RandomForestClassifier, set_config)

@app.cell
def _(df):
    rf_target = df['is_delay']
    rf_df = df.iloc[:, 1::]
    return rf_df, rf_target

@app.cell
def _(rf_df, to_integer, RandomForestClassifier, ColumnTransformer, FunctionTransformer, OneHotEncoder, Pipeline, set_config):
    # Preprocess the data

    def preprocess_dates(X):
        X = X.copy()
        X.iloc[:, 0] = X.iloc[:, 0].map(to_integer)
        return X

    cat_features = ['Reporting_Airline', 'Origin', 'OriginState', 'Dest', 'DestState']
    num_features = [col for col in rf_df.columns if col not in cat_features + ['FlightDate']]

    rf_preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', num_features),
            ('date', FunctionTransformer(preprocess_dates), ['FlightDate']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ])

    rf_pipeline = Pipeline([
        ('preprocessor', rf_preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=77,
            class_weight='balanced',
        ))
    ])
    return (cat_features, num_features, rf_pipeline)

@app.cell
def _(rf_df, target, train_test_split, rf_pipeline):

    rf_Xtrain, rf_Xtest, rf_ytrain, rf_ytest = train_test_split(rf_df, target, test_size=0.2, random_state=77)
   
    print(rf_Xtrain.shape)
    print(rf_ytrain.shape)
    
    rf_pipeline.fit(rf_Xtrain, rf_ytrain)
    return (rf_Xtrain, rf_Xtest, rf_ytrain, rf_ytest)

@app.cell
def _(rf_pipeline, rf_Xtest, rf_ytest, accuracy_score, confusion_matrix, classification_report, plt, sns):
    rf_y_pred = rf_pipeline.predict(rf_Xtest)

    rf_accuracy = accuracy_score(rf_ytest, rf_y_pred)
    rf_cm = confusion_matrix(rf_ytest, rf_y_pred)
    rf_cr = classification_report(rf_ytest, rf_y_pred)

    print(f"Random Forest Accuracy: {rf_accuracy:.2%}")
    print("Confusion Matrix:")
    print(rf_cm)
    print("Classification Report:")
    print(rf_cr)

    rf_labels = ['No Delay', 'Delay']

    # Plot
    plt.figure(figsize=(6, 5))
    rf_hm = sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', xticklabels=rf_labels, yticklabels=rf_labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    rf_hm

    return (rf_hm)#(rf_y_pred,)

@app.cell
def _(rf_pipeline, rf_df, cat_features, num_features, OneHotEncoder, pd, plt, sns):
    # Extract the trained RandomForest model
    rf_model = rf_pipeline.named_steps["classifier"]
    
    # Get feature names from the preprocessor
    pp = rf_pipeline.named_steps["preprocessor"]
    
    # Get names for numeric and date features
    numeric_names = num_features + ['FlightDate']
    
    # Get one-hot encoded column names from the fitted encoder
    ohe = rf_pipeline.named_steps.preprocessor.named_transformers_["cat"]
    ohe_feature_names = ohe.get_feature_names_out(cat_features)
    
    # Combine all feature names in the order used by the model
    feature_names = numeric_names + list(ohe_feature_names)
    
    # Get importances
    importances = rf_model.feature_importances_
    
    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(20)

    # Plot
    plt.figure(figsize=(10, 6))
    rf_importance_plot = sns.barplot(data=fi_df, x="Importance", y="Feature")
    plt.tight_layout()
    plt.show()

    rf_importance_plot
    return (fi_df)
 
@app.cell
def _(mo):
    mo.md(r"""# Logistic Regression""")
    return

@app.cell
def _():
    from sklearn.linear_model import LogisticRegression
    return (
        LogisticRegression,
    )

@app.cell
def _(ann_df, target, train_test_split):
    log_trainX, log_testX, log_trainY, log_testY = train_test_split(ann_df, target, test_size=0.3, random_state=42)
    print(log_trainX.shape, log_testX.shape)
    return log_trainX, log_testX, log_trainY, log_testY

@app.cell
def _(LogisticRegression, log_trainX, log_trainY):
    log_model = LogisticRegression(
                    max_iter=1000, 
                    solver='liblinear', 
                    class_weight='balanced', 
                    random_state=42)
    log_model.fit(log_trainX, log_trainY)
    return log_model

@app.cell
def _(log_model, log_testX, log_testY, accuracy_score, precision_score, recall_score, f1_score):
    log_preds = log_model.predict(log_testX)
    log_accuracy = accuracy_score(log_testY, log_preds)
    log_precision = precision_score(log_testY, log_preds)
    log_recall = recall_score(log_testY, log_preds)
    log_f1 = f1_score(log_testY, log_preds)

    print(f"Accuracy: {log_accuracy:.2%}\nPrecision: {log_precision:.2%}\nRecall: {log_recall:.2%}\nF1 Score: {log_f1:.2%}")
    return log_preds

@app.cell
def _(confusion_matrix, log_testY, log_preds, plt, sns):
    cm_log = confusion_matrix(log_testY, log_preds)
    print("Confusion Matrix:")
    print(cm_log)

    log_labels = ['No Delay', 'Delay']

    # Plot
    plt.figure(figsize=(6, 5))
    log_hm = sns.heatmap(cm_log, annot=True, fmt='d', cmap='Blues', xticklabels=log_labels, yticklabels=log_labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    # plt.show()

    return (cm_log, log_hm)



@app.cell
def _(mo):
    mo.md(r"""# Hierarchal Clustering""")
    return


@app.cell
def _():
    from sklearn.decomposition import PCA  
    from sklearn.preprocessing import normalize 
    from sklearn.metrics import silhouette_score 
    import scipy.cluster.hierarchy as shc 
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.cluster import KMeans
    return (
        AgglomerativeClustering,
        KMeans,
        PCA,
        normalize,
        shc,
        silhouette_score,
    )


@app.cell
def _(PCA, ann_df, normalize, pd, plt, scaler, shc, target):
    cluster_attr = ann_df.head(10000)
    target_clust = target.head(10000)

    df_scaled_hclust = scaler.fit_transform(cluster_attr) 

    # Normalizing the data so that the data approximately  
    # follows a Gaussian distribution 
    df_normalized_hclust = normalize(df_scaled_hclust) 

    # Converting the numpy array into a pandas DataFrame 
    df_normalized_hclust = pd.DataFrame(df_normalized_hclust) 

    pca = PCA(n_components = 2) 
    df_principal_hclust = pca.fit_transform(df_normalized_hclust) 
    df_principal_hclust = pd.DataFrame(df_principal_hclust) 
    df_principal_hclust.columns = ['P1', 'P2'] 


    plt.figure(figsize =(8, 8)) 
    plt.title('Visualising the data') 
    Dendrogram = shc.dendrogram((shc.linkage(df_principal_hclust, method ='ward'))) 
    plt.show()

    return cluster_attr, df_principal_hclust, target_clust


@app.cell
def _(AgglomerativeClustering, df_principal_hclust, plt, silhouette_score):
    silhouette_scores = [] 

    for i in range(2, 20):
        ac = AgglomerativeClustering(n_clusters = i)
    
        # Visualizing the clustering 
        plt.figure(figsize =(6, 6)) 
        plt.scatter(df_principal_hclust['P1'], df_principal_hclust['P2'],  
                   c = ac.fit_predict(df_principal_hclust), cmap ='rainbow') 
        plt.show() 

        silhouette_scores.append( 
            silhouette_score(df_principal_hclust, ac.fit_predict(df_principal_hclust))) 
    return (silhouette_scores,)


@app.cell
def _(plt, silhouette_scores):
    # Plotting a bar graph to compare the results 
    plt.bar([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], silhouette_scores) 

    plt.xlabel('Number of clusters', fontsize = 20) 
    plt.ylabel('S(i)', fontsize = 20) 

    plt.show() 

    print(silhouette_scores)
    return


@app.cell
def _(AgglomerativeClustering, df_principal_hclust, plt):
    ac5 = AgglomerativeClustering(n_clusters = 5)

    clusters = ac5.fit_predict(df_principal_hclust)
    
    # Visualizing the clustering 
    plt.figure(figsize =(6, 6)) 
    plt.scatter(df_principal_hclust['P1'], df_principal_hclust['P2'],  
               c=clusters, cmap ='rainbow') 
    plt.show() 
    return (clusters,)


@app.cell
def _(clusters, pd, target_clust):
    df_cluster=pd.DataFrame({'Actual':target_clust,'Cluster':clusters})
    # Create a cross-tabulation
    cross_tab = pd.crosstab(df_cluster['Actual'], df_cluster['Cluster'])

    print(cross_tab)
    return


@app.cell
def _(mo):
    mo.md(r"""# K-Means Clustering""")
    return


@app.cell
def _(KMeans, cluster_attr, pd, target_clust):
    kmeans = KMeans(n_clusters=5, random_state=12)
    kmeans.fit(cluster_attr)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    print(labels)
    print(centers)

    df_kmeans=pd.DataFrame({'Actual':target_clust,'Cluster':labels})

    kmeans_cross_tab = pd.crosstab(df_kmeans['Actual'], df_kmeans['Cluster'])

    print(kmeans_cross_tab)
    return


if __name__ == "__main__":
    app.run()
