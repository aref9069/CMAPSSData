import pandas as pd
import numpy as np
import sklearn as sk
def load_and_shuffle(path1, path2) : 
    validation_df = pd.read_csv(path2)
    train_df = pd.read_csv(path1)
    validation_df = validation_df.sample(frac = 1)
    train_df = validation_df.sample(frac = 1)
    return validation_df, train_df
def isolate_labels(df, label_name = "label") : 
    label_list = np.array(df[label_name])
    df = df.drop("label", axis="columns")
    return df, label_list
def load_TabNet(n_d = 32, n_a = 32, n_steps = 3) : 
    from pytorch_tabnet.tab_model import TabNetClassifier
    classifier = TabNetClassifier(n_d = n_d, n_a = n_a, n_steps = n_steps)
    return classifier
def train(x_train, y_train, x_test, y_test, classifer, metric = ["accuracy", "balanced_accuracy"], epochs = 500, patience = 200, batch_size = 256, virtual_batch_size = 64) :
    classifer.fit(X_train = x_train, y_train = y_train, eval_set = [(x_test, y_test)], eval_metric = metric, max_epochs = epochs, patience = patience, batch_size = batch_size, virtual_batch_size = virtual_batch_size) 
for i in ["FD001_tabular.csv_labeled.csv", "FD002_tabular.csv_labeled.csv", "FD003_tabular.csv_labeled.csv","FD004_tabular.csv_labeled.csv"] : 
    validation_df, train_df = load_and_shuffle("/home/jovyan/workspace/CMAPSSData/CMAPPS_Cleaned_Data/train_" + i,"/home/jovyan/workspace/CMAPSSData/CMAPPS_Cleaned_Data/test_" + i)
    train_df, label_train = isolate_labels(train_df)
    validation_df, label_validation = isolate_labels(validation_df)
    tabular_classifier = load_TabNet()
    history = train(train_df.values, label_train, validation_df.values, label_validation, tabular_classifier)
    model_path = "/home/jovyan/workspace/CMAPSSData/D1_Trained_Model" + i[:5]
    tabular_classifier.save_model(model_path)