import pandas as pd
import os
for i in ["train_FD001_tabular.csv","train_FD002_tabular.csv","train_FD003_tabular.csv","train_FD004_tabular.csv", "test_FD001_tabular.csv","test_FD002_tabular.csv","test_FD003_tabular.csv","test_FD004_tabular.csv"]:
    df = pd.read_csv("CMAPPS_Cleaned_Data/"+i)
    #Set columns 0 to 27
    df.columns = [str(i) for i in range(0,28)]
    df.drop(["25","26", "27"], axis="columns", inplace=True)
    counter = 0
    overall_counter = 0 
    label_list = []
    for j in range(len(df["0"])) :
        if j == len(df["0"])-1:
            counter += 1
            for z in range(counter//3) : 
                label_list.append(0)
            for z in range(counter//3) :
                label_list.append(1)
            for z in range(counter//3) :
                label_list.append(2)
            remainder = counter%3
            for z in range(remainder):
                label_list.append(2)
        elif df["0"][j] == df["0"][j+1]: 
            counter += 1
        elif df["0"][j] != df["0"][j+1] : 
            overall_counter += 1
            for z in range(counter//3) : 
                label_list.append(0)
            for z in range(counter//3) : 
                label_list.append(1)
            for z in range(counter//3) : 
                label_list.append(2)
            remainder = counter%3
            for z in range(remainder):
                label_list.append(2)
            counter = 1
    df["label"] = label_list
    df.to_csv("CMAPPS_Cleaned_Data/"+i+"_labeled.csv", index=False)
        