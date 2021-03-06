
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import pathlib2 as pl2

def read_data():
    proj_path = os.path.join((os.getcwd()),os.path.pardir)
    raw_path = os.path.join(proj_path, "data","raw")
    train_file_path=os.path.join(raw_path, "train.csv")
    test_file_path=os.path.join(raw_path, "test.csv")
    train_df = pd.read_csv(train_file_path, index_col="PassengerId")
    test_df = pd.read_csv(test_file_path, index_col="PassengerId")
    test_df['Survived']=-999
    titanic_df = pd.concat((train_df,test_df))
    return titanic_df


def process_data(df):
    return (df
           .assign(Title=lambda tdf: tdf.Name.map(getTitle))
            .pipe(fill_missing_values)
            .assign(Fare_Bin=lambda tdf: pd.qcut(tdf.Fare,4, labels=["very_low","low","medium","high"]))
            .assign(Agestate=lambda tdf: np.where(tdf["Age"]>=18,"Adults","Child"))
            .assign(FamilySize=lambda tdf: tdf.Parch + tdf.SibSp + 1)
            .assign(IsMother=lambda tdf:np.where(((tdf.Sex=='female') & (tdf.Age > 18) & (tdf.Parch > 0) & (tdf.Title != "Miss")),1,0))
            .assign(Cabin=lambda tdf: np.where(tdf.Cabin=="T",np.NaN,tdf.Cabin))
            .assign(Deck=lambda tdf: tdf.Cabin.map(get_deck))
            .assign(IsMale=lambda tdf: np.where(tdf.Sex=="male",1,0))
            .pipe(pd.get_dummies, columns=["Deck","Pclass","Title","Fare_Bin","Embarked","Agestate"])
            .drop(["Cabin","Name","Ticket","Parch","SibSp","Sex"],axis=1)
            .pipe(reorder_columns)           
           )

getTitle = lambda name: title_group[name.split(",")[1].split('.')[0].strip().lower()]
title_group = {'mr':'Mr',
'mrs':'Mrs',
'miss':'Miss',
'master':'Master',
'don':'Sir',
'rev':'Sir',
'dr':'Officer',
'mme':'Mrs',
'ms':'Mrs',
'major':'Officer',
'lady':'Lady',
'sir':'Sir',
'mlle':"Miss",
'col':'Officer',
'capt':'Officer',
'the countess':'Lady',
'jonkheer':"Sir",
'dona':"Lady"
}


get_deck = lambda cabin: np.where(pd.notnull(cabin),str(cabin)[0].upper(),'Z')


def fill_missing_values(df):
    df.Embarked.fillna('C',inplace=True)
    mdfare=df[(df.Pclass==3) & (df.Embarked=="S")]['Fare'].median()
    df.Fare.fillna(mdfare,inplace=True)
    title_age_median = df.groupby("Title").Age.transform('median')
    df.Age.fillna(title_age_median,inplace=True)
    return df

def reorder_columns(df):
    cols = ['Survived'] + [col for col in df.columns if col != 'Survived']
    df=df[cols]
    return df


def write_data(df):
    pl2.Path(os.path.join(os.path.pardir,"data","processed")).mkdir(parents=True, exist_ok=True)
    pr_train_path=os.path.join(os.path.pardir,"data","processed","train.csv")
    pr_test_path=os.path.join(os.path.pardir,"data","processed","test.csv")

    df.loc[df.Survived!=-999].to_csv(pr_train_path)
    colms = [c for c in df.columns if c!='Survived']
    df.loc[df.Survived==-999,colms].to_csv(pr_test_path)


if __name__ == "__main__":
    df = read_data();
    df = process_data(df);
    write_data(df);