import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_target_distribution(df,col):
    plt.figure(figsize=(7, 5))

    ax = sns.countplot(data=df, x=col)

    for container in ax.containers:
        ax.bar_label(container)

    plt.title('Distribution of '+col+' column')
    plt.xlabel('Category')
    plt.ylabel('Number')
    plt.show()


def plot_confusion_matrix(cm,name):
        plt.figure(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix for '+ name)
        plt.show()

def heatmap_features(df):
     plt.figure(figsize=(7,5))
     df=pd.get_dummies(df)
     corr_map=df.corr()
     sns.heatmap(corr_map)
     plt.show()

def plot_model_metrics(result):
    models = result.columns.tolist()

    fig, ax = plt.subplots(layout='constrained')

    res = ax.grouped_bar(result.T, group_spacing=4)
    for container in res.bar_containers:
        ax.bar_label(container, padding=5,fmt='%.2f')

    ax.set_title('models and their metrics')
    ax.legend(loc='upper left', ncols=2)
    ax.set_ylim(0, 1)

    plt.show()    
