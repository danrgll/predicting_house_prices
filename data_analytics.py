from matplotlib import pyplot as plt
import data_preprocessing as dp
import seaborn as sns
import pandas as pd
sns.set_theme(style="ticks", color_codes=True)


# ToDo:
def plot_feature_data(feature1_data, feature2_data=None, target_data=None):
    if feature2_data is not None:
        plt.plot(feature1_data, feature2_data)
    else:
        plt.plot(feature1_data)
    plt.show()


# ToDo:
def scatter_feature_data(feature1_data, feature2_data=None, target_data=None):
    if feature2_data is not None:
        plt.scatter(feature1_data, feature2_data)
    elif target_data != None:
        plt.scatter()
    plt.show()


def scatter_feature_to_target(df, feature, target, kind="strip", jitter=True):
    """
    scatter categorial feature to numerous target
    :param df: pandas dataframe
    :param feature: string
    :param target: string
    :param kind: string(“strip”, “swarm”, “box”, “violin”, “boxen”, “point”, “bar”, or “count”)
    :return:
    """
    df = dp.load_data(df)
    sns.catplot(x=feature, y=target, kind=kind, jitter=jitter, data=df)
    plt.savefig('scatter_feature_target')


if __name__ == '__main__':
    scatter_feature_to_target("train.csv", "LotConfig", "SalePrice", jitter=False)
