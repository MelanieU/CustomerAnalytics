import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pylab as pl
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn import cluster, tree, decomposition
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from matplotlib import cm


def kmeans_display(num_list, orders_clean, n_clusters=8):
    """This functions takes a list of selected features as argument and the dataset and number of clusters as keywords.
    The output are graphs that can be used to evaluate kmeans clustering"""
    
    df_original = orders_clean[num_list]
    customers = orders_clean[['shipping_company']]

    for column in num_list:
    
        if df_original[column].dtype == 'object':
            le = LabelEncoder()
            df_original[column] = le.fit_transform(df_original[column])
    
        df_original[column] = preprocessing.scale(df_original[column])
        
    df_original = df_original.loc[df_original['lineitem_quantity'] <= 12]
           
    df_original = pd.concat([customers, df_original], axis=1).dropna()
    df = df_original.groupby('shipping_company').mean()

    print (df.head())
           
    distortion = []
    for i in range(1,21):
        km = KMeans(n_clusters=i,
                      n_init=10,
                      random_state=0)
        km.fit(df)
        distortion.append(km.inertia_)

    km = cluster.KMeans(n_clusters=n_clusters, max_iter=300, random_state=0)
    df['cluster'] = km.fit_predict(df[num_list])
    
    pca = decomposition.PCA(n_components=2, whiten=True)
    pca.fit(df_original[num_list])
    df['x'] = pca.fit_transform(df[num_list])[:, 0]
    df['y'] = pca.fit_transform(df[num_list])[:, 1]
   
    km = KMeans(n_clusters=n_clusters, random_state=0)
    test_km = km.fit_predict(df)
    
    cluster_labels = np.unique(test_km)
    n_cluster = cluster_labels.shape[0]
    sil_vals = silhouette_samples(df, test_km, metric = 'euclidean')
    y_ax_lower, y_ax_upper = 0, 0
    yticks = []
    
    plt.figure(facecolor='w', figsize=(20,5))
    plt.subplot(121)
    plt.plot(range(1,21), distortion, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    
    plt.subplot(122)
    plt.scatter(df['x'], df['y'], c=df['cluster'])
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    
    plt.figure(facecolor='w', figsize=(10,5))
    for i, c in enumerate(cluster_labels):
        c_sil_vals = sil_vals[test_km == c]
        c_sil_vals.sort()
    
        y_ax_upper += len(c_sil_vals)
        color = cm.jet(float(i)/ n_cluster)
               
        plt.barh(range(y_ax_lower, y_ax_upper),
                c_sil_vals,
                height=1.0,
                edgecolor = 'none',
                color=color)

        yticks.append((y_ax_lower + y_ax_upper) / 2.)
    
        y_ax_lower += len(c_sil_vals)
    
    sil_avg = np.mean(sil_vals)
        
    plt.axvline(sil_avg,
               color='red',
               linestyle='--')
    plt.yticks(yticks, cluster_labels +1)
    plt.ylabel('Cluster')
    plt.xlabel('Silhouette coefficient')


def kmeans_model(num_list, orders_clean, n_clusters=9):
    """ This functions takes a list of selected features as argument and the dataset as well as the optimized number of clusters
    as keywords"""
    
    df_original = orders_clean[num_list]
    customers = orders_clean[['shipping_company']]

    df_feat = pd.concat([customers, df_original], axis=1).dropna(axis=0)

    for column in num_list:
    
        if df_original[column].dtype == 'object':
            le = LabelEncoder()
            df_original[column] = le.fit_transform(df_original[column])
    
        df_original[column] = preprocessing.scale(df_original[column])
        
        df_original = df_original.loc[df_original['lineitem_quantity'] <= 12]
           
    df_original = pd.concat([customers, df_original], axis=1).dropna(axis=0)
    df = df_original.groupby('shipping_company').mean()

    km = cluster.KMeans(n_clusters=n_clusters, max_iter=300, random_state=0)
    df['cluster'] = km.fit_predict(df[num_list])
    clustered = df_feat.reset_index().merge(df.reset_index()[['shipping_company','cluster']], on='shipping_company')
    
    return clustered

def stats_comparison(i, orders_trunc):
    """This function takes a column name as argument and performs ttest analysis to look for difference between the average
    sales grouped by categories of that feature and overall average sales"""
    
    cat = orders_trunc.groupby(i)['lineitem_total']\
        .agg({
            'sub_average_$': 'mean',
            'sub_sum_$': 'sum'
       }).reset_index()
    cat['overall_average_$'] = orders_trunc['lineitem_total'].mean()
    cat['overall_sum_$'] = orders_trunc['lineitem_total'].sum()
    cat['rest_sum_$'] = cat['overall_sum_$'] - cat['sub_sum_$']
    cat['rest_average_$'] = (cat['overall_average_$']*cat['overall_average_$'] - cat['sub_sum_$']*cat['sub_average_$'])/cat['rest_sum_$']

    cat['std'] = np.std(orders_trunc['lineitem_total'], axis=0)
    cat['z_score'] = (cat['sub_average_$']-cat['rest_average_$'])/np.std(orders_trunc['lineitem_total'], axis=0)
    cat['prob'] = np.around(stats.norm.cdf(cat.z_score), decimals = 10)
    cat['significant'] = [(lambda x: 1 if x > 0.9 else -1 if x < 0.1 else 0)(i) for i in cat['prob']]

    #pd.options.display.float_format = '{:,.0f}'.format
    return cat