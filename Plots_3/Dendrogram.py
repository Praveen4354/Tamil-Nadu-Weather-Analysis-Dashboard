import streamlit as st
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
import plotly.express as px

def dendrogram(df, features):

    import plotly.figure_factory as ff
    from scipy.cluster.hierarchy import linkage

    transposed_sampled_master = df[features].transpose()

    Z = linkage(transposed_sampled_master, method='centroid')

    feature_labels = transposed_sampled_master.index.tolist()

    fig = ff.create_dendrogram(transposed_sampled_master, orientation='bottom', linkagefun=lambda x: Z, labels=feature_labels)
    fig.update_layout(height=700, title='Feature-based Dendrogram')
    st.plotly_chart(fig, use_container_width=True)


def Dendrogram(df):
    numcols = df.select_dtypes(include=['number']).columns
    st.write("### Parallel Coordinates")
    with st.expander("Show code"):
        st.code("Code")
    
    features = st.multiselect("Select Features:", list(numcols), list(numcols[:15])) 
    
    if features:
        dendrogram(df[numcols], features)