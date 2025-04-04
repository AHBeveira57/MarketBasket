import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

# Streamlit Title
st.title("🛒 Market Basket Analysis App")

# Upload File
uploaded_file = st.file_uploader("📂 Upload your transactions CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # **Auto-Detect Dataset Format**
    if df.shape[1] == 1:  
        transactions = df.apply(lambda row: row.dropna().tolist(), axis=1)
    
    elif df.shape[1] == 2:  
        txn_col, item_col = df.columns
        df[item_col] = df[item_col].astype(str)
        basket = df.groupby(txn_col)[item_col].apply(list).reset_index()
        transactions = basket[item_col].tolist()
    
    else:
        df = df.iloc[:, 1:]  # Assume first column is transaction ID & remove it
        # transactions = df.values.tolist()  # Convert entire table to transactions list
        transactions = df.apply(lambda row: row.dropna().tolist(), axis=1)

    # **Mandatory Encoding**
    te = TransactionEncoder()
    transactions_encoded = te.fit(transactions).transform(transactions)
    transactions = pd.DataFrame(transactions_encoded, columns=te.columns_).astype(int)

    # **User Inputs**
    model_choice = st.selectbox("🛠️ Choose Algorithm", ["Apriori", "FP-Growth"])
    min_support = st.slider("📊 Select Minimum Support", 0.01, 0.1, 0.02, 0.01)
    confidence_threshold = st.slider("⚡ Select Confidence Threshold", 0.1, 1.0, 0.35, 0.05)
    num_itemsets = st.slider("🔢 Number of Frequent Itemsets", 5, 50, 10, 5)
    visualization_type = st.radio("📊 Choose Visualization Type", ["Network Graph", "Heatmap", "Bar Chart"])

    # **Apply Selected Algorithm**
    if model_choice == "Apriori":
        frequent_itemsets = apriori(transactions, min_support=min_support, use_colnames=True)
    else:
        frequent_itemsets = fpgrowth(transactions, min_support=min_support, use_colnames=True)

    # Remove "frozenset()" from itemsets
    # frequent_itemsets['itemsets'] = frequent_itemsets['itemsets'].apply(lambda x: ', '.join(list(x)))
    frequent_itemsets_sorted = frequent_itemsets.sort_values(by="support", ascending=False).head(num_itemsets)

    # **Display Frequent Itemsets**
    st.subheader(f"📊 Frequent Itemsets ({model_choice})")
    st.dataframe(frequent_itemsets_sorted)

    # **Generate Association Rules**
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    
    # **Apply Confidence Threshold**
    rules = rules[rules['confidence'] >= confidence_threshold]
    
    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))

    # **User Input: Number of Rules to Display**
    num_rules = st.slider("🔗 Number of Association Rules", 5, 50, 10, 5)
    rules_sorted = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values(by="lift", ascending=False).head(num_rules)

    # **Display Association Rules**
    st.subheader("🔗 Association Rules")
    st.dataframe(rules_sorted)

    # **Choose Visualization**
    st.subheader("📌 Data Visualization")

    if visualization_type == "Network Graph":
        st.subheader("📌 Market Basket Network Graph")
        G = nx.DiGraph()
        for _, rule in rules_sorted.iterrows():
            G.add_edge(rule['antecedents'], rule['consequents'], weight=rule['lift'])
        
        plt.figure(figsize=(12,6))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray')
        st.pyplot(plt)

    elif visualization_type == "Heatmap":
        st.subheader("🔥 Association Rules Heatmap")

    # Create a pivot table with Lift as values
        pivot_data = rules_sorted.pivot(index="antecedents", columns="consequents", values="lift")

    # Plot heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot_data, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5)
        plt.xlabel("Consequents (RHS)")
        plt.ylabel("Antecedents (LHS)")
        plt.title(f"🔥 Association Rules Heatmap ({num_rules} Rules)")
        st.pyplot(plt)



    elif visualization_type == "Bar Chart":
        st.subheader("📊 Bar Charts for Frequent Itemsets & Association Rules")

        # Bar Chart 1: Top Frequent Itemsets by Support
        st.subheader("📌 Top Frequent Itemsets")
        top_items = frequent_itemsets.nlargest(num_rules, 'support')
        plt.figure(figsize=(10, 5))
        plt.barh(top_items['itemsets'].astype(str), top_items['support'], color='skyblue')
        plt.xlabel("Support")
        plt.ylabel("Frequent Itemsets")
        plt.title("Top Frequent Itemsets by Support")
        plt.gca().invert_yaxis()  # Ensure largest appears at the top
        st.pyplot(plt)

        # Bar Chart 2: Top Association Rules by Lift
        st.subheader("🚀 Top Association Rules")
        top_rules = rules_sorted.nlargest(num_rules, 'lift')

        plt.figure(figsize=(10, 5))
        plt.barh(top_rules.apply(lambda x: f"{x['antecedents']} → {x['consequents']}", axis=1), 
             top_rules['lift'], color='salmon')
        plt.xlabel("Lift")
        plt.ylabel("Association Rules")
        plt.title("Top Association Rules by Lift")
        plt.gca().invert_yaxis()  # Ensure largest appears at the top
        st.pyplot(plt)


