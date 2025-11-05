import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg
import joblib
import io
import datetime
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Impulse Spending Predictor Dashboard", layout="wide")

@st.cache_data
def load_data():
    try:
        return pd.read_csv("Impulse_Advanced_Feature_Engineered.csv.gz")
    except Exception as e:
        st.error(f"❌ Failed to load data: {e}")
        return pd.DataFrame()

df = load_data()

#st.set_page_config(page_title="Impulse Spending Predictor Dashboard", layout="wide")
st.title("🛍️ Impulse Spending Predictor Dashboard")
st.markdown("Analyze transaction patterns, visualize impulse predictions, and uncover user-specific spending behavior.")

user_ids = df['User_ID'].unique()
selected_user = st.sidebar.selectbox("Select User ID", sorted(user_ids))
score_range = st.sidebar.slider("Impulse Score Range", 0.0, 1.0, (0.0, 1.0), 0.05)

tab1, tab2, tab4 = st.tabs(["📊 Descriptive", "🎯 User Insights", "📤 Real-Time Prediction"])

with tab1:
    filtered_df = df[df['User_ID'] == selected_user]
    st.subheader(f"Transaction Overview for User {selected_user}")

    # --- Visual 1: Impulse vs Non-Impulse by Category ---
    st.subheader("Impulse vs Non-Impulse by Category")
    cat_imp = filtered_df.groupby(['Category', 'Is_Impulse_RuleBased']).size().unstack().fillna(0)
    fig1 = go.Figure(data=[
        go.Bar(name='Non-Impulse', x=cat_imp.index, y=cat_imp[0], marker_color='teal'),
        go.Bar(name='Impulse', x=cat_imp.index, y=cat_imp[1], marker_color='crimson')
    ])
    fig1.update_layout(barmode='stack', title="Category-wise Impulse Classification")
    st.plotly_chart(fig1, use_container_width=True)
    st.caption("📊 Shows how different categories split between impulse and non-impulse purchases.")

    # --- Visual 2: Days Until Salary Distribution ---
    st.subheader("Days Until Salary Distribution (Impulse vs Non-Impulse)")
    fig2 = px.histogram(filtered_df, x='Days_Until_Salary', color='Is_Impulse_RuleBased',
                        barmode='overlay', nbins=30,
                        color_discrete_map={0: 'steelblue', 1: 'darkorange'})
    fig2.update_layout(title="How Salary Cycle Affects Impulse Transactions")
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("💰 Users show more impulsive behavior as the next salary date gets closer.")

    # --- Visual 3: Transaction Count by Hour ---
    st.subheader("Transaction Count by Hour of Day")
    hour_counts = filtered_df.groupby('Transaction_Hour')['Is_Impulse_RuleBased'].value_counts().unstack().fillna(0)
    fig3 = go.Figure(data=[
        go.Bar(name='Non-Impulse', x=hour_counts.index, y=hour_counts[0], marker_color='royalblue'),
        go.Bar(name='Impulse', x=hour_counts.index, y=hour_counts[1], marker_color='orangered')
    ])
    fig3.update_layout(barmode='stack', title="Impulse vs Non-Impulse Across Hours")
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("🕒 Users are more likely to make impulse purchases during specific hours (e.g., late evening).")

    # --- Visual 4: Impulse Rate by Category ---
    st.subheader("Impulse Rate by Category")
    category_risk = filtered_df.groupby('Category')['Is_Impulse_RuleBased'].mean().sort_values(ascending=False)
    fig4 = px.bar(category_risk, x=category_risk.index, y=category_risk.values,
                labels={'x': 'Category', 'y': 'Impulse Rate'},
                color=category_risk.values,
                color_continuous_scale='Reds')
    fig4.update_layout(title="Average Impulse Likelihood per Category")
    st.plotly_chart(fig4, use_container_width=True)
    st.caption("🔥 Highlights the categories with the highest tendency toward impulse spending.")

    # --- Visual 5: Impulse Rate by Hour of Day ---
    st.subheader("Impulse Rate by Hour of Day")
    hourly_impulse = filtered_df.groupby('Transaction_Hour')['Is_Impulse_RuleBased'].mean()
    fig5 = px.bar(hourly_impulse, x=hourly_impulse.index, y=hourly_impulse.values,
                labels={'x': 'Hour of Day', 'y': 'Impulse Rate'},
                color=hourly_impulse.values,
                color_continuous_scale='Viridis')
    fig5.update_layout(title="When Are Users Most Impulsive?")
    st.plotly_chart(fig5, use_container_width=True)
    st.caption("🧠 Reveals the most impulsive times of the day based on user behavior.")

    # --- Visual 6: Spending Amount by Impulse Classification ---
    st.subheader("Spending Amount by Impulse Classification")
    fig6 = px.box(filtered_df, x='Is_Impulse_RuleBased', y='Amount',
                color='Is_Impulse_RuleBased',
                color_discrete_map={0: 'seagreen', 1: 'darkred'})
    fig6.update_layout(title="Do Impulse Transactions Cost More?",
                    xaxis_title="Is Impulse (Rule-Based)", yaxis_title="Amount")
    st.plotly_chart(fig6, use_container_width=True)
    st.caption("💸 Understand whether users spend more during impulsive purchases.")

    # --- Visual 7: Impulse Rate by Transaction Type ---
    if 'Transaction_Type' in filtered_df.columns:
        st.subheader("Impulse Rate by Transaction Type")
        tx_type_impulse = filtered_df.groupby('Transaction_Type')['Is_Impulse_RuleBased'].mean().sort_values(ascending=False)
        fig7 = px.bar(tx_type_impulse, x=tx_type_impulse.index, y=tx_type_impulse.values,
                    color=tx_type_impulse.values, color_continuous_scale='Blues',
                    labels={'x': 'Transaction Type', 'y': 'Impulse Rate'})
        fig7.update_layout(title="How Transaction Types Relate to Impulse Spending")
        st.plotly_chart(fig7, use_container_width=True)
        st.caption("🧾 Shows which transaction types (e.g., Online, In-store) are more impulse-prone.")

with tab2:
    st.header("🎯 User Behavioral Insights")
    user_df = df[df['User_ID'] == selected_user].copy()

    # --- Impulse Scorecards ---
    impulse_rate = user_df['Is_Impulse_RuleBased'].mean() * 100
    impulse_count = user_df[user_df['Is_Impulse_RuleBased'] == 1].shape[0]
    non_impulse_count = user_df[user_df['Is_Impulse_RuleBased'] == 0].shape[0]
    avg_gap = user_df['Time_Since_Last_Transaction'].mean()
    days_since_last = (pd.Timestamp.today() - pd.to_datetime(user_df['Transaction_DateTime']).max()).days

    col1, col2, col3 = st.columns(3)
    col1.metric("💥 Impulse Rate", f"{impulse_rate:.2f}%")
    col2.metric("⏱️ Avg. Time Gap (min)", f"{avg_gap:.1f}")
    

    # --- Category Drilldown ---
    st.subheader("🔍 Impulse Breakdown by Category")
    category = st.selectbox("Select a Category", sorted(user_df['Category'].dropna().unique()))
    cat_df = user_df[user_df['Category'] == category]
    if not cat_df.empty:
        fig_cat = px.histogram(cat_df, x='Transaction_Hour', color='Is_Impulse_RuleBased', barmode='group',
                               title=f"Impulse Distribution by Hour in '{category}'", labels={"Is_Impulse_RuleBased": "Impulse"})
        st.plotly_chart(fig_cat, use_container_width=True)

    # --- Compare with Average User ---
    st.subheader("📊 Compare with Average User")
    comparison_df = pd.DataFrame({
        'Metric': ['Impulse Rate', 'Avg. Time Since Last Txn (min)', 'Transactions Made'],
        'Selected User': [impulse_rate, avg_gap, user_df.shape[0]],
        'Average User': [df['Is_Impulse_RuleBased'].mean() * 100, df['Time_Since_Last_Transaction'].mean(), df.groupby('User_ID').size().mean()]
    })
    fig_compare = px.bar(comparison_df.melt(id_vars='Metric', var_name='User', value_name='Value'),
                         x='Metric', y='Value', color='User', barmode='group',
                         title="Selected User vs. Average User")
    st.plotly_chart(fig_compare, use_container_width=True)

    # Smart Tips Tab
    st.subheader("🧠 Smart Spending Tips for You")

    # Impulse Rate Tip
    impulse_rate = user_df['Is_Impulse_RuleBased'].mean() * 100
    if impulse_rate > 60:
        st.markdown("🔴 _You're acting on impulse very frequently. Consider using the '24-hour rule' before making purchases._")
    elif impulse_rate > 30:
        st.markdown("🟠 _You're occasionally pulled by impulse. Creating a wishlist and revisiting it weekly might help._")
    else:
        st.markdown("🟢 _Great control! Rewarding yourself consciously can reinforce good habits._")

    # Impulse Badge
    if impulse_rate > 60:
        st.info("🔹 You are an **Impulse Hunter** 🔴")
    elif impulse_rate > 30:
        st.info("🔹 You are a **Fence-Sitter** 🟠")
    else:
        st.info("🔹 You are a **Budget Guardian** 🟢")

    # Peak Impulse Hour
    if 'Transaction_Hour' in user_df:
        peak_hour = user_df[user_df['Is_Impulse_RuleBased'] == 1]['Transaction_Hour'].mode().values[0]
        if peak_hour >= 20:
            st.markdown("🌙 _Impulse spikes in the evening. Logging off shopping apps after 8 PM might help._")

    # Short Gaps Between Purchases
    if 'Time_Since_Last_Transaction' in user_df:
        impulse_gap = user_df[user_df['Is_Impulse_RuleBased'] == 1]['Time_Since_Last_Transaction'].mean()
        if impulse_gap < 120:
            st.markdown("⏰ _You often shop with short intervals. Try enforcing a 2-hour cooling period._")

    # Salary Proximity Behavior
    if 'Days_Until_Salary' in user_df:
        avg_days_until_salary = user_df[user_df['Is_Impulse_RuleBased'] == 1]['Days_Until_Salary'].mean()
        if avg_days_until_salary < 5:
            st.markdown("💸 _Impulse purchases increase before payday. Set specific spend limits during this window._")

    # High-Risk Category
    if 'Category_Risk_Score' in user_df:
        top_risky_cats = user_df[user_df['Is_Impulse_RuleBased'] == 1].groupby('Category')['Category_Risk_Score'].mean()
        if not top_risky_cats.empty:
            risky_cat = top_risky_cats.sort_values(ascending=False).index[0]
            st.markdown(f"🛎️ _You're most likely to impulse shop in: **{risky_cat}**. Add those items to a 3-day review list._")

    # Social Media Influence
    if 'Purchase_After_Social_Min' in user_df:
        recent_after_session = user_df[user_df['Purchase_After_Social_Min'] < 10].shape[0]
        total_impulses = user_df[user_df['Is_Impulse_RuleBased'] == 1].shape[0]
        if total_impulses > 0 and recent_after_session / total_impulses > 0.4:
            st.markdown("📲 _Social scrolls are triggering spending. Try screen-free zones before making purchases._")


with tab4:
    st.subheader("📤 Real-Time Prediction from a Single User")
    st.markdown("Upload **transaction data** and **social media session data** for a single user (spanning ~2 months). The system will classify impulse behavior in real time.")

    st.markdown("### Step 1: Upload Transaction Data")
    transaction_file = st.file_uploader("📂 Upload Transaction Data (CSV)", type="csv", key="txn")

    st.markdown("### Step 2: Upload Social Media Session Data")
    session_file = st.file_uploader("📂 Upload Social Media Session Data (CSV)", type="csv", key="social")

    if transaction_file and session_file:
        try:
            transactions = pd.read_csv(transaction_file)
            sessions = pd.read_csv(session_file)
            st.success("✅ Files uploaded successfully!")

            # Single user assumption
            user_id = transactions['User_ID'].iloc[0]
            st.markdown(f"Analyzing data for user: **{user_id}**")

            # Convert timestamps
            transactions['Transaction_Timestamp'] = pd.to_datetime(transactions['Transaction_Timestamp'])
            sessions['Session_Start'] = pd.to_datetime(sessions['Session_Start'])
            sessions['Session_End'] = pd.to_datetime(sessions['Session_End'])

            transactions = transactions.sort_values("Transaction_Timestamp").reset_index(drop=True)
            sessions = sessions.sort_values("Session_End").reset_index(drop=True)

            # Feature engineering
            transactions['Transaction_Hour'] = transactions['Transaction_Timestamp'].dt.hour
            transactions['Days_Until_Salary'] = 30 - transactions['Transaction_Timestamp'].dt.day.clip(upper=30)
            transactions['Is_First_Transaction'] = transactions.index == 0
            transactions['Time_Since_Last_Transaction'] = transactions['Transaction_Timestamp'].diff().dt.total_seconds().div(60).fillna(9999)

            category_risk = df.groupby('Category')['Is_Impulse_RuleBased'].mean().to_dict()
            transactions['Category_Risk_Score'] = transactions['Category'].map(category_risk).fillna(0.5)

            session_avg = df.groupby('User_ID')['Session_Duration_Min'].mean().mean()
            transactions['Avg_Session_Duration_User'] = session_avg

            # Calculate Purchase_After_Social_Min
            purchase_after_social = []
            for tx_time in transactions['Transaction_Timestamp']:
                past_sessions = sessions[sessions['Session_End'] <= tx_time]
                if not past_sessions.empty:
                    last_session_end = past_sessions['Session_End'].max()
                    diff_minutes = (tx_time - last_session_end).total_seconds() / 60.0
                    purchase_after_social.append(diff_minutes)
                else:
                    purchase_after_social.append(9999)
            transactions['Purchase_After_Social_Min'] = purchase_after_social

            # Ad_Same_Hour (1 if a session overlaps same hour as transaction)
            transactions['Ad_Same_Hour'] = transactions['Transaction_Timestamp'].apply(
                lambda tx_time: any(
                    (tx_time >= s) and (tx_time <= e)
                    for s, e in zip(sessions['Session_Start'], sessions['Session_End'])
                )
            ).astype(int)

            # Load trained model and predict
            model = joblib.load("final_catboost_model.pkl")
            feature_order = [
                'Time_Since_Last_Transaction',
                'Is_First_Transaction',
                'Avg_Session_Duration_User',
                'Days_Until_Salary',
                'Purchase_After_Social_Min',
                'Ad_Same_Hour',
                'Amount',
                'Transaction_Hour',
                'Category_Risk_Score',
            ]
            transactions['Predicted_Is_Impulse'] = model.predict(transactions[feature_order])
            transactions['Impulse_Score'] = model.predict_proba(transactions[feature_order])[:, 1]

            # Results
            # st.markdown("### ✅ Predicted Impulse Transactions")
            # st.dataframe(
            #     transactions[['Transaction_Timestamp', 'Amount', 'Category', 'Predicted_Is_Impulse', 'Impulse_Score',]],
            #     use_container_width=True
            # )

            try:
                transactions['Week'] = transactions['Transaction_Timestamp'].dt.to_period('W').apply(lambda r: r.start_time)
                weekly_rates = transactions.groupby('Week')['Predicted_Is_Impulse'].mean().sort_index()

                st.subheader("📈 Weekly Impulse Trend")

                if len(weekly_rates) >= 2:
                    last_week, prev_week = weekly_rates.iloc[-1], weekly_rates.iloc[-2]
                    change = last_week - prev_week

                    if change > 0.01:
                        st.info(f"🔺 Your impulse spending increased by **{change * 100:.1f}%** compared to the previous week.")
                    elif change < -0.01:
                        st.info(f"🔻 Your impulse spending decreased by **{abs(change * 100):.1f}%** compared to the previous week.")
                    else:
                        st.info("ℹ️ Your impulse spending stayed about the same as last week.")
                else:
                    st.info("ℹ️ Not enough weekly data to compare impulse behavior.")
            except Exception as e:
                st.warning(f"⚠️ Could not generate weekly trend: {e}")

            # Download
            csv = transactions.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Predictions", data=csv, file_name="user_impulse_predictions.csv")

                        # --- VISUALIZATIONS ---
            st.markdown("### 📈 Visual Insights from Predictions")

            # 1. Impulse Rate Over Time
            transactions['Week'] = transactions['Transaction_Timestamp'].dt.to_period('W').apply(lambda r: r.start_time)
            weekly_impulse = transactions.groupby('Week')['Predicted_Is_Impulse'].mean() * 100
            fig1 = px.line(weekly_impulse, title="Impulse Rate Over Time (Weekly)", markers=True, labels={'value': 'Impulse Rate (%)', 'Week': 'Week'})
            st.plotly_chart(fig1, use_container_width=True)

            # 2. Spending Amount by Impulse Classification
            fig2 = px.box(transactions, x='Predicted_Is_Impulse', y='Amount', color='Predicted_Is_Impulse', 
                         title="Spending Amount by Impulse Classification", 
                         labels={'Predicted_Is_Impulse': 'Impulse Prediction'})
            st.plotly_chart(fig2, use_container_width=True)

            # 3. Weekly Goal Tracker (Compare % Impulse)
            impulse_by_week = transactions.groupby('Week')['Predicted_Is_Impulse'].mean()
            if len(impulse_by_week) >= 2:
                last_two = impulse_by_week.sort_index().iloc[-2:]
                delta = (last_two.iloc[-1] - last_two.iloc[0]) * 100
                arrow = "⬆️" if delta > 0 else ("⬇️" if delta < 0 else "🟰")
                trend_msg = f"{arrow} Your impulse spending {'increased' if delta > 0 else 'decreased' if delta < 0 else 'stayed the same'} by {abs(delta):.2f}% compared to the previous week."
                st.info(trend_msg)
            else:
                st.info("ℹ️ Not enough weekly data to compare impulse behavior.")

            # 4. Days Until Salary vs Impulse Score
            fig4 = px.histogram(transactions, x='Days_Until_Salary', color='Predicted_Is_Impulse', barmode='overlay',
                                title="Impulse Prediction vs Days Until Salary",
                                labels={'Predicted_Is_Impulse': 'Impulse Prediction'})
            st.plotly_chart(fig4, use_container_width=True)

            # Download
            csv = transactions.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Predictions", data=csv, file_name="user_impulse_predictions.csv")

        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")

    else:
        st.info("⬆️ Please upload both required CSV files to begin prediction.")

