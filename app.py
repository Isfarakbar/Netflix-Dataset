import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Page settings
st.set_page_config(page_title="Netflix Data Analysis", layout="wide")
st.title("📺 Netflix Dataset Analysis")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv('netflix_titles.csv')

df = load_data()

# Data cleaning
df['date_added'] = pd.to_datetime(df['date_added'].str.strip(), errors='coerce')
df['year_added'] = df['date_added'].dt.year
df['month_added'] = df['date_added'].dt.month
df['duration_num'] = df['duration'].str.extract('(\d+)').astype(float)
df['duration_type'] = df['duration'].str.extract('([a-zA-Z]+)')
df['country'].fillna('Unknown', inplace=True)
df['rating'].fillna('Not Rated', inplace=True)
df['duration_num'].fillna(df['duration_num'].median(), inplace=True)
df.dropna(subset=['director', 'cast', 'date_added'], inplace=True)

# Sidebar
st.sidebar.header("Navigation")
section = st.sidebar.radio("Go to", ["Dataset Info", "Visualizations", "Statistics", "Regression", "Probability"])

# Section 1: Dataset Info
if section == "Dataset Info":
    st.subheader("📊 Dataset Overview")
    st.write("Shape:", df.shape)
    st.dataframe(df.head(10))

    st.write("### Column Info")
    buffer = []
    df.info(buf=buffer)
    st.text('\n'.join(map(str, buffer)))

    st.write("### Descriptive Statistics")
    st.dataframe(df.describe(include='all'))

# Section 2: Visualizations
elif section == "Visualizations":
    st.subheader("📈 Visualizations")

    st.markdown("### Type Count (Bar Plot)")
    fig1, ax1 = plt.subplots()
    sns.countplot(x='type', data=df, palette='Set2', ax=ax1)
    ax1.set_title('Content Type Count')
    st.pyplot(fig1)

    st.markdown("### Top 5 Countries (Pie Chart)")
    top_countries = df['country'].value_counts().head(5)
    fig2, ax2 = plt.subplots()
    top_countries.plot.pie(autopct='%1.1f%%', colors=sns.color_palette('Set3'), ax=ax2)
    ax2.set_ylabel('')
    ax2.set_title('Top 5 Countries on Netflix')
    st.pyplot(fig2)

    st.markdown("### Release Year Distribution (Histogram)")
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    sns.histplot(df['release_year'], bins=30, kde=True, ax=ax3)
    ax3.set_title('Distribution of Release Years')
    ax3.set_xlabel('Year')
    st.pyplot(fig3)

# Section 3: Statistics
elif section == "Statistics":
    st.subheader("📏 Descriptive Stats & Confidence Intervals")

    durations = df[df['duration_type'] == 'min']['duration_num']
    mean = durations.mean()
    ci = stats.t.interval(0.95, len(durations)-1, loc=mean, scale=stats.sem(durations))

    st.write(f"**Average Duration (minutes):** {mean:.2f}")
    st.write(f"**95% Confidence Interval:** ({ci[0]:.2f}, {ci[1]:.2f})")

    st.markdown("### Frequency Tables")
    st.write("**Type:**")
    st.write(df['type'].value_counts())
    st.write("**Rating:**")
    st.write(df['rating'].value_counts())
    st.write("**Country (Top 10):**")
    st.write(df['country'].value_counts().head(10))

# Section 4: Regression
elif section == "Regression":
    st.subheader("📉 Linear Regression: Year Added vs Duration")

    model_data = df[(df['duration_type'] == 'min') & (df['year_added'].notnull())]
    X = model_data[['year_added']]
    y = model_data['duration_num']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    reg = LinearRegression().fit(X_train, y_train)
    st.write("**Regression Coefficients:**", reg.coef_)
    st.write("**Intercept:**", reg.intercept_)

    fig4, ax4 = plt.subplots()
    ax4.scatter(X_test, y_test, color='blue', alpha=0.5, label='Actual')
    ax4.plot(X_test, reg.predict(X_test), color='red', label='Predicted')
    ax4.set_title('Year Added vs Duration')
    ax4.set_xlabel('Year Added')
    ax4.set_ylabel('Duration (min)')
    ax4.legend()
    st.pyplot(fig4)

# Section 5: Probability
elif section == "Probability":
    st.subheader("🎲 Probability Distribution - Duration")

    durations = df[df['duration_type'] == 'min']['duration_num']
    st.write("**Normal Distribution Fit**")

    mu, std = stats.norm.fit(durations)
    fig5, ax5 = plt.subplots()
    sns.histplot(durations, bins=30, kde=False, stat='density', ax=ax5)
    xmin, xmax = ax5.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    ax5.plot(x, p, 'k', linewidth=2)
    ax5.set_title("Fit Normal Distribution to Duration (minutes)")
    st.pyplot(fig5)

    st.write(f"Mean = {mu:.2f}, Standard Deviation = {std:.2f}")
