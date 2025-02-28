# %%
#pip install streamlit psycopg2-binary plotly xlsxwriter

# %%
import streamlit as st
import pandas as pd
import psycopg2
import plotly.graph_objects as go
from io import BytesIO

# %%
DB_CONFIG = {
    "dbname": "my_db",
    "user": "postgres",
    "password": "citi1234",
    "host": "34.42.254.97",
    "port": "5432"
}

# %%
def get_company_data(company_name):
    conn = psycopg2.connect(**DB_CONFIG)
    query = """
        SELECT question, answer, score_type, score, pillar, explanation
        FROM company_scores
        WHERE company_name = %s
    """
    df = pd.read_sql(query, conn, params=(company_name,))
    conn.close()
    return df

# %%
def get_company_names():
    conn = psycopg2.connect(**DB_CONFIG)
    query = "SELECT DISTINCT company_name FROM company_scores"
    df = pd.read_sql(query, conn)
    conn.close()
    return df["company_name"].tolist()

# %%
def get_aggregated_score(data):
    agg_row = data[data["score_type"] == "c"]
    if not agg_row.empty:
        return agg_row["score"].values[0], agg_row["explanation"].values[0]
    return None, None


def get_pillar_scores(data):
    pillar_data = data[data["score_type"] == "p"]
    return pillar_data[["pillar", "score", "explanation"]]


# %%
# Function to generate Excel file
def generate_excel(data):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        data.to_excel(writer, index=False, sheet_name="Company Scores")
    output.seek(0)
    return output

def calculate_dynamic_score(data):
    pillar_scores = {}
    for _, row in data.iterrows():
        pillar = row["pillar"]
        score = row["score"]
        if pd.notna(pillar) and pd.notna(score):
            pillar_scores[pillar] = pillar_scores.get(pillar, 0) + score
    
    total_score = sum(pillar_scores.values())
    return total_score, pillar_scores

def wrap_text(text, width=80):
    """Wrap text every 'width' characters for better hover display."""
    return '<br>'.join(text[i:i+width] for i in range(0, len(text), width))

def create_treemap_chart(data):
    labels = []
    parents = []
    values = []
    explanations = []
    for _, row in data.iterrows():
        if row["score_type"] == "c":
            labels.append("Aggregated Score")
            parents.append("")
            values.append(row["score"])  # Should be 13
            explanations.append("Overall governance score based on summed values.")
        elif row["score_type"] == "p":
            labels.append(f"Pillar {row['pillar']}")
            parents.append("Aggregated Score")
            values.append(row["score"])  # Only sums its own children
            explanations.append(wrap_text(row["explanation"] , 80))
        else:
            labels.append(row["question"])
            parents.append(f"Pillar {row['pillar']}")
            values.append(row["score"])
            explanations.append(wrap_text(row["answer"], 80))

    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        textinfo="label+value",
        hoverinfo="label",
        customdata=explanations,
        #hovertemplate="<b>%{label}</b><br>Score: %{value}<br>Explanation: %{customdata}",
        hovertemplate="<b>%{label}</b><br>Score: %{value}<br>%{customdata}<extra></extra>",  # Uses <extra></extra> to avoid duplication

        marker=dict(colorscale="viridis")
    ))

    fig.update_layout(
        margin=dict(t=40, l=10, r=10, b=10),
        paper_bgcolor="#f0f2f6"
    )

    return fig

# %%
st.set_page_config(page_title="Sunburst Chart", layout="wide")
st.markdown("""
    <style>
        /* Change background color */
        .stApp {
            background-color: #f0f2f6;  /* Light gray */
        }
        div[data-baseweb="select"] {
            border: 2px solid darkblue !important;
            border-radius: 8px;
            padding: 5px;
        }
        .hoverlayer .hovertext {
            max-width: 500px !important;  /* Increase max width */
            white-space: normal !important;  /* Allow text wrapping */
            word-wrap: break-word !important;
            text-align: left !important;
        }

        div[data-testid="stSelectbox"] *:focus {
            outline: none !important;
            box-shadow: none !important;
            border: none !important;
        }

        label[data-testid="stWidgetLabel"] {
            font-size: 18px;
            font-weight: bold;
            color: darkblue;
            font-family: Arial, sans-serif;
        }
    </style>
""", unsafe_allow_html=True)
st.title("📊 Corporate Governance Score Dashboard")


# %%
companies = get_company_names()



# %%
selected_company = st.selectbox("", companies, index=None, placeholder="Select a Company")

st.write(f"✅ You selected: **{selected_company}**")

# %%
# Display Results

import plotly.io as pio
pio.renderers.default = "browser"
if selected_company:
    data = get_company_data(selected_company)
    total_score, pillar_scores = calculate_dynamic_score(data)

    if not data.empty:
        st.subheader("Governance Score")
        fig = create_treemap_chart(data)
        st.plotly_chart(fig)
        st.download_button(
        label="📥 Download Data as Excel",
        data=generate_excel(data),
        file_name=f"{selected_company}_Governance_Report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )