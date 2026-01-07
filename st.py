import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from scipy.stats import poisson
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Khel-Metrics | Premier League Predictor",
    page_icon="",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<script>
const isMobile = window.innerWidth < 768;
window.parent.postMessage(
    { type: "STREAMLIT_MOBILE", isMobile },
    "*"
);
</script>
""", unsafe_allow_html=True)

if "is_mobile" not in st.session_state:
    st.session_state.is_mobile = st.experimental_get_query_params().get("is_mobile", [False])[0] == "true"



# Custom CSS for beautiful styling
st.markdown("""
<style>
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Card styling */
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        color: white;
        font-size: 3.5em;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        margin-bottom: 10px;
        font-family: 'Arial Black', sans-serif;
    }
    
    .sub-header {
        text-align: center;
        color: #f0f0f0;
        font-size: 1.3em;
        margin-bottom: 30px;
    }
    
    /* Team name styling */
    .team-name {
        font-size: 1.5em;
        font-weight: bold;
        color: #2c3e50;
        margin: 10px 0;
    }
    
    /* Probability bars */
    .prob-container {
        background: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Stats box */
    .stats-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin: 5px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    
    /* Match history card */
    .match-card {
        background: white;
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.2em;
        font-weight: bold;
        padding: 15px 40px;
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.95);
    }
    
    /* Table position badges */
    .pos-badge {
        display: inline-block;
        width: 30px;
        height: 30px;
        border-radius: 50%;
        text-align: center;
        line-height: 30px;
        font-weight: bold;
        color: white;
        margin-right: 10px;
    }
    
    .pos-1 { background: #FFD700; color: #000; }
    .pos-2-4 { background: #4CAF50; }
    .pos-relegation { background: #f44336; }
    .pos-other { background: #9E9E9E; }

    /* Compact table rows */
    .table-row .table-metric-label {
        font-size: 0.9em;
        color: #000000;
    }
    .table-row .table-metric-value {
        font-size: 1.4em;
        font-weight: bold;
    }
    .team-name-sm {
        font-size: 1.2em;
        font-weight: bold;
        color: #2c3e50;
    }

    @media (max-width: 768px) {
        .table-row {
            padding: 8px 10px !important;
            margin: 4px 0 !important;
        }
        .table-row .pos-badge {
            width: 26px;
            height: 26px;
            line-height: 26px;
            font-size: 0.9em;
            margin-right: 8px;
        }
        .table-row img {
            width: 28px !important;
            margin-right: 6px;
        }
        .team-name-sm {
            font-size: 1.05em;
        }
        .table-row .table-metric-label {
            font-size: 0.75em;
        }
        .table-row .table-metric-value {
            font-size: 1.1em;
        }
        .table-row > div {
            gap: 8px !important;
        }

        /* Stack flex rows vertically on mobile */
        .metric-card > div {
            flex-direction: column !important;
            align-items: flex-start !important;
            gap: 12px !important;
        }

        /* Ensure inner flex containers also wrap */
        .metric-card div[style*="display: flex"] {
            flex-direction: column !important;
            align-items: flex-start !important;
            gap: 10px !important;
        }

        /* Center logos & badges on mobile */
        .metric-card img {
            margin-bottom: 6px;
        }

        /* Prevent text overflow */
        .metric-card span,
        .metric-card div {
            max-width: 100%;
            word-wrap: break-word;
        }
    }

</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_team_logos():
    """Cache team logos dictionary"""
    return {
    "Manchester City": "https://resources.premierleague.com/premierleague/badges/t43.svg",
    "Arsenal": "https://resources.premierleague.com/premierleague/badges/t3.svg",
    "Liverpool": "https://resources.premierleague.com/premierleague/badges/t14.svg",
    "Chelsea": "https://resources.premierleague.com/premierleague/badges/t8.svg",
    "Manchester Utd": "https://resources.premierleague.com/premierleague/badges/t1.svg",
    "Tottenham": "https://resources.premierleague.com/premierleague/badges/t6.svg",
    "Newcastle Utd": "https://resources.premierleague.com/premierleague/badges/t4.svg",
    "Brighton": "https://resources.premierleague.com/premierleague/badges/t36.svg",
    "Brentford": "https://resources.premierleague.com/premierleague/badges/t94.svg",
    "Crystal Palace": "https://resources.premierleague.com/premierleague/badges/t31.svg",
    "Aston Villa": "https://resources.premierleague.com/premierleague/badges/t7.svg",
    "Fulham": "https://upload.wikimedia.org/wikipedia/en/e/eb/Fulham_FC_%28shield%29.svg",
    "West Ham": "https://resources.premierleague.com/premierleague/badges/t21.svg",
    "Wolves": "https://resources.premierleague.com/premierleague/badges/t39.svg",
    "Everton": "https://resources.premierleague.com/premierleague/badges/t11.svg",
    "Bournemouth": "https://resources.premierleague.com/premierleague/badges/t91.svg",
    "Nott'ham Forest": "https://resources.premierleague.com/premierleague/badges/t17.svg",
    "Leicester City": "https://resources.premierleague.com/premierleague/badges/t13.svg",
    "Ipswich Town": "https://resources.premierleague.com/premierleague/badges/t40.svg",
    "Southampton": "https://resources.premierleague.com/premierleague/badges/t20.svg",
    #burnley,leeds,sunderland
    "Burnley": "https://resources.premierleague.com/premierleague25/badges-alt/90.svg",
    "Leeds United":"https://resources.premierleague.com/premierleague25/badges-alt/2.svg",
    "Sunderland":"https://resources.premierleague.com/premierleague25/badges-alt/56.svg"
        # ... rest of your logos
    }

# Team logo URLs (using placeholder - replace with actual URLs)
TEAM_LOGOS =get_team_logos()

@st.cache_data
def load_data():
    """Load all required data files"""
    try:
        simulated_matches = pd.read_csv("simulated_matches_with_probs.csv")
        season_simulation = pd.read_csv("season_simulation_distributions.csv")
        historical_data = pd.read_csv("data2_filled_safe.csv")
        historical_data['match_date'] = pd.to_datetime(historical_data['match_date'])
        return simulated_matches, season_simulation, historical_data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

@st.cache_resource
def load_models():
    """Load trained models"""
    try:
        home_model = joblib.load("home_xgb_poisson.pkl")
        away_model = joblib.load("away_xgb_poisson.pkl")
        feature_cols = joblib.load("feature_cols.pkl")
        return home_model, away_model, feature_cols
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

def get_team_logo(team_name):
    """Get team logo HTML"""
    logo_url = TEAM_LOGOS.get(team_name, "")
    if logo_url:
        return f'<img src="{logo_url}" width="50" style="vertical-align: middle; margin-right: 10px;">'
    return ""

def render_stats_box(label, value):
    """Reusable stat pill with improved hierarchy."""
    return f"""
    <div class="stats-box">
        <div style="font-size: 0.85em; letter-spacing: 0.5px; opacity: 0.85; text-transform: uppercase;">
            {label}
        </div>
        <div style="font-size: 1.9em; font-weight: 800; margin-top: 6px; color: #f7f7ff;">
            {value}
        </div>
    </div>
    """

def display_home_history(matches_df, home_team, away_team):
    """Render recent matches for the selected home team."""
    display_match_history(matches_df, f"{home_team} - Last 5 Matches", home_team, away_team)

def display_away_history(matches_df, away_team, home_team):
    """Render recent matches for the selected away team."""
    display_match_history(matches_df, f"{away_team} - Last 5 Matches", away_team, home_team)


def display_prediction_result(home_team, away_team, pred_home, pred_away, prob_home, prob_draw, prob_away):

    st.markdown("---")
    st.markdown(
        "<h2 style='text-align: center; color: white;'>Match Prediction</h2>",
        unsafe_allow_html=True
    )

    # ---------- CARD HTML (single source of truth) ----------
    home_card_html = f"""
    <div class="metric-card" style="text-align: center;">
        {get_team_logo(home_team)}
        <div class="team-name">{home_team}</div>
        <div style="font-size: 3em; font-weight: bold; color: #667eea;">
            {pred_home:.1f}
        </div>
        <div style="color: #666;">Expected Goals</div>
    </div>
    """

    away_card_html = f"""
    <div class="metric-card" style="text-align: center;">
        {get_team_logo(away_team)}
        <div class="team-name">{away_team}</div>
        <div style="font-size: 3em; font-weight: bold; color: #764ba2;">
            {pred_away:.1f}
        </div>
        <div style="color: #666;">Expected Goals</div>
    </div>
    """

    # ---------- RESPONSIVE LAYOUT ----------
    if st.session_state.get("is_mobile", False):
        # Stacked (mobile portrait)
        st.markdown(home_card_html, unsafe_allow_html=True)
        st.markdown(
            "<div style='text-align:center;color:white;font-size:1.5em;margin:10px 0;'>VS</div>",
            unsafe_allow_html=True
        )
        st.markdown(away_card_html, unsafe_allow_html=True)
    else:
        # Side-by-side (desktop / landscape)
        col1, col2, col3 = st.columns([2, 1, 2])
        with col1:
            st.markdown(home_card_html, unsafe_allow_html=True)
        with col2:
            st.markdown(
                "<div style='padding-top:80px;text-align:center;color:white;font-size:2em;'>VS</div>",
                unsafe_allow_html=True
            )
        with col3:
            st.markdown(away_card_html, unsafe_allow_html=True)

    # ---------- PROBABILITY BAR CHART ----------
    st.markdown(
        "<h3 style='text-align: center; color: white; margin-top: 30px;'>"
        "Match Outcome Probabilities</h3>",
        unsafe_allow_html=True
    )

    fig = go.Figure(data=[
        go.Bar(
            x=[f'{home_team} Win', 'Draw', f'{away_team} Win'],
            y=[prob_home * 100, prob_draw * 100, prob_away * 100],
            marker_color=['#667eea', '#95a5a6', '#764ba2'],
            text=[
                f'{prob_home*100:.1f}%',
                f'{prob_draw*100:.1f}%',
                f'{prob_away*100:.1f}%'
            ],
            textposition='auto'
        )
    ])

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=14),
        height=300 if st.session_state.get("is_mobile", False) else 400,
        showlegend=False,
        yaxis=dict(title='Probability (%)', range=[0, 100]),
        xaxis=dict(title='')
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---------- MOST LIKELY OUTCOME ----------
    outcomes = [
        (prob_home, f"{home_team} Win"),
        (prob_draw, "Draw"),
        (prob_away, f"{away_team} Win")
    ]
    most_likely = max(outcomes, key=lambda x: x[0])

    st.markdown(f"""
    <div class="metric-card"
         style="text-align: center;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;">
        <h3>Most Likely Outcome</h3>
        <div style="font-size: 2em; font-weight: bold; margin: 10px 0;">
            {most_likely[1]}
        </div>
        <div style="font-size: 1.2em;">
            Probability: {most_likely[0]*100:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)


def display_match_history(matches_df, title, home_team, away_team):
    """Display match history in a beautiful format"""
    st.markdown(f"<h3 style='color: white;'>{title}</h3>", unsafe_allow_html=True)
    
    if matches_df.empty:
        st.info("No recent matches found")
        return
    
    # Filter matches before November 7, 2025
    cutoff_date = pd.to_datetime("2025-11-07")
    matches_df = matches_df[matches_df['match_date'] < cutoff_date]
    
    if matches_df.empty:
        st.info("No matches found before November 7, 2025")
        return
    
    for idx, match in matches_df.iterrows():
        match_date = pd.to_datetime(match['match_date']).strftime('%d %b %Y')
        h_team = match['home_team']
        a_team = match['away_team']
        h_goals = match.get('home_goals', '?')
        a_goals = match.get('away_goals', '?')
        
        # Determine result color for the team
        if pd.notna(h_goals) and pd.notna(a_goals):
            h_goals = int(h_goals)
            a_goals = int(a_goals)
            
            if h_team == home_team or a_team == home_team:
                if (h_team == home_team and h_goals > a_goals) or (a_team == home_team and a_goals > h_goals):
                    border_color = "#4CAF50"  # Win
                elif h_goals == a_goals:
                    border_color = "#FFC107"  # Draw
                else:
                    border_color = "#f44336"  # Loss
            else:
                border_color = "#667eea"
        else:
            border_color = "#9E9E9E"
            h_goals = "?"
            a_goals = "?"
        
        st.markdown(f"""
        <div class="match-card" style="border-left-color: {border_color};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="flex: 1;">
                    <strong>{h_team}</strong>
                </div>
                <div style="flex: 0 0 100px; text-align: center; font-size: 1.5em; font-weight: bold; color: {border_color};">
                    {h_goals} - {a_goals}
                </div>
                <div style="flex: 1; text-align: right;">
                    <strong>{a_team}</strong>
                </div>
            </div>
            <div style="text-align: center; color: #666; font-size: 0.9em; margin-top: 5px;">
                {match_date}
            </div>
        </div>
        """, unsafe_allow_html=True)

def get_recent_matches(historical_data, team, cutoff_date, limit=5):
    """Return the most recent matches for a team before a cutoff date."""
    return historical_data[
        (
            (historical_data["home_team"] == team)
            | (historical_data["away_team"] == team)
        )
        & (historical_data["match_date"] < cutoff_date)
    ].sort_values("match_date", ascending=False).head(limit)


def calculate_team_stats(matches_df, team):
    """Compute simple per-match averages for a team from recent matches."""
    if matches_df is None or matches_df.empty:
        return {"shots": np.nan, "possession": np.nan, "goals": np.nan}

    df = matches_df.copy()
    df["team_shots"] = np.where(
        df["home_team"] == team, df["home_shots"], df["away_shots"]
    )
    df["team_possession"] = np.where(
        df["home_team"] == team, df["home_possession"], df["away_possession"]
    )
    df["team_goals"] = np.where(
        df["home_team"] == team, df["home_goals"], df["away_goals"]
    )

    return {
        "shots": df["team_shots"].mean(),
        "possession": df["team_possession"].mean(),
        "goals": df["team_goals"].mean(),
    }


def compute_match_intensity(pred_home, pred_away, prob_home, prob_draw, prob_away):
    """
    Estimate match intensity using expected total goals and balance of outcomes.
    - Higher combined xG -> more events.
    - Closer win probabilities + decent draw chance -> tighter contest.
    """
    total_xg = pred_home + pred_away
    max_prob = max(prob_home, prob_draw, prob_away)
    draw_bonus = 0.3 if prob_draw >= 0.25 else 0
    balance_bonus = 0.4 if max_prob <= 0.55 else 0.1 if max_prob <= 0.65 else 0
    intensity_score = total_xg + draw_bonus + balance_bonus

    if intensity_score >= 3.6:
        return "High"
    if intensity_score >= 2.6:
        return "Medium"
    return "Low"


def match_prediction_page():
    """Main match prediction page"""
    st.markdown('<div class="main-header"> KHEL-METRICS</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Premier League Match Predictor | XGBoost Powered</div>', unsafe_allow_html=True)
    
    # Load data
    simulated_matches, _, historical_data = load_data()
    home_model, away_model, feature_cols = load_models()
    
    if simulated_matches is None or home_model is None:
        st.error("Failed to load required files. Please ensure all model files are present.")
        return
    
    # Team selection
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    available_teams = sorted(pd.unique(pd.concat([
        simulated_matches['home_team_name'], 
        simulated_matches['away_team_name']
    ])))
    
    with col1:
        st.markdown("###  Home Team")
        home_team = st.selectbox("Select Home Team", available_teams, key="home")
    
    with col2:
        st.markdown("###  Away Team")
        away_team = st.selectbox("Select Away Team", available_teams, key="away")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Predict button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        predict_button = st.button(" PREDICT MATCH", use_container_width=True)
    
    if predict_button:
        if home_team == away_team:
            st.warning(" Please select different teams!")
            return
        
        with st.spinner(" Analyzing match data..."):
            # Get prediction from simulated matches
            match_pred = simulated_matches[
                (simulated_matches['home_team_name'] == home_team) & 
                (simulated_matches['away_team_name'] == away_team)
            ]
            
            if not match_pred.empty:
                match_pred = match_pred.iloc[0]
                pred_home = match_pred['exp_home_goals']
                pred_away = match_pred['exp_away_goals']
                prob_home = match_pred['prob_home_win']
                prob_draw = match_pred['prob_draw']
                prob_away = match_pred['prob_away_win']
            else:
                # Fallback: use default values
                pred_home = 1.5
                pred_away = 1.2
                prob_home = 0.45
                prob_draw = 0.25
                prob_away = 0.30

            cutoff_date = pd.to_datetime("2025-11-07")
            home_recent = get_recent_matches(historical_data, home_team, cutoff_date)
            away_recent = get_recent_matches(historical_data, away_team, cutoff_date)
            home_stats = calculate_team_stats(home_recent, home_team)
            away_stats = calculate_team_stats(away_recent, away_team)

            def fmt_stat(value, suffix=""):
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    return "N/A"
                return f"{value:.1f}{suffix}"
            
            # Display prediction
            display_prediction_result(home_team, away_team, pred_home, pred_away, 
                                    prob_home, prob_draw, prob_away)
            
            # Match statistics
            st.markdown("<h2 style='text-align: center; color: white; margin-top: 40px;'> Additional Statistics</h2>", unsafe_allow_html=True)
            
            stats = [
                (" Expected Goals (H/A)", f"{pred_home:.1f} / {pred_away:.1f}"),
                (" Shots per game (H/A)", f"{fmt_stat(home_stats['shots'])} / {fmt_stat(away_stats['shots'])}"),
                (" Possession (H/A)", f"{fmt_stat(home_stats['possession'], '%')} / {fmt_stat(away_stats['possession'], '%')}"),
                (" Goals per game (H/A)", f"{fmt_stat(home_stats['goals'])} / {fmt_stat(away_stats['goals'])}")
            ]

            if st.session_state.is_mobile:
                for label, value in stats:
                    st.markdown(render_stats_box(label, value), unsafe_allow_html=True)
            else:
                stat_cols = st.columns(4)
                for col, (label, value) in zip(stat_cols, stats):
                    with col:
                        st.markdown(render_stats_box(label, value), unsafe_allow_html=True)

            
            intensity = compute_match_intensity(pred_home, pred_away, prob_home, prob_draw, prob_away)
            summary_stats = [
                (" Expected Winner", max([(prob_home, home_team), (prob_draw, "Draw"), (prob_away, away_team)], key=lambda x: x[0])[1]),
                (" Total Goals (xG)", f"{pred_home + pred_away:.1f}"),
                (" Goal Difference (xG)", f"{abs(pred_home - pred_away):.1f}"),
                (" Match Intensity", intensity)
            ]
            
            if st.session_state.is_mobile:
                for label, value in summary_stats:
                    st.markdown(f"""
                    <div class="stats-box">
                        <div style="font-size: 0.9em; opacity: 0.9;">{label}</div>
                        <div style="font-size: 1.5em; font-weight: bold; margin-top: 5px;">{value}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                summary_cols = st.columns(4)
                for col, (label, value) in zip(summary_cols, summary_stats):
                    with col:
                        st.markdown(f"""
                        <div class="stats-box">
                            <div style="font-size: 0.9em; opacity: 0.9;">{label}</div>
                            <div style="font-size: 1.5em; font-weight: bold; margin-top: 5px;">{value}</div>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Match history section
            st.markdown("<h2 style='text-align: center; color: white; margin-top: 40px;'> Match History</h2>", unsafe_allow_html=True)
            
            if st.session_state.is_mobile:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                display_home_history(home_recent, home_team, away_team)
                st.markdown("</div><br>", unsafe_allow_html=True)
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                display_away_history(away_recent, away_team, home_team)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                history_cols = st.columns(2)
                with history_cols[0]:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    display_home_history(home_recent, home_team, away_team)
                    st.markdown("</div>", unsafe_allow_html=True)
                with history_cols[1]:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    display_away_history(away_recent, away_team, home_team)
                    st.markdown("</div>", unsafe_allow_html=True)
            
            # Head to head
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            h2h_matches = historical_data[
                (((historical_data['home_team'] == home_team) & (historical_data['away_team'] == away_team)) |
                ((historical_data['home_team'] == away_team) & (historical_data['away_team'] == home_team))) &
                (historical_data['match_date'] < cutoff_date)
            ].sort_values('match_date', ascending=False).head(4)
            
            display_match_history(h2h_matches, f" Head-to-Head (Last 4 Meetings)", home_team, away_team)
            st.markdown("</div>", unsafe_allow_html=True)

def season_simulation_page():
    """Season simulation table page"""
    st.markdown('<div class="main-header"> SEASON SIMULATION</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Premier League 2025/26 - Monte Carlo Simulation (10,000 iterations)</div>', unsafe_allow_html=True)
    
    # Load data
    _, season_simulation, _ = load_data()
    
    if season_simulation is None:
        st.error("Failed to load season simulation data.")
        return
    
    # Sort by average position
    season_simulation = season_simulation.sort_values('avg_position').reset_index(drop=True)
    
    # Key metrics at top
    st.markdown("<h2 style='text-align: center; color: white;'> Key Statistics</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        title_favorite = season_simulation.iloc[0]
        st.markdown(f"""
        <div class="metric-card" style="text-align: center; background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);">
            {get_team_logo(title_favorite['team'])}
            <div style="font-size: 1.2em; font-weight: bold; color: #000; margin: 10px 0;">Title Favorite</div>
            <div style="font-size: 1.5em; font-weight: bold; color: #000;">{title_favorite['team']}</div>
            <div style="font-size: 1.1em; color: #333;">{title_favorite['prob_win_league']*100:.1f}% chance</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        top_points = season_simulation.iloc[0]['avg_points']
        st.markdown(f"""
        <div class="metric-card" style="text-align: center; background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); color: white;">
            <div style="font-size: 2.5em; margin: 10px 0;"></div>
            <div style="font-size: 1.2em; font-weight: bold;">Highest Avg Points</div>
            <div style="font-size: 2em; font-weight: bold; margin: 10px 0;">{top_points:.1f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        relegation_battle = season_simulation.iloc[-1]
        st.markdown(f"""
        <div class="metric-card" style="text-align: center; background: linear-gradient(135deg, #f44336 0%, #da190b 100%); color: white;">
            {get_team_logo(relegation_battle['team'])}
            <div style="font-size: 1.2em; font-weight: bold; margin: 10px 0;">Relegation Risk</div>
            <div style="font-size: 1.5em; font-weight: bold;">{relegation_battle['team']}</div>
            <div style="font-size: 1.1em;">{relegation_battle['prob_relegation']*100:.1f}% chance</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        total_teams = len(season_simulation)
        st.markdown(f"""
        <div class="metric-card" style="text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
            <div style="font-size: 2.5em; margin: 10px 0;"></div>
            <div style="font-size: 1.2em; font-weight: bold;">Total Teams</div>
            <div style="font-size: 2em; font-weight: bold; margin: 10px 0;">{total_teams}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Premier League Table
    st.markdown("<h2 style='text-align: center; color: white; margin-top: 40px;'> Simulated Premier League Table</h2>", unsafe_allow_html=True)
    
    # Create beautiful table
    for idx, row in season_simulation.iterrows():
        position = idx + 1
        team = row['team']
        avg_points = row['avg_points']
        avg_position = row['avg_position']
        prob_win = row['prob_win_league'] * 100
        prob_top4 = row['prob_top4'] * 100
        prob_relegation = row['prob_relegation'] * 100
        
        # Position badge color
        if position == 1:
            badge_class = "pos-1"
        elif position <= 4:
            badge_class = "pos-2-4"
        elif position >= len(season_simulation) - 2:
            badge_class = "pos-relegation"
        else:
            badge_class = "pos-other"
        
        # Row color based on position
        if position == 1:
            # Gold
            bg_gradient = "linear-gradient(90deg, rgba(255,215,0,0.45) 0%, rgba(255,255,255,0.95) 100%)"

        elif position <= 4:
            # Champions League: Green
            bg_gradient = "linear-gradient(90deg, rgba(76,175,80,0.45) 0%, rgba(255,255,255,0.95) 100%)"

        elif position == 5:
            # Europa League: Light green (#90EE90)
            bg_gradient = "linear-gradient(90deg, rgba(144,238,144,0.45) 0%, rgba(255,255,255,0.95) 100%)"

        elif position >= len(season_simulation) - 2:
            # Bottom 3: Red
            bg_gradient = "linear-gradient(90deg, rgba(244,67,54,0.45) 0%, rgba(255,255,255,0.95) 100%)"

        else:
            # Others: Purple/blue tone
            bg_gradient = "linear-gradient(90deg, rgba(102,126,234,0.35) 0%, rgba(255,255,255,0.95) 100%)"

        
        st.markdown(f"""
        <div class="metric-card table-row" style="background: {bg_gradient}; margin: 5px 0; padding: 15px;">
            <div style="display: flex; align-items: center; justify-content: space-between; gap: 12px;">
                <div style="display: flex; align-items: center; flex: 2;">
                    <span class="pos-badge {badge_class}">{position}</span>
                    {get_team_logo(team)}
                    <span class="team-name-sm">{team}</span>
                </div>
                <div style="flex: 3; display: flex; justify-content: space-around; align-items: center;">
                    <div class="table-metric">
                        <div class="table-metric-label">Avg Points</div>
                        <div class="table-metric-value" style="color: #667eea;">{avg_points:.1f}</div>
                    </div>
                    <div class="table-metric">
                        <div class="table-metric-label">Win League</div>
                        <div class="table-metric-value" style="color: #734F96;">{prob_win:.1f}%</div>
                    </div>
                    <div class="table-metric">
                        <div class="table-metric-label">Top 4</div>
                        <div class="table-metric-value" style="color: #4CAF50;">{prob_top4:.1f}%</div>
                    </div>
                    <div class="table-metric">
                        <div class="table-metric-label">Relegation</div>
                        <div class="table-metric-value" style="color: #f44336;">{prob_relegation:.1f}%</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Probability distribution charts
    st.markdown("<h2 style='text-align: center; color: white; margin-top: 40px;'> Probability Distributions</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Title race chart
        top_teams = season_simulation.nlargest(8, 'prob_win_league')
        
        fig = go.Figure(data=[
            go.Bar(
                x=top_teams['team'],
                y=top_teams['prob_win_league'] * 100,
                marker_color=['#FFD700', '#C0C0C0', '#CD7F32', '#667eea', '#764ba2', '#9b59b6', '#3498db', '#e74c3c'],
                text=[f'{val*100:.1f}%' for val in top_teams['prob_win_league']],
                textposition='auto',
                textfont=dict(size=14, color='white', family='Arial')
            )
        ])
        
        fig.update_layout(
            title=' Title Race Probabilities',
            plot_bgcolor='rgba(255,255,255,0.9)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#2c3e50', size=12),
            height=400,
            showlegend=False,
            yaxis=dict(title='Probability (%)', gridcolor='rgba(0,0,0,0.1)'),
            xaxis=dict(title='', tickangle=-45)
        )
        
        fig.update_layout(height=300 if st.session_state.is_mobile else 400)
        st.plotly_chart(fig, use_container_width=True)

    
    with col2:
        # Top 4 race chart
        top4_teams = season_simulation.nlargest(10, 'prob_top4')
        
        fig = go.Figure(data=[
            go.Bar(
                x=top4_teams['team'],
                y=top4_teams['prob_top4'] * 100,
                marker_color='#4CAF50',
                text=[f'{val*100:.1f}%' for val in top4_teams['prob_top4']],
                textposition='auto',
                textfont=dict(size=14, color='white', family='Arial')
            )
        ])
        
        fig.update_layout(
            title=' Top 4 Probabilities',
            plot_bgcolor='rgba(255,255,255,0.9)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#2c3e50', size=12),
            height=400,
            showlegend=False,
            yaxis=dict(title='Probability (%)', gridcolor='rgba(0,0,0,0.1)'),
            xaxis=dict(title='', tickangle=-45)
        )
        
        fig.update_layout(height=300 if st.session_state.is_mobile else 400)
        st.plotly_chart(fig, use_container_width=True)

    
    # Relegation battle
    st.markdown("<h3 style='text-align: center; color: white; margin-top: 20px;'>⚠️ Relegation Battle</h3>", unsafe_allow_html=True)
    
    relegation_teams = season_simulation.nlargest(8, 'prob_relegation')
    
    fig = go.Figure(data=[
        go.Bar(
            x=relegation_teams['team'],
            y=relegation_teams['prob_relegation'] * 100,
            marker_color='#f44336',
            text=[f'{val*100:.1f}%' for val in relegation_teams['prob_relegation']],
            textposition='auto',
            textfont=dict(size=14, color='white', family='Arial')
        )
    ])
    
    fig.update_layout(
        plot_bgcolor='rgba(255,255,255,0.9)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#2c3e50', size=12),
        height=400,
        showlegend=False,
        yaxis=dict(title='Relegation Probability (%)', gridcolor='rgba(0,0,0,0.1)'),
        xaxis=dict(title='', tickangle=-45)
    )

    fig.update_layout(height=300 if st.session_state.is_mobile else 400)
    st.plotly_chart(fig, use_container_width=True)

    
    # Points distribution
    st.markdown("<h2 style='text-align: center; color: white; margin-top: 40px;'> Expected Points Distribution</h2>", unsafe_allow_html=True)
    
    fig = go.Figure(data=[
        go.Bar(
            x=season_simulation['team'],
            y=season_simulation['avg_points'],
            error_y=dict(type='data', array=season_simulation['std_points'], visible=True),
            marker_color=[
                '#FFD700' if i == 0 else '#4CAF50' if i < 4 else '#90EE90' if i == 4 else  '#f44336' if i >= len(season_simulation) - 3 else '#667eea'
                for i in range(len(season_simulation))
            ],
            text=[f'{val:.1f}' for val in season_simulation['avg_points']],
            textposition='outside',
            textfont=dict(size=12, color='white', family='Arial')
        )
    ])
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        
        font=dict(color='white', size=14),        
        
        legend=dict(
            font=dict(
                color='black',                   
                size=14
            )
        ),
        
        showlegend=True
    )

    
    fig.update_layout(height=300 if st.session_state.is_mobile else 400)
    st.plotly_chart(fig, use_container_width=True)

    
    # Legend
    st.markdown("""
    <div style="text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 25px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); margin: 20px 0;">
        <h4 style="color: #ffffff; margin: 0 0 20px 0; font-size: 1.4em; font-weight: 600;">Legend</h4>
        <div style="display: flex; justify-content: center; gap: 25px; align-items: center; flex-wrap: wrap;">
            <div style="display: flex; flex-direction: column; align-items: center; gap: 8px;">
                <div style="background: linear-gradient(135deg, #ffcc00 0%, #ffd633 100%); min-width: 60px; height: 50px; display: flex; align-items: center; justify-content: center; border-radius: 8px; font-weight: bold; font-size: 1.1em; color: #2c3e50; box-shadow: 0 2px 6px rgba(255, 204, 0, 0.4);">
                    1
                </div>
                <span style="color: #ffffff; font-size: 0.9em; font-weight: 500;">Champion</span>
            </div>
            <div style="display: flex; flex-direction: column; align-items: center; gap: 8px;">
                <div style="background: linear-gradient(135deg, #4CAF50 0%, #4CAF50 100%); min-width: 60px; height: 50px; display: flex; align-items: center; justify-content: center; border-radius: 8px; font-weight: bold; font-size: 1.1em; color: #ffffff; box-shadow: 0 2px 6px rgba(0, 204, 255, 0.4);">
                    2-4
                </div>
                <span style="color: #ffffff; font-size: 0.9em; font-weight: 500;">Champions League</span>
            </div>
            <div style="display: flex; flex-direction: column; align-items: center; gap: 8px;">
                <div style="background: linear-gradient(135deg, #66ff66 0%, #66ff66 100%); min-width: 60px; height: 50px; display: flex; align-items: center; justify-content: center; border-radius: 8px; font-weight: bold; font-size: 1.1em; color: #2c3e50; box-shadow: 0 2px 6px rgba(102, 255, 102, 0.4);">
                    5
                </div>
                <span style="color: #ffffff; font-size: 0.9em; font-weight: 500;">Europa League</span>
            </div>
            <div style="display: flex; flex-direction: column; align-items: center; gap: 8px;">
                <div style="background: linear-gradient(135deg, #667eea 0%, #667eea 100%); min-width: 60px; height: 50px; display: flex; align-items: center; justify-content: center; border-radius: 8px; font-weight: bold; font-size: 1.1em; color: #ffffff; box-shadow: 0 2px 6px rgba(153, 153, 153, 0.4);">
                    6-17
                </div>
                <span style="color: #ffffff; font-size: 0.9em; font-weight: 500;">Other Positions</span>
            </div>
            <div style="display: flex; flex-direction: column; align-items: center; gap: 8px;">
                <div style="background: linear-gradient(135deg, #ff6666 0%, #ff8080 100%); min-width: 60px; height: 50px; display: flex; align-items: center; justify-content: center; border-radius: 8px; font-weight: bold; font-size: 1.1em; color: #ffffff; box-shadow: 0 2px 6px rgba(255, 102, 102, 0.4);">
                    18-20
                </div>
                <span style="color: #ffffff; font-size: 0.9em; font-weight: 500;">Relegation Zone</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

        
    # Additional insights
    st.markdown("<h2 style='text-align: center; color: white; margin-top: 40px;'> Key Insights</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        most_consistent = season_simulation.loc[season_simulation['std_points'].idxmin()]
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #3498db 0%, #2980b9 100%); color: white; text-align: center;">
            <h4> Most Consistent</h4>
            {get_team_logo(most_consistent['team'])}
            <div style="font-size: 1.3em; font-weight: bold; margin: 10px 0;">{most_consistent['team']}</div>
            <div>Std Dev: {most_consistent['std_points']:.2f} points</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        most_volatile = season_simulation.loc[season_simulation['std_points'].idxmax()]
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #e67e22 0%, #d35400 100%); color: white; text-align: center;">
            <h4> Most Unpredictable</h4>
            {get_team_logo(most_volatile['team'])}
            <div style="font-size: 1.3em; font-weight: bold; margin: 10px 0;">{most_volatile['team']}</div>
            <div>Std Dev: {most_volatile['std_points']:.2f} points</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        tight_race = season_simulation.iloc[3]['avg_points'] - season_simulation.iloc[0]['avg_points']
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #9b59b6 0%, #8e44ad 100%); color: white; text-align: center;">
            <h4>🏁 Title Race Margin</h4>
            <div style="font-size: 3em; margin: 10px 0;"></div>
            <div style="font-size: 1.5em; font-weight: bold;">{abs(tight_race):.1f}</div>
            <div>points gap (1st to 4th)</div>
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main application"""
    # Sidebar navigation
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h1 style="color: #667eea;"> KHEL-METRICS</h1>
        <p style="color: #666;">Premier League Analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    #  Responsive Navigation
    if st.session_state.get("is_mobile", False):
        with st.expander("☰ Menu"):
            page = st.radio(
                "Navigate",
                [" Match Prediction", " Season Simulation"],
                label_visibility="collapsed"
            )
    else:
        page = st.sidebar.radio(
            "Navigate",
            [" Match Prediction", " Season Simulation"],
            label_visibility="collapsed",
            key="sidebar_nav_radio"  
        )


    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="padding: 15px; background: rgba(102, 126, 234, 0.1); border-radius: 10px; margin: 10px 0;">
        <h4 style="color: #667eea;"> About</h4>
        <p style="font-size: 0.9em; color: #666;">
        Khel-Metrics uses advanced XGBoost models with Poisson regression 
        to predict Premier League matches and simulate entire seasons.
        Disclaimer
        KhelMetrics models are trained on football data available up to November 1, 2025. Predictions and insights
        are generated using statistical and machine learning methods, and should be interpreted as analytical estimates
        rather than guarantees. Past performance does not ensure future outcomes. 
        </p>
        <p style="font-size: 0.9em; color: #666;">
        <strong>Features:</strong><br>
        • 10,000 Monte Carlo simulations<br>
        • GPU-accelerated predictions<br>
        • Historical match analysis<br>
        • Real-time probability calculations
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8em;">
        <p>Powered by XGBoost & PyTorch</p>
        <p>© 2025 Khel-Metrics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Page routing
    if page == " Match Prediction":
        match_prediction_page()
    else:
        season_simulation_page()
    
    if st.session_state.is_mobile:
        with st.expander("☰ Menu"):
            page = st.radio("Navigate", [" Match Prediction", " Season Simulation"])
    else:
        page = st.sidebar.radio(
            "Navigate",
            [" Match Prediction", " Season Simulation"],
            label_visibility="collapsed"
        )


if __name__ == "__main__":
    main()