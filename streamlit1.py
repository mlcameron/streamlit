import io
import numpy as np
import pandas as pd
import polars as pl
import requests
import streamlit as st
import altair as alt

MATCH_IDS = ["1886347"]
PITCH_WIDTH = 600
PITCH_HEIGHT = int(PITCH_WIDTH * 60 / 100)


@st.cache_data(show_spinner=False)
def load_possessions(match_id: str) -> pl.DataFrame:
    """Download match events and keep only non-constant player possessions."""
    url = f"https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches/{match_id}/{match_id}_dynamic_events.csv"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    df = pl.read_csv(io.BytesIO(resp.content))
    df_pp = df.filter(pl.col("event_type") == "player_possession")
    const_mask = df_pp.select(((pl.all().max() == pl.all().min()).fill_null(True))).row(0)
    cols_to_drop = [c for c, is_const in zip(df_pp.columns, const_mask) if is_const]
    return df_pp.drop(cols_to_drop)


def prep_player_frame(df_pp: pl.DataFrame, player_name: str):
    """Filter for a player and compute derived coordinates for plotting."""
    df = df_pp.filter(pl.col("player_name") == player_name).to_pandas()
    df["x"] = pd.to_numeric(df["x_start"], errors="coerce")
    df["y"] = pd.to_numeric(df["y_start"], errors="coerce")
    for col in [
        "x",
        "y",
        "x_end",
        "y_end",
        "pass_angle_received",
        "pass_distance_received",
        "pass_angle",
        "pass_distance",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["pass_angle_received_rad"] = np.deg2rad(df["pass_angle_received"])
    df["pass_angle_rad"] = np.deg2rad(df["pass_angle"])

    df["x_recv"] = df["x"] + df["pass_distance_received"] * np.cos(df["pass_angle_received_rad"])
    df["y_recv"] = df["y"] + df["pass_distance_received"] * np.sin(df["pass_angle_received_rad"])
    df["x_pass"] = df["x_end"] + df["pass_distance"] * np.cos(df["pass_angle_rad"])
    df["y_pass"] = df["y_end"] + df["pass_distance"] * np.sin(df["pass_angle_rad"])

    id_cols = ["x", "y"]
    numeric_extra_cols = df.drop(columns=id_cols).select_dtypes(include="number").columns.tolist()
    df_long = df.melt(id_vars=id_cols, value_vars=numeric_extra_cols, var_name="metric", value_name="value")
    return df, df_long


st.title("Interactive Football Pitch + Detail View")

selected_match = st.selectbox("Select Match", sorted(MATCH_IDS))
df_pp = load_possessions(selected_match)
player_names = sorted(df_pp["player_name"].unique())
selected_player = st.selectbox("Select Player", player_names)

df, df_long = prep_player_frame(df_pp, selected_player)

st.caption(f"Click a point on the pitch to explore metrics. Match id: {selected_match}")

# --- Reshape the data for the bar chart ---
# We want one row per (x, y, metric) combination
df["x"] = pd.to_numeric(df["x_start"], errors="coerce")
df["y"] = pd.to_numeric(df["y_start"], errors="coerce")
for col in ["x", "y", "x_end", "y_end",
	"pass_angle_received", "pass_distance_received",
	"pass_angle", "pass_distance"]:
	df[col] = pd.to_numeric(df[col], errors="coerce")

df["pass_angle_received_rad"] = np.deg2rad(df["pass_angle_received"])
df["pass_angle_rad"] = np.deg2rad(df["pass_angle"])

# Endpoint of the received vector (from start point)
df["x_recv"] = df["x"] + df["pass_distance_received"] * np.cos(df["pass_angle_received_rad"])
df["y_recv"] = df["y"] + df["pass_distance_received"] * np.sin(df["pass_angle_received_rad"])

# Endpoint of the pass vector (from end point)
df["x_pass"] = df["x_end"] + df["pass_distance"] * np.cos(df["pass_angle_rad"])
df["y_pass"] = df["y_end"] + df["pass_distance"] * np.sin(df["pass_angle_rad"])

id_cols = ["x", "y"]  # plus anything that identifies the event/point
numeric_extra_cols = (
	df.drop(columns=id_cols)
	.select_dtypes(include="number")
	.columns.tolist()
)

df_long = df.melt(
	id_vars=id_cols,
	value_vars=numeric_extra_cols,
	var_name="metric",
	value_name="value",
)

# --- Define a selection based on x and y (your coordinates) ---
point_sel = alt.selection_point(fields=["x", "y"], nearest=True, empty="none")

scatter = (
	alt.Chart(df)
	.mark_circle(size=80)
	.encode(
		x=alt.X("x:Q", scale=alt.Scale(domain=[-52.5, 52.5]), title=""),
		y=alt.Y("y:Q", scale=alt.Scale(domain=[-34, 34]), title=""),
		tooltip=[
			"player_name",
			#"team_name",
			#"period",
			#"timestamp",
			#"event_type",
			"x",
			"y",
			"x_end",
			"y_end",
			"pass_outcome",
		],
	)
	.properties(width=PITCH_WIDTH, height=PITCH_HEIGHT)
	.add_params(point_sel)
)

highlight_scatter = (
	alt.Chart(df)
	.mark_circle(size=200)
	.encode(
		x="x:Q",
		y="y:Q",
		color=alt.value("red"),  # highlight color
	)
	.transform_filter(point_sel)  # only the selected point
)

end_point = (
	alt.Chart(df)
	.mark_circle(
		size=200,
		filled=False,         # no fill
		stroke="yellow",      # outline color
		strokeWidth=3,
	)
	.encode(
		x=alt.X("x_end:Q", scale=alt.Scale(domain=[-52.5, 52.5])),
		y=alt.Y("y_end:Q", scale=alt.Scale(domain=[-34, 34])),
	)
	.transform_filter(point_sel)  # same selected row → uses x_end/y_end
)

pass_line = (
	alt.Chart(df)
	.mark_rule(
		color="yellow",
		strokeWidth=1,
		#strokeDash=[4, 4],  # dashed / dotted effect
	)
	.encode(
		x="x:Q",
		x2="x_end:Q",
		y="y:Q",
		y2="y_end:Q",
	)
	.transform_filter(point_sel)
)

# --- Line for *received* angle/distance, from (x, y) to (x_recv, y_recv) ---
recv_vec = (
	alt.Chart(df)
	.mark_rule(
		color="cyan",
		strokeWidth=2,
		strokeDash=[4, 4],  # dotted/dashed
	)
	.encode(
		x="x:Q",
		x2="x_recv:Q",
		y="y:Q",
		y2="y_recv:Q",
	)
	.transform_filter(point_sel)
)

# --- Line for *pass* angle/distance, from (x_end, y_end) to (x_pass, y_pass) ---
successful_vec = (
	alt.Chart(df)
	.mark_rule(
		strokeWidth=2,
		strokeDash=[4, 4], 
	)  # solid by default
	.encode(
		x="x_end:Q",
		x2="x_pass:Q",
		y="y_end:Q",
		y2="y_pass:Q",
		color=alt.value("green"),
	)
	.transform_filter(point_sel)
	.transform_filter("datum.pass_outcome == 'successful'")
	.transform_filter("datum.x_pass != null && datum.y_pass != null")
)

unsuccessful_vec = (
	alt.Chart(df)
	.mark_rule(
		strokeWidth=2,
		strokeDash=[4, 4],  # dotted/dashed
	)
	.encode(
		x="x_end:Q",
		x2="player_targeted_x_pass:Q",
		y="y_end:Q",
		y2="player_targeted_y_pass:Q",
		color=alt.value("red"),
	)
	.transform_filter(point_sel)
	.transform_filter("datum.pass_outcome == 'unsuccessful'")
	.transform_filter(
		"datum.player_targeted_x_pass != null && datum.player_targeted_y_pass != null"
	)
)

# --- Bar chart showing attributes for the selected point ---
bars = (
	alt.Chart(df_long)
	.mark_bar()
	.encode(
		x=alt.X("metric:N", title="Metric"),
		y=alt.Y("value:Q", title="Value"),
		tooltip=["metric", "value"],
	)
	.transform_filter(point_sel)  # only show selected point's metrics
	.properties(width=300, height=300)
)

lines = []

# Outer pitch rectangle
rect_df = pd.DataFrame([
	{"x1": -52.5,   "y1": -34,  "x2": 52.5, "y2": -34},   # bottom
	{"x1": -52.5,   "y1": 34, "x2": 52.5, "y2": 34},  # top
	{"x1": -52.5,   "y1": -34,  "x2": -52.5,   "y2": 34},  # left
	{"x1": 52.5, "y1": -34,  "x2": 52.5, "y2": 34},  # right
])

outer = (
	alt.Chart(rect_df)
	.mark_rule(color="white", strokeWidth=3, clip=False)
	.encode(
		x="x1:Q",
		x2="x2:Q",
		y="y1:Q",
		y2="y2:Q",
	)
	.properties(width=600, height=360)
)

# Halfway line
halfway = alt.Chart(pd.DataFrame({'x': [0, 0], 'y': [-34, 34]})).mark_line(
	color='white', strokeWidth=1
).encode(x='x:Q', y='y:Q')

# Penalty boxes, etc. (example: left penalty area)
rect_df2 = pd.DataFrame([
	{"x1": -52.5,   "y1": -21,  "x2": -36, "y2": -21},   # bottom
	{"x1": -52.5,   "y1": 21, "x2": -36, "y2": 21},  # top
	{"x1": -36, "y1": -21,  "x2": -36, "y2": 21},  # right
])
pen_area_left = (
	alt.Chart(rect_df2)
	.mark_rule(color="white", strokeWidth=1, clip=False)
	.encode(
		x="x1:Q",
		x2="x2:Q",
		y="y1:Q",
		y2="y2:Q",
	)
	.properties(width=600, height=360)
)

rect_df3 = pd.DataFrame([
	{"x1": 36,   "y1": -21,  "x2": 52.5, "y2": -21},   # bottom
	{"x1": 36,   "y1": 21, "x2": 52.5, "y2": 21},  # top
	{"x1": 36, "y1": -21,  "x2": 36, "y2": 21},  # left
])
pen_area_right = (
	alt.Chart(rect_df3)
	.mark_rule(color="white", strokeWidth=1, clip=False)
	.encode(
		x="x1:Q",
		x2="x2:Q",
		y="y1:Q",
		y2="y2:Q",
	)
	.properties(width=600, height=360)
)

pitch = alt.layer(
	outer,
	halfway,
	pen_area_left,
	pen_area_right,
	scatter,
	highlight_scatter,
	end_point,
	pass_line,
	recv_vec,
	successful_vec,
	unsuccessful_vec
).properties(width=600, height=360)

combined = pitch | bars
st.altair_chart(combined, use_container_width=True)
