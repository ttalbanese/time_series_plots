import os
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
import polars as pl
from dataclasses import dataclass
from typing import List, Optional, Dict


@dataclass
class ProfileData:
    name: str
    shortname: str
    sources: List[str]
    cutoff: bool
    storm_name: List[str]
    storm_factor: Optional[float] = 1.0

    @property
    def combined_profile(self):
        return len(self.sources) > 1

    @property
    def save_loc(self):
        return os.path.join(r"static\canal_profile_plots", f"{self.shortname}.parquet")


def get_profiles() -> List[ProfileData]:
    return [
        ProfileData(
            name="Probable Maximum Flood Event",
            shortname="PMF",
            sources=[
                r"data\parquet\17_MileP_RoG.p59_RoG_PMP_E33_v2_profile.parquet",
                r"data\parquet\17_MileP_RoG.p29_RoG_PMP_E32_profile.parquet",
                r"data\parquet\17_MileP_RoG.p50_RoG_PMP_Center_v2_profile.parquet",
            ],
            cutoff=False,
            storm_name=["10_sqmi_PMF", "50_sqmi_PMF"],
        ),
        ProfileData(
            name="Half of Probable Maximum Flood",
            shortname="0.5xPMF",
            sources=[
                r"data\parquet\17_MileP_RoG.p12_RoG_PMP_E33_v2_0.5xPMF_profile.parquet",
                r"data\parquet\17_MileP_RoG.p02_RoG_PMP_E32_0.5xPMF_profile.parquet",
                r"data\parquet\17_MileP_RoG.p17_RoG_PMP_Center_v2_0.5xPMF_profile.parquet",
            ],
            cutoff=False,
            storm_name=["10_sqmi_0.5PMF", "50_sqmi_0.5PMF"],
        ),
        ProfileData(
            name="Half of Probable Maximum Flood (LLO half open)",
            shortname="RoG_PMP_Center_v2_0.5xPMF_LLOHalfOpen",
            sources=[
                r"data\parquet\17_MileP_RoG.p28_RoG_PMP_Center_v2_0.5xPMF_LLOHalfOpen_profile.parquet"
            ],
            cutoff=True,
            storm_name=["50_sqmi_0.5PMF"],
        ),
        ProfileData(
            name="Half of Probable Maximum Flood (LLO fully open)",
            shortname="RoG_PMP_Center_v2_0.5xPMF_LLOFullOpen",
            sources=[
                r"data\parquet\17_MileP_RoG.p25_RoG_PMP_Center_v2_0.5xPMF_LLOFullOpen_profile.parquet"
            ],
            cutoff=True,
            storm_name=["50_sqmi_0.5PMF"],
        ),
        ProfileData(
            name="Half of Probable Maximum Flood (Pre-Drained)",
            shortname="RoG_PMP_Center_v2_0.5xPMF_Empty",
            sources=[
                r"data\parquet\17_MileP_RoG.p23_RoG_PMP_Center_v2_0.5xPMF_Empty_profile.parquet"
            ],
            cutoff=True,
            storm_name=["50_sqmi_0.5PMF"],
        ),
        ProfileData(
            name="500-Years Rainfall Event (AEP 0.002)",
            shortname="RoG_500yrs",
            sources=[r"data\parquet\17_MileP_RoG.p20_RoG_500yrs_profile.parquet"],
            cutoff=False,
            storm_name=["500"],
        ),
        ProfileData(
            name="500-Years Rainfall Event (LLO fully open)",
            shortname="RoG_500yrs_LLOFullOpen",
            sources=[
                r"data\parquet\17_MileP_RoG.p38_RoG_500yrs_LLOFullOpen_profile.parquet"
            ],
            cutoff=False,
            storm_name=["500"],
        ),
        ProfileData(
            name="150% of 100-Years Rainfall Event Flood",
            shortname="RoG_150%of100yrs",
            sources=[r"data\parquet\17_MileP_RoG.p19_RoG_150%of100yrs_profile.parquet"],
            cutoff=False,
            storm_name=["150"],
        ),
        ProfileData(
            name="150% of 100-Years Rainfall Event (LLO fully open)",
            shortname="RoG_150%of100yrs_LLOFullOpen",
            sources=[
                r"data\parquet\17_MileP_RoG.p37_RoG_150%of100yrs_LLOFullOpen_profile.parquet"
            ],
            cutoff=False,
            storm_name=["150"],
        ),
        ProfileData(
            name="100-Years Rainfall Event (AEP 0.01)",
            shortname="RoG_100yrs",
            sources=[r"data\parquet\17_MileP_RoG.p18_RoG_100yrs_profile.parquet"],
            cutoff=False,
            storm_name=["100"],
        ),
        ProfileData(
            name="100-Years Rainfall Event (LLO fully open)",
            shortname="RoG_100yrs_LLOFullOpen",
            sources=[
                r"data\parquet\17_MileP_RoG.p26_RoG_100yrs_LLOFullOpen_profile.parquet"
            ],
            cutoff=False,
            storm_name=["100"],
        ),
        ProfileData(
            name="100-Years Rainfall Event (LLO half open)",
            shortname="RoG_100yrs_LLOHalfOpen",
            sources=[
                r"data\parquet\17_MileP_RoG.p27_RoG_100yrs_LLOHalfOpen_profile.parquet"
            ],
            cutoff=False,
            storm_name=["100"],
        ),
        ProfileData(
            name="100-Years Rainfall Event (Pre-Drained)",
            shortname="RoG_100yrs_Empty",
            sources=[r"data\parquet\17_MileP_RoG.p24_RoG_100yrs_Empty_profile.parquet"],
            cutoff=False,
            storm_name=["100"],
        ),
        ProfileData(
            name="50-Years Rainfall Event (AEP 0.02)",
            shortname="RoG_50yrs",
            sources=[r"data\parquet\17_MileP_RoG.p21_RoG_50yrs_profile.parquet"],
            cutoff=False,
            storm_name=["50"],
        ),
        ProfileData(
            name="60% of Probable Maximum Flood",
            shortname="RoG_PMP_Center_v2_0.6xPMF",
            sources=[
                r"data\parquet\17_MileP_RoG.p65_RoG_PMP_Center_v2_0.6xPMF_profile.parquet"
            ],
            cutoff=True,
            storm_name=["50_sqmi_PMF"],
            storm_factor=0.704,
        ),
        ProfileData(
            name="70% of Probable Maximum Flood",
            shortname="RoG_PMP_Center_v2_0.7xPMF",
            sources=[
                r"data\parquet\17_MileP_RoG.p67_RoG_PMP_Center_v2_0.7xPMF_profile.parquet"
            ],
            cutoff=True,
            storm_name=["50_sqmi_PMF"],
            storm_factor=0.778,
        ),
        ProfileData(
            name="80% of Probable Maximum Flood",
            shortname="RoG_PMP_Center_v2_0.8xPMF",
            sources=[
                r"data\parquet\17_MileP_RoG.p68_RoG_PMP_Center_v2_0.8xPMF_profile.parquet"
            ],
            cutoff=True,
            storm_name=["50_sqmi_PMF"],
            storm_factor=0.852,
        ),
        ProfileData(
            name="90% of Probable Maximum Flood",
            shortname="RoG_PMP_Center_v2_0.9xPMF",
            sources=[
                r"data\parquet\17_MileP_RoG.p69_RoG_PMP_Center_v2_0.9xPMF_profile.parquet"
            ],
            cutoff=True,
            storm_name=["50_sqmi_PMF"],
            storm_factor=0.926,
        ),
    ]


def get_profile_dict() -> Dict[str, ProfileData]:
    return {p.name: p for p in get_profiles()}


def get_profile_list():
    if "profiles" not in st.session_state:
        profiles_dict = get_profile_dict()
        st.session_state["profiles"] = list(profiles_dict.values())
        st.session_state["profiles_dict"] = profiles_dict
        st.session_state["profile_names"] = list(profiles_dict.keys())


def load_profiles():
    if "profiles_loaded" not in st.session_state:
        profiles_list = st.session_state["profiles"]
        df = pl.read_parquet([p.save_loc for p in profiles_list])

        common_timesteps = (
            df.pivot("Profile Name", values="WSE")
            .drop_nulls()
            .select(pl.col("Timestep"))
            .unique()
        )

        st.session_state["profiles_loaded"] = df.filter(
            pl.col("Timestep").is_in(common_timesteps)
        )

        st.session_state["common_timesteps"] = common_timesteps


def get_mileposts():
    if "mileposts" not in st.session_state:
        st.session_state["mileposts"] = st.session_state["profiles_loaded"].select(
            pl.col("Milepost").unique().sort().gather_every(5)
        )

        st.session_state["min_mile"] = st.session_state["mileposts"].min().item()
        st.session_state["max_mile"] = st.session_state["mileposts"].max().item()


def load_precip():
    if "precip_data" not in st.session_state:
        precip_data_loc = r"data\to_plot\precip_data.csv"
        precip_data = (
            pl.read_csv(
                precip_data_loc,
                schema_overrides={
                    "Timestep": pl.String,
                    "Rate": pl.Float32,
                    "Storm": pl.String,
                    "Unit": pl.String,
                },
            )
            .with_columns(
                pl.col("Timestep").str.to_datetime("%d %b %Y, %H:%M"),
            )
            .sort("Timestep")
            .with_columns(
                pl.col("Rate").shift(1).cum_sum().over("Storm").alias("Accumulation"),
            )
        )
        partitions = precip_data.partition_by("Storm", include_key=True, as_dict=True)

        for k, v in partitions.items():
            partitions[k] = v.upsample("Timestep", every="10m").with_columns(
                pl.col(["Rate", "Storm", "Unit"]).forward_fill(),
                pl.col("Accumulation").interpolate().fill_null(0),
            )
        st.session_state["precip_data"] = precip_data
        st.session_state["precip_partitions"] = partitions


def make_overall_selector():

    with st.expander("Canal Profile Selector"):
        st.multiselect(
            "Profiles to Plot",
            options=st.session_state["profile_names"],
            default=st.session_state["profile_names"],
            key="overall_options",
            label_visibility="collapsed",
        )


def make_overall_plot():

    options = st.session_state["overall_options"]
    mileposts = st.session_state["mileposts"]

    to_plot = st.session_state["profiles_loaded"].filter(
        (pl.col("Profile Name").is_in(options)) & (pl.col("Milepost").is_in(mileposts))
    )

    fig_2d = px.line(
        to_plot,
        x="Milepost",
        y="WSE",
        animation_frame="Timestep",
        color="Profile Name",
        width=1600,
        height=1000,
    ).update_xaxes(autorange="reversed")
    st.plotly_chart(fig_2d)


def make_detail_select():
    with st.expander("Canal Profile Selector"):
        st.selectbox(
            "Profile to Plot",
            options=st.session_state["profile_names"],
            index=0,
            key="detail_select",
            label_visibility="collapsed",
        )


def make_milepost_slider():
    # TODO: Switch to number input, have min and max be bound for selected plot
    min_mile = st.session_state["min_mile"]
    max_mile = st.session_state["max_mile"]

    mid_point = (min_mile + max_mile) / 2
    st.slider(
        "Milepost",
        min_value=min_mile,
        max_value=max_mile,
        value=mid_point,
        step=0.1,
        key="mile_slider",
    )


def make_detail_plot():

    fig = make_subplots(
        rows=4,
        cols=3,
        specs=[
            [{"colspan": 2, "rowspan": 3}, {}, {}],
            [{}, {}, {}],
            [{}, {}, {}],
            [{"colspan": 2}, {}, {}],
        ],
    )
    selected_profile_name = st.session_state["detail_select"]
    if selected_profile_name is None:
        st.write("Please select a profile")
        return None

    selected_profile = st.session_state["profiles_dict"][selected_profile_name]

    selected_storms = selected_profile.storm_name
    storm_factor = selected_profile.storm_factor

    selected_milepost = st.session_state["mile_slider"]
    available_mileposts = st.session_state["mileposts"]

    closest_milepost = (
        available_mileposts.with_columns(
            pl.col("Milepost").get(
                (pl.col("Milepost") - selected_milepost).abs().arg_min()
            )
        )
        .select(pl.first("Milepost"))
        .item()
    )

    # TODO: support multiple storms
    # TODO: 4 rows, 3 columns
    storm_name = selected_storms[0]

    storm_plots = []
    for ix, storm in enumerate(selected_storms):
        rates = (
            st.session_state["precip_data"]
            .filter(pl.col("Storm") == storm)
            .with_columns(pl.col("Rate") * storm_factor)
        )
        static_bar = px.bar(rates, x="Timestep", y="Rate", title=storm).update_layout(
            bargap=0
        )
        axis_number = 6 + (ix * 3)
        static_bar.data[0].xaxis = f"x{axis_number}"
        static_bar.data[0].yaxis = f"y{axis_number}"
        storm_plots.append(static_bar)

    profile = st.session_state["profiles_loaded"].filter(
        pl.col("Profile Name").eq(selected_profile_name)
    )

    rates = st.session_state["precip_data"].filter(pl.col("Storm") == storm_name)
    stage_df = profile.filter(pl.col("Milepost").eq(closest_milepost))

    static_line = px.line(stage_df, x="Timestep", y="WSE")
    static_line.data[0].xaxis = "x3"
    static_line.data[0].yaxis = "y3"

    hyeto_dot = px.scatter(
        stage_df,
        x="Timestep",
        y="WSE",
        animation_frame="Timestep",
    )
    profile_dot = px.scatter(
        stage_df,
        x="Milepost",
        y="WSE",
        animation_frame="Timestep",
    )

    profile_plot = px.line(
        profile,
        x="Milepost",
        y="WSE",
        animation_frame="Timestep",
        # color="Profile Name",
    ).update_xaxes(autorange="reversed")

    frames = []
    for ts in (
        st.session_state["common_timesteps"]
        .unique()
        .sort("Timestep")
        .to_series()
        .to_list()
    ):
        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
        profile_dot_frame = (
            go.Scatter(
                {
                    "hovertemplate": "Milepost=%{x}<br>WSE=%{y}<extra></extra>",
                    "legendgroup": "",
                    "marker": {"color": "#636efa", "symbol": "circle"},
                    "mode": "markers",
                    "name": "",
                    "orientation": "v",
                    "showlegend": False,
                    "type": "scatter",
                    "x": np.array([closest_milepost], dtype=object),
                    "xaxis": "x",
                    "y": np.array(
                        [
                            stage_df.filter(pl.col("Timestep") == ts)
                            .select("WSE")
                            .to_series()
                            .first()
                        ]
                    ),
                    "yaxis": "y",
                }
            ),
        )
        hyeto_dot_frame = (
            go.Scatter(
                {
                    "hovertemplate": "Timestep=%{x}<br>WSE=%{y}<extra></extra>",
                    "legendgroup": "",
                    "marker": {"color": "#636efa", "symbol": "circle"},
                    "mode": "markers",
                    "name": "",
                    "orientation": "v",
                    "showlegend": False,
                    "type": "scatter",
                    "x": np.array([ts], dtype=object),
                    "xaxis": "x3",
                    "y": np.array(
                        [
                            stage_df.filter(pl.col("Timestep") == ts)
                            .select("WSE")
                            .to_series()
                            .first()
                        ]
                    ),
                    "yaxis": "y3",
                }
            ),
        )

        profile_filtered = profile.filter(pl.col("Timestep") == ts)
        profile_frame = (
            go.Scatter(
                {
                    "hovertemplate": f"Timestep={ts_str}<br>Milepost=%{{x}}<br>WSE=%{{y}}<extra></extra>",
                    "legendgroup": "",
                    "line": {"color": "#3E4AEE", "dash": "solid"},
                    "marker": {"symbol": "circle"},
                    "mode": "lines",
                    "name": "",
                    "orientation": "v",
                    "showlegend": False,
                    "type": "scatter",
                    "x": profile_filtered.select("Milepost").to_series().to_numpy(),
                    "xaxis": "x",
                    "y": profile_filtered.select("WSE").to_series().to_numpy(),
                    "yaxis": "y",
                }
            ),
        )

        new_data = (
            profile_frame + profile_dot_frame + hyeto_dot_frame + static_line.data
        )
        for sp in storm_plots:
            new_data = new_data + sp.data
        new_frame = go.Frame(
            data=new_data,
            name=ts_str,
        )

        frames.append(new_frame)

    fig.add_trace(profile_plot.data[0], row=1, col=1)
    fig.add_trace(profile_dot.data[0], row=1, col=1)

    fig.add_trace(static_line.data[0], row=1, col=3)
    fig.add_trace(hyeto_dot.data[0], row=1, col=3)
    for ix, sp in enumerate(storm_plots):
        row = 2 + ix
        fig.add_trace(sp.data[0], row=row, col=3)
    fig["layout"]["xaxis"]["autorange"] = "reversed"
    fig["layout"]["yaxis3"]["title"] = "WSE"

    fig.update(frames=frames)
    fig.update_layout(
        sliders=profile_plot.layout["sliders"],
        width=1200,
        height=1000,
    )

    st.plotly_chart(fig)


# TODO: storm precipitation factor


@st.fragment
def make_overall_fragment():
    make_overall_selector()
    with st.spinner("Generating Overall Plot"):
        make_overall_plot()


@st.fragment
def make_detail_fragment():
    make_detail_select()
    make_milepost_slider()
    with st.spinner("Generating Detail Plot"):
        make_detail_plot()


def app():
    st.set_page_config(layout="wide")
    with st.spinner("Loading initial data"):
        get_profile_list()
        load_profiles()
        load_precip()
        get_mileposts()

    overall_tab, detail_tab = st.tabs(["Overall", "Detail"])
    with overall_tab:
        make_overall_fragment()

    with detail_tab:
        make_detail_fragment()


if __name__ == "__main__":
    app()
