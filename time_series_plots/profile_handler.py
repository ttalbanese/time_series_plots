import polars as pl
from dataclasses import dataclass
from typing import List, Optional, Dict
import os


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
        return os.path.join(r"data\to_plot", f"{self.shortname}.parquet")


@dataclass
class CanalConstants:
    e33_mp: float = 256.41288436
    e32_mp: float = 255.09186121


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
            name="50% of Probable Maximum Flood",
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
            name="50% of Probable Maximum Flood (LLO half open)",
            shortname="RoG_PMP_Center_v2_0.5xPMF_LLOHalfOpen",
            sources=[
                r"data\parquet\17_MileP_RoG.p28_RoG_PMP_Center_v2_0.5xPMF_LLOHalfOpen_profile.parquet"
            ],
            cutoff=True,
            storm_name=["50_sqmi_0.5PMF"],
        ),
        ProfileData(
            name="50% of Probable Maximum Flood (LLO fully open)",
            shortname="RoG_PMP_Center_v2_0.5xPMF_LLOFullOpen",
            sources=[
                r"data\parquet\17_MileP_RoG.p25_RoG_PMP_Center_v2_0.5xPMF_LLOFullOpen_profile.parquet"
            ],
            cutoff=True,
            storm_name=["50_sqmi_0.5PMF"],
        ),
        ProfileData(
            name="50% of Probable Maximum Flood (Pre-Drained)",
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


def get_profile_names() -> List[str]:
    return [p.name for p in get_profiles()]


def load_profile_data(profile: ProfileData) -> pl.DataFrame:
    if profile.combined_profile:
        profile_data = load_combined_profile(profile)
    else:
        profile_data = pl.read_parquet(profile.sources).select(
            pl.exclude(["Source", "Plan Name"])
        )

    profile_data = fix_time_and_sort(profile_data)
    
    if profile.cutoff:
        profile_data = cutoff_profile(profile_data)

    #return smooth_profile(profile_data)
    return profile_data


def smooth_profile(profile_data: pl.DataFrame) -> pl.DataFrame:
    # We could also smooth over time if we wanted. Woah!

    return profile_data.with_columns(pl.col("WSE").ewm_mean(span=5).over("Timestep"))


def cutoff_profile(profile_data: pl.DataFrame) -> pl.DataFrame:
    cutoff = CanalConstants().e32_mp
    return profile_data.filter(pl.col("Milepost") < cutoff)


def fix_time_and_sort(profile_data: pl.DataFrame) -> pl.DataFrame:
    return profile_data.with_columns(
        pl.col("Timestep").str.to_datetime(format="%d%b%Y %H:%M:%S")
    ).sort(by=["Milepost", "Timestep"])


def load_combined_profile(profile: ProfileData) -> pl.DataFrame:
    constants = CanalConstants()
    loaded_data = (
        pl.read_parquet(profile.sources)
        .select(pl.exclude("Source"))
        .pivot(on="Plan Name", values="WSE")
    )

    columns = loaded_data.columns

    e33_col = [c for c in columns if "33" in c][0]
    e32_col = [c for c in columns if "32" in c][0]
    center_col = [c for c in columns if "enter" in c][0]

    return loaded_data.with_columns(
        pl.when(pl.col("Milepost") > constants.e33_mp)
        .then(pl.col(e33_col))
        .when(pl.col("Milepost") > constants.e32_mp)
        .then(pl.col(e32_col))
        .otherwise(pl.col(center_col))
        .alias("WSE")
    ).select(["WSE", "Timestep", "Milepost"])


def handle_profile(profile: ProfileData) -> None:
    profile_data = load_profile_data(profile)
    profile_data = profile_data.with_columns(pl.lit(profile.name).alias("Profile Name"))

    profile_data.write_parquet(profile.save_loc)
