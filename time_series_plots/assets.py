import os
from dagster import asset, StaticPartitionsDefinition, AssetExecutionContext

from time_series_plots.hdf_handler import handle_plan
from time_series_plots.partion_defs import simulation_partitions
from time_series_plots.profile_handler import get_profile_dict, get_profile_names, handle_profile

@asset(
    partitions_def=StaticPartitionsDefinition(
        simulation_partitions
    )
)
def simulation_profiles(context: AssetExecutionContext):
    os.makedirs(r'data\parquet', exist_ok=True)
    src = context.partition_key
    plan_loc = os.path.join(r'\\00-2810-007\sim\RAS\20240702_17_MileP_RoG_Croc', src)
    handle_plan(plan_loc)


@asset(
    #deps = [simulation_profiles],
    partitions_def=StaticPartitionsDefinition(
        get_profile_names()
    )
)
def profiles_for_plotting(context: AssetExecutionContext):
    os.makedirs(r'data\to_plot', exist_ok=True)
    src = context.partition_key

    cur_profile = get_profile_dict()[src]

    handle_profile(cur_profile)