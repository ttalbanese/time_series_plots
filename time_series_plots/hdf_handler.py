import os
from dataclasses import dataclass
from typing import Dict, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from rashdf import RasGeomHdf, RasPlanHdf
from sklearn.neighbors import KNeighborsRegressor


@dataclass
class PlanData:
    file_source: str
    source: str
    mesh_cells: gpd.GeoDataFrame
    plan_params: dict
    plan_areas: list
    plan_name: str
    timesteps: np.ndarray


def reshape_coords(gdf) -> np.ndarray:
    return np.array(
        list(
            zip(
                gdf.centroid.y,
                gdf.centroid.x,
            )
        )
    )


def get_ras_info(source: str) -> PlanData:
    #return None
    with RasGeomHdf(source) as geom_hdf:
        with RasPlanHdf(source) as plan_hdf:
            mesh_cells = geom_hdf.mesh_cell_polygons()
            plan_params = plan_hdf.get_plan_param_attrs()
            plan_info = plan_hdf.get_plan_info_attrs()
            plan_areas = plan_params["2D Names"]
            plan_name = plan_info["Plan Name"]
            timesteps = plan_hdf[
                "/Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/Time Date Stamp"
            ][:]

            if isinstance(plan_areas, str):
                plan_areas = [plan_areas]

    return PlanData(
        file_source=source,
        source=source.split("\\")[-1],
        mesh_cells=mesh_cells,
        plan_params=plan_params,
        plan_areas=plan_areas,
        timesteps=timesteps,
        plan_name=plan_name,
    )


# Get computational cells within 10ft of the center profile line
def get_profile_cells(plan_data: PlanData) -> gpd.GeoDataFrame:

    profile = r"data\shapefiles\ProfileLine_selec_pl.shp"
    # Load canal profile polyline
    loaded_profile = gpd.read_file(profile)
    loaded_profile = loaded_profile.to_crs(plan_data.mesh_cells.crs)

    # Change canal profile polyline to 20ft wide polygon
    # assumes we are working in a crs with ft as the unit
    buffer = loaded_profile.buffer(10)
    profile_buffer = loaded_profile.set_geometry(buffer)
    return plan_data.mesh_cells.sjoin(profile_buffer)


def get_prediction_points_and_mileposts(
    plan_data: PlanData,
) -> Tuple[gpd.GeoDataFrame, np.ndarray]:
    transect_points = r"data\shapefiles\centerline_transects.shp"
    loaded_transects = gpd.read_file(transect_points).to_crs(plan_data.mesh_cells.crs).sort_values(by='Milepost').reset_index()

    loaded_transects = loaded_transects.loc[loaded_transects.index % 100 == 0]
    return reshape_coords(loaded_transects), loaded_transects["Milepost"].to_numpy()


def get_coords_by_plan(
    profile_cells: gpd.GeoDataFrame, plan_data: PlanData
) -> Dict[str, np.ndarray]:
    return {
        plan: reshape_coords(profile_cells.loc[profile_cells["mesh_name"] == plan])
        for plan in plan_data.plan_areas
    }


def predict_points(
    plan_data: PlanData,
    profile_cells: gpd.GeoDataFrame,
    coords_dict: Dict[str, gpd.GeoDataFrame],
    points_to_predict: np.ndarray,
    mileposts: np.ndarray,
) -> pd.DataFrame:
    results = []
    results_timesteps = []
    with RasPlanHdf(plan_data.file_source) as plan_hdf:
        for ix, ts in enumerate(plan_data.timesteps):
            x_list = []
            y_list = []
            for plan in plan_data.plan_areas:
                working = profile_cells.loc[profile_cells["mesh_name"] == plan]
                working_cell_ids = working["cell_id"]
                
                cells_of_interest = plan_hdf[
                    f"/Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/{plan}/Water Surface"
                ][ix, working_cell_ids]
                y_list.append(cells_of_interest)

                # get coordinates
                x_list.append(coords_dict[plan])
            X = np.concatenate(x_list)
            y = np.concatenate(y_list)
            results_timesteps.append([ts.decode("UTF-8")] * points_to_predict.shape[0])
            model = KNeighborsRegressor(weights="distance").fit(X, y)
            results.append(model.predict(points_to_predict))

    results_mileposts = [mileposts] * plan_data.timesteps.shape[0]

    return pd.DataFrame(
        {
            "WSE": np.concatenate(results),
            "Timestep": np.concatenate(results_timesteps),
            "Milepost": np.concatenate(results_mileposts),
            "Plan Name": plan_data.plan_name,
            "Source": plan_data.source,
        }
    ).sort_values(by=["Milepost", "Timestep"])


def interpolate_profile(plan_data: PlanData) -> pd.DataFrame:

    profile_cells = get_profile_cells(plan_data)
    points_to_predict, mileposts = get_prediction_points_and_mileposts(plan_data)

    coords_dict = get_coords_by_plan(profile_cells, plan_data)

    return predict_points(
        plan_data, profile_cells, coords_dict, points_to_predict, mileposts
    )


def save_profile(plan_data: PlanData, profile: pd.DataFrame) -> None:
    plan_name = plan_data.plan_name
    src_name = plan_data.source.replace(".hdf", "")
    #save_loc = os.path.join(r"data\csv", f"{src_name}_{plan_name}_profile.csv")

    #profile.to_csv(save_loc, index=False)

    profile.to_parquet(os.path.join(r"data\parquet", f"{src_name}_{plan_name}_profile.parquet"), index=False)


def handle_plan(source: str) -> None:
    plan_data = get_ras_info(source)
    profile = interpolate_profile(plan_data)
    save_profile(plan_data, profile)


if __name__ == "__main__":
    handle_plan(
        os.path.join(
            r"\\00-2810-007\sim\RAS\20240702_17_MileP_RoG_Croc", "17_MileP_RoG.p02.hdf"
        )
    )
