from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import xarray

from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.utils.config import Config


class japan(BaseDataset):
    """Data set class for the CAMELS US data set by [#]_ and [#]_.
    
    Parameters
    ----------
    cfg : Config
        The run configuration.
    is_train : bool 
        Defines if the dataset is used for training or evaluating. If True (training), means/stds for each feature
        are computed and stored to the run directory. If one-hot encoding is used, the mapping for the one-hot encoding 
        is created and also stored to disk. If False, a `scaler` input is expected and similarly the `id_to_int` input
        if one-hot encoding is used. 
    period : {'train', 'validation', 'test'}
        Defines the period for which the data will be loaded
    basin : str, optional
        If passed, the data for only this basin will be loaded. Otherwise the basin(s) are read from the appropriate
        basin file, corresponding to the `period`.
    additional_features : List[Dict[str, pd.DataFrame]], optional
        List of dictionaries, mapping from a basin id to a pandas DataFrame. This DataFrame will be added to the data
        loaded from the dataset and all columns are available as 'dynamic_inputs', 'evolving_attributes' and
        'target_variables'
    id_to_int : Dict[str, int], optional
        If the config argument 'use_basin_id_encoding' is True in the config and period is either 'validation' or 
        'test', this input is required. It is a dictionary, mapping from basin id to an integer (the one-hot encoding).
    scaler : Dict[str, Union[pd.Series, xarray.DataArray]], optional
        If period is either 'validation' or 'test', this input is required. It contains the centering and scaling
        for each feature and is stored to the run directory during training (train_data/train_data_scaler.yml).
        
    References
    ----------
    .. [#] A. J. Newman, M. P. Clark, K. Sampson, A. Wood, L. E. Hay, A. Bock, R. J. Viger, D. Blodgett, 
        L. Brekke, J. R. Arnold, T. Hopson, and Q. Duan: Development of a large-sample watershed-scale 
        hydrometeorological dataset for the contiguous USA: dataset characteristics and assessment of regional 
        variability in hydrologic model performance. Hydrol. Earth Syst. Sci., 19, 209-223, 
        doi:10.5194/hess-19-209-2015, 2015
    .. [#] Addor, N., Newman, A. J., Mizukami, N. and Clark, M. P.: The CAMELS data set: catchment attributes and 
        meteorology for large-sample studies, Hydrol. Earth Syst. Sci., 21, 5293-5313, doi:10.5194/hess-21-5293-2017,
        2017.
    """

    def __init__(self,
                 cfg: Config,
                 is_train: bool,
                 period: str,
                 basin: str = None,
                 additional_features: List[Dict[str, pd.DataFrame]] = [],
                 id_to_int: Dict[str, int] = {},
                 scaler: Dict[str, Union[pd.Series, xarray.DataArray]] = {}):
        super(japan, self).__init__(cfg=cfg,
                                       is_train=is_train,
                                       period=period,
                                       basin=basin,
                                       additional_features=additional_features,
                                       id_to_int=id_to_int,
                                       scaler=scaler)

    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        #print('kurobe data loaded')
        """Load input and output data from text files."""
        # get forcings
        dfs = []
        for forcing in self.cfg.forcings:
            df = load_camels_us_forcings(self.cfg.data_dir, basin, forcing)

            # rename columns
            if len(self.cfg.forcings) > 1:
                df = df.rename(columns={col: f"{col}_{forcing}" for col in df.columns})
            dfs.append(df)
        df = pd.concat(dfs, axis=1)

        # add discharge
        #df['QObs'] = load_camels_us_discharge(self.cfg.data_dir, basin)

        # replace invalid discharge values by NaNs
        qobs_cols = [col for col in df.columns if "qobs" in col.lower()]
        for col in qobs_cols:
            df.loc[df[col] < 0, col] = np.nan

        return df

    def _load_attributes(self) -> pd.DataFrame:
        return load_camels_us_attributes(self.cfg.data_dir, basins=self.basins)


def load_camels_us_attributes(data_dir: Path, basins: List[str] = []) -> pd.DataFrame:
    """Load Japan basin attributes from the unified attributes file

    Parameters
    ----------
    data_dir : Path
        Path to the Japan data directory. This folder must contain a 'camels_attributes_v2.0' folder 
        containing the unified attributes file.
    basins : List[str], optional
        If passed, return only attributes for the basins specified in this list. Otherwise, the attributes of all basins
        are returned.

    Returns
    -------
    pandas.DataFrame
        Basin-indexed DataFrame, containing the attributes as columns.
    """
    attributes_path = Path(data_dir) / 'camels_attributes_v2.0'

    if not attributes_path.exists():
        raise RuntimeError(f"Attribute folder not found at {attributes_path}")

    # 优先使用统一属性文件
    unified_file = attributes_path / 'camels_japan_attributes.txt'
    if unified_file.exists():
        df = pd.read_csv(unified_file, sep=';', header=0, dtype={'gauge_id': str})
        df = df.set_index('gauge_id')
    else:
        # 如果没有统一文件，尝试读取分类文件
        txt_files = list(attributes_path.glob('camels_japan_*.txt'))
        if not txt_files:
            raise RuntimeError(f"No attribute files found in {attributes_path}")
        
        # Read-in attributes into one big dataframe
        dfs = []
        for txt_file in txt_files:
            df_temp = pd.read_csv(txt_file, sep=';', header=0, dtype={'gauge_id': str})
            df_temp = df_temp.set_index('gauge_id')
            dfs.append(df_temp)
        
        df = pd.concat(dfs, axis=1)

    if basins:
        # 确保basin ID格式一致
        basins_str = [str(b).zfill(3) for b in basins]
        if any(b not in df.index for b in basins_str):
            missing_basins = [b for b in basins_str if b not in df.index]
            raise ValueError(f'Some basins are missing static attributes: {missing_basins}')
        df = df.loc[basins_str]

    return df


def load_camels_us_forcings(data_dir: Path, basin: str, forcings: str) -> pd.DataFrame:
    """Load the forcing data for a basin of the CAMELS US data set.

    Parameters
    ----------
    data_dir : Path
        Path to the CAMELS US directory. This folder must contain a 'basin_mean_forcing' folder containing one 
        subdirectory for each forcing. The forcing directories have to contain 18 subdirectories (for the 18 HUCS) as in
        the original CAMELS data set. In each HUC folder are the forcing files (.txt), starting with the 8-digit basin 
        id.
    basin : str
        8-digit USGS identifier of the basin.
    forcings : str
        Can be e.g. 'daymet' or 'nldas', etc. Must match the folder names in the 'basin_mean_forcing' directory.

    Returns
    -------
    pd.DataFrame
        Time-indexed DataFrame, containing the forcing data.
    int
        Catchment area (m2), specified in the header of the forcing file.
    """
    # 优先使用优化后的数据，如果不存在则使用原始数据
    optimized_path = data_dir / 'output_merged_optimized'
    if optimized_path.exists():
        forcing_path = optimized_path
    else:
        forcing_path = data_dir / 'output_merged_direct'
    if not forcing_path.is_dir():
        raise OSError(f"{forcing_path} does not exist")

    file_path = list(forcing_path.glob(f'extracted_varssim{basin}.csv'))
    if file_path:
        file_path = file_path[0]
    else:
        raise FileNotFoundError(f'No file for Basin {basin} at {file_path}')
    df = pd.read_csv(file_path)
    # 转换时间列为datetime格式并设置为索引
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    # 在重采样前，将无效值（-999）替换为NaN
    if "Obs flow" in df.columns:
        df.loc[df["Obs flow"] < 0, "Obs flow"] = np.nan
    if "QObs" in df.columns:
        df.loc[df["QObs"] < 0, "QObs"] = np.nan
    # 重新采样到日频率，填充缺失值（如果数据不是日频率）
    if df.index.freq is None or df.index.freq != 'D':
        df = df.resample('D').mean()
    # 检查是否使用优化后的数据（已包含合并变量）
    is_optimized = 'wind_speed' in df.columns or 'swvl_deep' in df.columns or 'PET' in df.columns
    
    if is_optimized:
        # 优化后的数据处理：填充缺失值
        vars_to_fill = ['sd', 'snowc', 'sf', 'swvl1', 'swvl_deep', 'stl1', 'wind_speed', 'PET']
        for var in vars_to_fill:
            if var in df.columns:
                df[var] = df[var].ffill().bfill()
    else:
        # 原始数据处理：对新变量的缺失值进行前向填充，然后后向填充（处理序列开头的缺失值）
        new_vars = ['asn', 'sd', 'swvl1', 'swvl2', 'swvl3', 'swvl4', 'u10', 'v10']
        for var in new_vars:
            if var in df.columns:
                df[var] = df[var].ffill().bfill()
        
        # 合并风速: u10, v10 -> wind_speed
        if 'u10' in df.columns and 'v10' in df.columns:
            df['wind_speed'] = np.sqrt(df['u10']**2 + df['v10']**2)
            df = df.drop(columns=['u10', 'v10'])
        
        # 合并深层土壤: swvl2-4 -> swvl_deep
        if all(col in df.columns for col in ['swvl2', 'swvl3', 'swvl4']):
            df['swvl_deep'] = (df['swvl2'] + df['swvl3'] + df['swvl4']) / 3
            df = df.drop(columns=['swvl2', 'swvl3', 'swvl4'])
        
        # 移除不需要的雪变量
        vars_to_remove = ['sde', 'rsn', 'src', 'asn']
        for var in vars_to_remove:
            if var in df.columns:
                df = df.drop(columns=[var])
    
    # 重命名列（统一处理）
    # 如果目标列已存在，删除源列避免重复；否则重命名
    rename_dict = {
        "Obs flow": "QObs",  # 目标变量
        "Precip": "hist_pr",  # 降水
        "Temp": "hist_tas",  # 气温
    }
    
    # 处理重命名：如果目标列已存在，删除源列；否则重命名
    for old_name, new_name in rename_dict.items():
        if old_name in df.columns:
            if new_name in df.columns:
                # 目标列已存在，删除源列
                df = df.drop(columns=[old_name])
            else:
                # 目标列不存在，重命名
                df = df.rename(columns={old_name: new_name})

    return df


def load_camels_us_discharge(data_dir: Path, basin: str) -> pd.Series:
    """Load the discharge data for a basin of the CAMELS US data set.

    Parameters
    ----------
    data_dir : Path
        Path to the CAMELS US directory. This folder must contain a 'usgs_streamflow' folder with 18
        subdirectories (for the 18 HUCS) as in the original CAMELS data set. In each HUC folder are the discharge files 
        (.txt), starting with the 8-digit basin id.
    basin : str
        8-digit USGS identifier of the basin.
    area : int
        Catchment area (m2), used to normalize the discharge.

    Returns
    -------
    pd.Series
        Time-index pandas.Series of the discharge values (mm/day)
    """

    discharge_path = data_dir / 'usgs_streamflow'
    file_path = list(discharge_path.glob(f'**/{basin}_streamflow_qc.csv'))
    if file_path:
        file_path = file_path[0]
    else:
        raise FileNotFoundError(f'No file for Basin {basin} at {file_path}')

    col_names = ['basin', 'Year', 'Mnth', 'Day', 'QObs']
    df = pd.read_csv(file_path, header=None, names=col_names)
    df["date"] = pd.to_datetime(df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str), format="%Y/%m/%d")
    df = df.set_index("date")

    # normalize discharge from cubic feet per second to mm per day
    #df.QObs = 28316846.592 * df.QObs * 86400 / (area * 10**6)

    return df.QObs
