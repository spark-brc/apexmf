""" PEST support utilities: 12/4/2019 created by Seonggyu Park
    last modified day: 10/14/2020 by Seonggyu Park
"""

from unittest.mock import NonCallableMagicMock
import pandas as pd
import numpy as np
import time
from pyemu.pst.pst_utils import SFMT,IFMT,FFMT
import os
import shutil
import socket
import multiprocessing as mp
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm
from termcolor import colored
import glob


opt_files_path = os.path.join(
                    os.path.dirname(os.path.abspath( __file__ )),
                    'opt_files')
foward_path = os.path.dirname(os.path.abspath( __file__ ))


def create_apexmf_con(
                wd,  sim_start, cal_start, cal_end,
                cha_file=None, subs=None,
                gw_level=None, grids=None,
                lai_file=None, lai_subs=None,
                salt_subs=None,
                riv_parm=None,  baseflow=None,
                fdc=None, min_fdc=None, max_fdc=None, interval_num=None,
                time_step=None,
                pp_included=None
                ):
    """create apexmf.con file containg APEX-MODFLOW model PEST initial settings

    Args:
        wd (`str`): APEX-MODFLOW working directory
        subs (`list`): reach numbers to be extracted
        grids (`list`): grid numbers to be extracted
        sim_start (`str`): simulation start date e.g. '1/1/2000'
        cal_start (`str`): calibration start date e.g., '1/1/2001'
        cal_end (`str`): calibration end date e.g., '12/31/2005'
        lai_subs (`list`): lai sub numbers to be extracted
        time_step (`str`, optional): model time step. Defaults to None ('day'). e.g., 'day', 'month', 'year'
        riv_parm (`str`, optional): river parameter activation. Defaults to None ('n').
        depth_to_water (`str`, optional): extracting simulated depth to water activation. Defaults to None ('n').
        baseflow (`str`, optional): extracting baseflow ratio activation. Defaults to None ('n').

    Returns:
        dataframe: return APEX-MODFLOW PEST configure settings as dataframe and exporting it as apexmf.con file.
    """
    if cha_file is None:
        cha_file = 'n'
        subs = 'n'
    if gw_level is None:
        gw_level ='n'
        grids = 'n'
    else:
        gw_level = 'y'
    if lai_file is None:
        lai_file = 'n'
        lai_subs = 'n'
    if salt_subs is None:
        salt_subs = 'n'

    if time_step is None:
        time_step = 'day'
    if riv_parm is None:
        riv_parm = 'n'
    else:
        riv_parm = 'y'
    if baseflow is None:
        baseflow = 'n'
    else:
        baseflow = 'y'
    if fdc is None:
        fdc = 'n'
    if  min_fdc is None:
        min_fdc = 10 
    if  max_fdc is None:
        max_fdc = 90
    if  interval_num is None:
        interval_num = 20
    if pp_included is None:
        pp_included = 'n'

    col01 = [
        'wd', 'mfwd', 'sim_start', 'cal_start', 'cal_end',
        'cha_file', 'subs', 
        'gw_level', 'grids',
        'lai_file', 'lai_subs',
        'salt_subs',
        'riv_parm', 'baseflow',
        'fdc', 'min_fdc', 'max_fdc', 'interval_num',
        'time_step',
        'pp_included',
        ]
    col02 = [
        wd, wd+'/MODFLOW', sim_start, cal_start, cal_end, 
        cha_file, subs,
        gw_level, grids,
        lai_file, lai_subs,
        salt_subs, 
        riv_parm, baseflow,
        fdc, min_fdc, max_fdc, interval_num,
        time_step,
        pp_included,
        ]
    df = pd.DataFrame({'names': col01, 'vals': col02})
    with open(os.path.join(wd, 'apexmf.con'), 'w', newline='') as f:
        f.write("# apexmf.con created by apexmf\n")
        df.to_csv(f, sep='\t', encoding='utf-8', index=False, header=False)
    return df

def init_setup(wd):
    filesToCopy = [
        "beopest64.exe",
        "i64pest.exe",
        "i64pwtadj1.exe",
        "crop.parm",
        "apex.parm.xlsx"
        ]
    
    suffix = ' passed'

    for j in filesToCopy:
        if not os.path.isfile(os.path.join(wd, j)):
            shutil.copy2(os.path.join(opt_files_path, j), os.path.join(wd, j))
            print(" '{}' file copied ...".format(j) + colored(suffix, 'green'))
    # if not os.path.isfile(os.path.join(wd, 'forward_run.py')):
    shutil.copy2(os.path.join(foward_path, 'forward_run.py'), os.path.join(wd, 'forward_run.py'))
    print(" '{}' file copied ...".format('forward_run.py') + colored(suffix, 'green'))        
    shutil.copy2(os.path.join(foward_path, 'salt_forward_run.py'), os.path.join(wd, 'salt_forward_run.py'))
    print(" '{}' file copied ...".format('salt_forward_run.py') + colored(suffix, 'green'))      

def fix_riv_pkg(wd, riv_file):
    """ Delete duplicate river cells in an existing MODFLOW river packgage.

    Args:
        wd ('str'): path of the working directory.
        riv_file ('str'): name of river package.
    """

    with open(os.path.join(wd, riv_file), "r") as fp:
        lines = fp.readlines()
        new_lines = []
        for line in lines:
            #- Strip white spaces
            line = line.strip()
            if line not in new_lines:
                new_lines.append(line)
            else:
                print('here')

    output_file = "{}_fixed".format(riv_file)
    with open(os.path.join(wd, output_file), "w") as fp:
        fp.write("\n".join(new_lines))    

# stf discharge
def extract_day_stf(rch_file, channels, start_day, cali_start_day, cali_end_day):
    """extract a daily simulated streamflow from the output.rch file,
        store it in each channel file.

    Args:
        - rch_file (`str`): the path and name of the existing output file
        - channels (`list`): channel number in a list, e.g. [9, 60]
        - start_day ('str'): simulation start day after warm period, e.g. '1/1/1985'
        - end_day ('str'): simulation end day e.g. '12/31/2005'

    Example:
        apexmf_pst_utils.extract_month_str('path', [9, 60], '1/1/1993', '1/1/1993', '12/31/2000')
    """

    for i in channels:
        sim_stf = pd.read_csv(
                        rch_file,
                        delim_whitespace=True,
                        skiprows=9,
                        usecols=[0, 1, 8],
                        names=["idx", "sub", "str_sim"],
                        index_col=0)
        sim_stf = sim_stf.loc["REACH"]

        sim_stf_f = sim_stf.loc[sim_stf["sub"] == int(i)]
        sim_stf_f = sim_stf_f.drop(['sub'], axis=1)
        sim_stf_f.index = pd.date_range(start_day, periods=len(sim_stf_f.str_sim))
        sim_stf_f = sim_stf_f[cali_start_day:cali_end_day]
        sim_stf_f.to_csv('cha_{:03d}.txt'.format(i), sep='\t', encoding='utf-8', index=True, header=False, float_format='%.7e')
        print('cha_{:03d}.txt file has been created...'.format(i))
    print('Finished ...')


def extract_month_stf(rch_file, channels, start_day, cali_start_day, cali_end_day):
    """extract a simulated streamflow from the output.rch file,
       store it in each channel file.

    Args:
        - rch_file (`str`): the path and name of the existing output file
        - channels (`list`): channel number in a list, e.g. [9, 60]
        - start_day ('str'): simulation start day after warm period, e.g. '1/1/1985'
        - end_day ('str'): simulation end day e.g. '12/31/2005'

    Example:
        apexmf_pst_utils.extract_month_str('path', [9, 60], '1/1/1993', '1/1/1993', '12/31/2000')
    """

    for i in channels:
        sim_stf = pd.read_csv(
                        rch_file,
                        delim_whitespace=True,
                        skiprows=9,
                        usecols=[0, 1, 8],
                        names=["idx", "sub", "str_sim"],
                        index_col=0)
        sim_stf = sim_stf.loc["REACH"]
        sim_stf_f = sim_stf.loc[sim_stf["sub"] == int(i)]
        sim_stf_f = sim_stf_f.drop(['sub'], axis=1)
        sim_stf_f.index = pd.date_range(start_day, periods=len(sim_stf_f.str_sim), freq='M')
        sim_stf_f = sim_stf_f[cali_start_day:cali_end_day]
        sim_stf_f.to_csv('stf_{:03d}.txt'.format(i), sep='\t', encoding='utf-8', index=True, header=False, float_format='%.7e')
        print('stf_{:03d}.txt file has been created...'.format(i))
    print('Finished ...')
    return sim_stf_f


# extract sed
def extract_month_sed(rch_file, channels, start_day, cali_start_day, cali_end_day):
    """extract a simulated sediment from the output.rch file,
       store it in each channel file.

    Args:
        - rch_file (`str`): the path and name of the existing output file
        - channels (`list`): channel number in a list, e.g. [9, 60]
        - start_day ('str'): simulation start day after warm period, e.g. '1/1/1985'
        - end_day ('str'): simulation end day e.g. '12/31/2005'

    Example:
        apexmf_pst_utils.extract_month_str('path', [9, 60], '1/1/1993', '1/1/1993', '12/31/2000')
    """

    for i in channels:
        sim_stf = pd.read_csv(
                        rch_file,
                        delim_whitespace=True,
                        skiprows=9,
                        usecols=[0, 1, 12],
                        names=["idx", "sub", "sed_sim"],
                        index_col=0)
        sim_stf = sim_stf.loc["REACH"]
        sim_stf_f = sim_stf.loc[sim_stf["sub"] == int(i)]
        sim_stf_f = sim_stf_f.drop(['sub'], axis=1)
        sim_stf_f.index = pd.date_range(start_day, periods=len(sim_stf_f.sed_sim), freq='M')
        sim_stf_f = sim_stf_f[cali_start_day:cali_end_day]
        sim_stf_f.to_csv('sed_{:03d}.txt'.format(i), sep='\t', encoding='utf-8', index=True, header=False, float_format='%.7e')
        print('sed_{:03d}.txt file has been created...'.format(i))
    print('Finished ...')

# extract baseflow
def extract_month_baseflow(sub_file, channels, start_day, cali_start_day, cali_end_day):
    """ extract a simulated baseflow rates from the output.sub file,
        store it in each channel file.

    Args:
        - sub_file (`str`): the path and name of the existing output file
        - channels (`list`): channel number in a list, e.g. [9, 60]
        - start_day ('str'): simulation start day after warm period, e.g. '1/1/1985'
        - end_day ('str'): simulation end day e.g. '12/31/2005'

    Example:
        apexmf_pst_utils.extract_month_baseflow('path', [9, 60], '1/1/1993', '1/1/1993', '12/31/2000')
    """
    gwqs = []
    subs = []
    for i in channels:
        sim_stf = pd.read_csv(
                        sub_file,
                        delim_whitespace=True,
                        skiprows=9,
                        usecols=[1, 3, 10, 11, 19],
                        names=["date", "filter", "surq", "gwq", "latq"],
                        index_col=0)
        
        sim_stf_f = sim_stf.loc[i]
        # sim_stf_f["filter"]= sim_stf_f["filter"].astype(str) 
        sim_stf_f = sim_stf_f[sim_stf_f['filter'].astype(str).map(len) < 13]
        sim_stf_f = sim_stf_f.drop(['filter'], axis=1)
        sim_stf_f.index = pd.date_range(start_day, periods=len(sim_stf_f.surq), freq='M')
        sim_stf_f = sim_stf_f[cali_start_day:cali_end_day]
        # sim_stf_f.to_csv('gwq_{:03d}.txt'.format(i), sep='\t', encoding='utf-8', index=True, header=False, float_format='%.7e')
        
        sim_stf_f['surq'] = sim_stf_f['surq'].astype(float)
        sim_stf_f['bf_rate'] = sim_stf_f['gwq']/ (sim_stf_f['surq'] + sim_stf_f['latq'] + sim_stf_f['gwq'])
        sim_stf_f.loc[sim_stf_f['gwq'] < 0, 'bf_rate'] = 0     
        bf_rate = sim_stf_f['bf_rate'].mean()
        # bf_rate = bf_rate.item()
        subs.append('bfr_{:03d}'.format(i))
        gwqs.append(bf_rate)
        print('Average baseflow rate for {:03d} has been calculated ...'.format(i))
    # Combine lists into array
    bfr_f = np.c_[subs, gwqs]
    with open('baseflow_ratio.out', "w", newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        for item in bfr_f:
            writer.writerow([(item[0]),
                '{:.4f}'.format(float(item[1]))
                ])
    print('Finished ...\n')


def extract_depth_to_water(grid_ids, start_day, end_day):
    """extract a simulated streamflow from the output.rch file,
        store it in each channel file.

    Args:
        - rch_file (`str`): the path and name of the existing output file
        - channels (`list`): channel number in a list, e.g. [9, 60]
        - start_day ('str'): simulation start day, e.g. '1/1/1985'
        - end_day ('str'): simulation end day e.g. '12/31/2000'

    Example:
        pest_utils.extract_depth_to_water('path', [9, 60], '1/1/1993', '12/31/2000')
    """
    if not os.path.exists('MODFLOW/amf_MODFLOW_obs_head'):
        raise Exception("'amf_MODFLOW_obs_head' file not found")
    if not os.path.exists('MODFLOW/modflow.obs'):
        raise Exception("'modflow.obs' file not found")
    mf_obs_grid_ids = pd.read_csv(
                        'MODFLOW/modflow.obs',
                        sep=r'\s+',
                        usecols=[3, 4],
                        skiprows=2,
                        header=None
                        )
    col_names = mf_obs_grid_ids.iloc[:, 0].tolist()

    # set index by modflow grid ids
    mf_obs_grid_ids = mf_obs_grid_ids.set_index([3])
    mf_sim = pd.read_csv(
                        'MODFLOW/amf_MODFLOW_obs_head', skiprows=1, sep=r'\s+',
                        names=col_names,
                        # usecols=grid_ids,
                        )
    mf_sim = mf_sim.loc[:, grid_ids]
    mf_sim.index = pd.date_range(start_day, periods=len(mf_sim))
    mf_sim = mf_sim[start_day:end_day]
    for i in grid_ids:
        elev = mf_obs_grid_ids.loc[i].values  # use land surface elevation to get depth to water
        (mf_sim.loc[:, i] - elev).to_csv(
        # abs(elev - mf_sim.loc[:, i]).to_csv(
                        'dtw_{}.txt'.format(i), sep='\t', encoding='utf-8',
                        index=True, header=False, float_format='%.7e'
                        )
        print('dtw_{}.txt file has been created...'.format(i))
    print('Finished ...')


def cvt_strobd_dtm():
    stf_obd_inf = 'streamflow.obd'
    stf_obd = pd.read_csv(
                        stf_obd_inf,
                        sep='\t',
                        index_col=0,
                        parse_dates=True,
                        na_values=[-999, '']
                        )
    mstf_obd = stf_obd.resample('M').mean()
    mstf_obd.to_csv('streamflow_month.obd', float_format='%.2f', sep='\t', na_rep=-999)

# extracct lai
def extract_day_lai(sao_df, subs, start_day, cali_start_day, cali_end_day):
    lai_sim_files_ = []
    for i in subs:
        lai_df = sao_df.loc[sao_df['SAID']==i, ['GIS', 'TIME', 'LAI-']]
        lai_df = lai_df.groupby(['GIS', 'TIME']).sum()
        lai_df.index = pd.date_range(start_day, periods=len(lai_df))
        lai_df = lai_df[cali_start_day:cali_end_day]
        lai_df.to_csv('lai_{:03d}.txt'.format(i), sep='\t', encoding='utf-8', index=True, header=False, float_format='%.7e')
        print('lai_{:03d}.txt file has been created...'.format(i))
        lai_sim_files_.append('lai_{:03d}.txt'.format(i))
    print('Finished ...')
    return lai_sim_files_


def extract_mon_lai(sao_df, subs, start_day, cali_start_day, cali_end_day):
    lai_sim_files_ = []
    for i in subs:
        lai_df = sao_df.loc[sao_df['SAID']==i, ['GIS', 'TIME', 'LAI-']]
        lai_df = lai_df.groupby(['GIS', 'TIME']).sum()
        lai_df.index = pd.date_range(start_day, periods=len(lai_df), freq='M')
        lai_df = lai_df[cali_start_day:cali_end_day]
        lai_df.to_csv('lai_{:03d}.txt'.format(i), sep='\t', encoding='utf-8', index=True, header=False, float_format='%.7e')
        print('lai_{:03d}.txt file has been created...'.format(i))
        lai_sim_files_.append('lai_{:03d}.txt'.format(i))
    print('Finished ...')        
    return lai_sim_files_


def stf_obd_to_ins(srch_file, col_name, start_day, end_day, time_step=None):
    """extract a simulated streamflow from the output.rch file,
        store it in each channel file.

    Args:
        - rch_file (`str`): the path and name of the existing output file
        - channels (`list`): channel number in a list, e.g. [9, 60]
        - start_day ('str'): calibration start day, e.g. '1/1/1993'
        - end_day ('str'): calibration end day e.g. '12/31/2000'
        - time_step (`str`): day, month, year

    Example:
        pest_utils.extract_month_str('path', [9, 60], '1/1/1993', '12/31/2000')
    """ 
    if time_step is None:
        time_step = 'day'

    if time_step == 'month':
        stf_obd_inf = 'stf_mon.obd'
    else:
        stf_obd_inf = 'streamflow.obd'
    stf_obd = pd.read_csv(
                        stf_obd_inf,
                        sep='\t',
                        usecols=['date', col_name],
                        index_col=0,
                        parse_dates=True,
                        na_values=[-999, '']
                        )
    
    stf_obd = stf_obd[start_day:end_day]
    stf_sim = pd.read_csv(
                        srch_file,
                        delim_whitespace=True,
                        names=["date", "str_sim"],
                        index_col=0,
                        parse_dates=True)
    result = pd.concat([stf_obd, stf_sim], axis=1)
    result['tdate'] = pd.to_datetime(result.index)
    result['month'] = result['tdate'].dt.month
    result['year'] = result['tdate'].dt.year
    result['day'] = result['tdate'].dt.day

    if time_step == 'day':
        result['ins'] = (
                        'l1 w !{}_'.format(col_name) + result["year"].map(str) +
                        result["month"].map('{:02d}'.format) +
                        result["day"].map('{:02d}'.format) + '!'
                        )
    elif time_step == 'month':
        result['ins'] = 'l1 w !{}_'.format(col_name) + result["year"].map(str) + result["month"].map('{:02d}'.format) + '!'
    else:
        print('are you performing a yearly calibration?')
    result['{}_ins'.format(col_name)] = np.where(result[col_name].isnull(), 'l1', result['ins'])

    with open(srch_file+'.ins', "w", newline='') as f:
        f.write("pif ~" + "\n")
        result['{}_ins'.format(col_name)].to_csv(f, sep='\t', encoding='utf-8', index=False, header=False)
    print('{}.ins file has been created...'.format(srch_file))
    return result['{}_ins'.format(col_name)]


def mf_obd_to_ins(wt_file, col_name, start_day, end_day, time_step=None):
    """extract a simulated streamflow from the output.rch file,
        store it in each channel file.

    Args:
        - rch_file (`str`): the path and name of the existing output file
        - channels (`list`): channel number in a list, e.g. [9, 60]
        - start_day ('str'): simulation start day, e.g. '1/1/1993'
        - end_day ('str'): simulation end day e.g. '12/31/2000'

    Example:
        pest_utils.extract_month_str('path', [9, 60], '1/1/1993', '12/31/2000')
    """ 

    if time_step is None:
        time_step = 'day'
    if time_step == 'month':
        wt_obd_inf = 'dtw_mon.obd'
    else:
        wt_obd_inf = 'dtw_day.obd'
    mf_obd = pd.read_csv(
                        'MODFLOW/' + wt_obd_inf,
                        sep='\t',
                        usecols=['date', col_name],
                        index_col=0,
                        parse_dates=True,
                        na_values=[-999, ""]
                        )
    mf_obd = mf_obd[start_day:end_day]

    wt_sim = pd.read_csv(
                        wt_file,
                        delim_whitespace=True,
                        names=["date", "str_sim"],
                        index_col=0,
                        parse_dates=True)
    result = pd.concat([mf_obd, wt_sim], axis=1)

    result['tdate'] = pd.to_datetime(result.index)
    result['day'] = result['tdate'].dt.day
    result['month'] = result['tdate'].dt.month
    result['year'] = result['tdate'].dt.year
    if time_step == 'day':
        result['ins'] = (
                        'l1 w !{}_'.format(col_name) + result["year"].map(str) +
                        result["month"].map('{:02d}'.format) +
                        result["day"].map('{:02d}'.format) + '!'
                        )    
    elif time_step == 'month':
        result['ins'] = 'l1 w !{}_'.format(col_name) + result["year"].map(str) + result["month"].map('{:02d}'.format) + '!'
    else:
        print('are you performing a yearly calibration?')
    result['{}_ins'.format(col_name)] = np.where(result[col_name].isnull(), 'l1', result['ins'])

    with open(wt_file+'.ins', "w", newline='') as f:
        f.write("pif ~" + "\n")
        result['{}_ins'.format(col_name)].to_csv(f, sep='\t', encoding='utf-8', index=False, header=False)
    print('{}.ins file has been created...'.format(wt_file))

    # return result['{}_ins'.format(col_name)]


def lai_obd_to_ins(srch_file, col_name, start_day, end_day, time_step=None):
    """extract a simulated streamflow from the output.rch file,
        store it in each channel file.

    Args:
        - rch_file (`str`): the path and name of the existing output file
        - channels (`list`): channel number in a list, e.g. [9, 60]
        - start_day ('str'): calibration start day, e.g. '1/1/1993'
        - end_day ('str'): calibration end day e.g. '12/31/2000'
        - time_step (`str`): day, month, year

    Example:
        pest_utils.extract_month_str('path', [9, 60], '1/1/1993', '12/31/2000')
    """ 
    if time_step is None:
        time_step = 'day'

    if time_step == 'month':
        stf_obd_inf = 'lai_mon.obd'
    else:
        stf_obd_inf = 'lai_day.obd'
    stf_obd = pd.read_csv(
                        stf_obd_inf,
                        sep='\t',
                        usecols=['date', col_name],
                        index_col=0,
                        parse_dates=True,
                        na_values=[-999, '']
                        )
    
    stf_obd = stf_obd[start_day:end_day]
    stf_sim = pd.read_csv(
                        srch_file,
                        delim_whitespace=True,
                        names=["date", "lai_sim"],
                        index_col=0,
                        parse_dates=True)
    result = pd.concat([stf_obd, stf_sim], axis=1)
    result['tdate'] = pd.to_datetime(result.index)
    result['month'] = result['tdate'].dt.month
    result['year'] = result['tdate'].dt.year
    result['day'] = result['tdate'].dt.day

    if time_step == 'day':
        result['ins'] = (
                        'l1 w !{}_'.format(col_name) + result["year"].map(str) +
                        result["month"].map('{:02d}'.format) +
                        result["day"].map('{:02d}'.format) + '!'
                        )
    elif time_step == 'month':
        result['ins'] = 'l1 w !{}_'.format(col_name) + result["year"].map(str) + result["month"].map('{:02d}'.format) + '!'
    else:
        print('are you performing a yearly calibration?')
    result['{}_ins'.format(col_name)] = np.where(result[col_name].isnull(), 'l1', result['ins'])

    with open(srch_file+'.ins', "w", newline='') as f:
        f.write("pif ~" + "\n")
        result['{}_ins'.format(col_name)].to_csv(f, sep='\t', encoding='utf-8', index=False, header=False)
    print('{}.ins file has been created...'.format(srch_file))
    return result['{}_ins'.format(col_name)]


def fdc_obd_to_ins(fdc_sims, fdc_obds):
    for fdc_sim_inf, fdc_obd_inf in zip(fdc_sims, fdc_obds):
        fdc_obd = pd.read_csv(
                            fdc_obd_inf,
                            sep='\t',
                            names=["interval", "obd_slp"],
                            index_col=0,
                            na_values=[-999, '']
                            )
        fdc_sim = pd.read_csv(
                            fdc_sim_inf,
                            delim_whitespace=True,
                            names=["interval", "sim_slp"],
                            index_col=0
                            )
        result = pd.concat([fdc_obd, fdc_sim], axis=1)
        result['ins'] = (
                        'l1 w !{}_slp_'.format(fdc_sim_inf[:-4])+result.index.map('{:03d}'.format)+'!'
                        )
        with open(fdc_sim_inf+'.ins', "w", newline='') as f:
            f.write("pif ~" + "\n")
            result['ins'].to_csv(f, sep='\t', encoding='utf-8', index=False, header=False)
        print('{}.ins file has been created...'.format(fdc_sim_inf))


def salt_obd_to_ins(sub_id, salt_sim_file, obd_file, col_name, start_day, end_day, time_step=None):
    """create instruction files using simulated and observed data

    Args:
        salt_sim_file (path): salt.output.channels path
        obd_file (path): salt obd file
        col_name (str): obd col name
        start_day (str): simulation start day
        end_day (str): calibration end day
        time_step (str, optional): simulation time step. Defaults to None.

    Returns:
        dataframe: ins data
    """
    
     
    if time_step is None:
        time_step = 'day'
    if time_step == 'month' or time_step == 'mon' or time_step == 'm':
        time_step = 'mon'

    salt_obd = pd.read_csv(
                        obd_file,
                        sep='\t',
                        usecols=['date', col_name],
                        index_col=0,
                        parse_dates=True,
                        na_values=[-999, '']
                        )
    # Remove pandas rows with duplicate indices
    salt_obd = salt_obd[~salt_obd.index.duplicated(keep='first')]
    salt_obd = salt_obd[start_day:end_day]
    salt_sim = pd.read_csv(
                        salt_sim_file,
                        delim_whitespace=True,
                        names=["date", "salt_sim"],
                        index_col=0,
                        parse_dates=True)
    result = pd.concat([salt_obd, salt_sim], axis=1)
    result['tdate'] = pd.to_datetime(result.index)
    result['month'] = result['tdate'].dt.month
    result['year'] = result['tdate'].dt.year
    result['day'] = result['tdate'].dt.day


    # if time_step == 'day':
    #     result['ins'] = (
    #                     'l1 w !{}_'.format(col_name) + result["year"].map(str)[2:] +
    #                     result["month"].map('{:02d}'.format) +
    #                     result["day"].map('{:02d}'.format) + '!'
    #                     )
    # elif time_step == 'mon':
    #     result['ins'] = 'l1 w !{}_'.format(col_name) + result["year"].map(str) + result["month"].map('{:02d}'.format) + '!'
    # else:
    #     print('are you performing a yearly calibration?')
        
    # result['{}_ins'.format(col_name)] = np.where(result[col_name].isnull(), 'l1', result['ins'])

    # with open(salt_sim_file+'.ins', "w", newline='') as f:
    #     f.write("pif ~" + "\n")
    #     result['{}_ins'.format(col_name)].to_csv(f, sep='\t', encoding='utf-8', index=False, header=False)
    # print('{}.ins file has been created...'.format(salt_sim_file))
    # return result['{}_ins'.format(col_name)]


    col_namef = col_name[0]+col_name[5:]
    
    if time_step == 'day':
        result['ins'] = (
                        'l1 w !d{:03d}_{}_'.format(sub_id, col_namef) + result["year"].map(str) +
                        result["month"].map('{:02d}'.format) +
                        result["day"].map('{:02d}'.format) + '!'
                        )
    elif time_step == 'mon':
        result['ins'] = 'l1 w !m{:03d}_{}_'.format(sub_id, col_namef)+ result["year"].map(str) + result["month"].map('{:02d}'.format) + '!'
    else:
        print('are you performing a yearly calibration?')

    result['{}_ins'.format(col_name)] = np.where(result[col_name].isnull(), 'l1', result['ins'])

    with open(salt_sim_file+'.ins', "w", newline='') as f:
        f.write("pif ~" + "\n")
        result['{}_ins'.format(col_name)].to_csv(f, sep='\t', encoding='utf-8', index=False, header=False)
    print('{}.ins file has been created...'.format(salt_sim_file))
    return result['{}_ins'.format(col_name)]


def extract_month_avg(cha_file, channels, start_day, cal_day=None, end_day=None):
    """extract a simulated streamflow from the channel_day.txt file,
        store it in each channel file.

    Args:
        - cha_file (`str`): the path and name of the existing output file
        - channels (`list`): channel number in a list, e.g. [9, 60]
        - start_day ('str'): simulation start day after warm period, e.g. '1/1/1993'
        - end_day ('str'): simulation end day e.g. '12/31/2000'

    Example:
        pest_utils.extract_month_str('path', [9, 60], '1/1/1993', '12/31/2000')
    """

    for i in channels:
        # Get only necessary simulated streamflow and convert monthly average streamflow
        os.chdir(cha_file)
        print(os.getcwd())
        df_str = pd.read_csv(
                            "channel_day.txt",
                            delim_whitespace=True,
                            skiprows=3,
                            usecols=[6, 8],
                            names=['name', 'flo_out'],
                            header=None
                            )
        df_str = df_str.loc[df_str['name'] == 'cha{:02d}'.format(i)]
        df_str.index = pd.date_range(start_day, periods=len(df_str.flo_out))
        mdf = df_str.resample('M').mean()
        mdf.index.name = 'date'
        if cal_day is None:
            cal_day = start_day
        else:
            cal_day = cal_day
        if end_day is None:
            mdf = mdf[cal_day:]
        else:
            mdf = mdf[cal_day:end_day]
        mdf.to_csv('cha_mon_avg_{:03d}.txt'.format(i), sep='\t', float_format='%.7e')
        print('cha_{:03d}.txt file has been created...'.format(i))
        return mdf


def model_in_to_template_file(model_in_file, tpl_file=None):
    """write a template file for a APEX parameter value file (model.in).

    Args:
        model_in_file (`str`): the path and name of the existing model.in file
        tpl_file (`str`, optional):  template file to write. If None, use
            `model_in_file` +".tpl". Default is None
    Note:
        Uses names in the first column in the pval file as par names.

    Example:
        pest_utils.model_in_to_template_file('path')

    Returns:
        **pandas.DataFrame**: a dataFrame with template file information
    """

    if tpl_file is None:
        tpl_file = model_in_file + ".tpl"
    mod_df = pd.read_csv(
                        model_in_file,
                        delim_whitespace=True,
                        header=None, skiprows=0,
                        names=["parnme", "parval1"])
    mod_df.index = mod_df.parnme
    mod_df.loc[:, "tpl"] = mod_df.parnme.apply(lambda x: " ~   {0:15s}   ~".format(x[3:-4]))
    # mod_df.loc[:, "tpl"] = mod_df.parnme.apply(lambda x: " ~   {0:15s}   ~".format(x[3:7]))
    with open(tpl_file, 'w') as f:
        f.write("ptf ~\n")
        # f.write("{0:10d} #NP\n".format(mod_df.shape[0]))
        SFMT_LONG = lambda x: "{0:<50s} ".format(str(x))
        f.write(mod_df.loc[:, ["parnme", "tpl"]].to_string(
                                                        col_space=0,
                                                        formatters=[SFMT, SFMT],
                                                        index=False,
                                                        header=False,
                                                        justify="left"))
    return mod_df


def tobd_fdcobd(df, colnam, min_fdc, max_fdc, interval_num, plot_show=None):
    """convert time series of streamflow obd to fdc slope obd

    Args:
        df (`dataframe`): time series of streamflow observation data
        colnam (`str`): streamflow obd column name
        min_fdc (`int`): minimum value of exceedance range
        max_fdc (`int`): maximum value of exceedance range
        interval_num (`int`): number of intervals for observed points to be used for calibration
        plot_show (`boolean`, optional): show flow duration curve with slopes. Defaults to None.

    Returns:
        `dataframe`: slopes from fdc stored in *.obd file
    """
    sort = np.sort(df.loc[:, colnam])[::-1]
    exceedence = np.arange(1.,len(sort)+1) / len(sort)
    exc_x = exceedence*100

    df = pd.DataFrame({'exceed': exc_x, 'flow': sort})
    # argument: how many intervals you want to get for slope?
    df = df.loc[(df['exceed']>=min_fdc) & (df['exceed']<max_fdc)]
    intervals = np.linspace(min_fdc, max_fdc, num=interval_num+1)
    slopes = []
    if plot_show is not None:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(df['exceed'], df['flow'])
        for i in range (len(intervals)-1):
            min_set = intervals[i]
            max_set = intervals[i+1]
            df2 = df.loc[(df['exceed']>=min_set) & (df['exceed']<max_set)]
            m, b = np.polyfit(df2['exceed'], df2['flow'], 1)
            ax.text(df2['exceed'].mean(), df2['flow'].mean(), '{0:.4f}'.format(m), fontsize=8)
            ax.plot(df2['exceed'], m*df2['exceed'] + b, color='red', alpha=0.5)
            slopes.append(m)
        plt.show()
    if plot_show is None:
        for i in range (len(intervals)-1):
            min_set = intervals[i]
            max_set = intervals[i+1]
            df2 = df.loc[(df['exceed']>=min_set) & (df['exceed']<max_set)]
            m, b = np.polyfit(df2['exceed'], df2['flow'], 1)
            slopes.append(m)
    slopes_df_ = pd.DataFrame({colnam:slopes})
    slopes_df_.to_csv('fdc_{}.obd'.format(colnam), sep='\t', na_rep=-999, header=False, float_format='%.7e')
    return slopes_df_


def extract_slopesFrTimeSim(
            rch_file, channels, start_day, cali_start_day, cali_end_day, 
            min_fdc, max_fdc, interval_num, time_step=None, plot_show=None):
    """convert time series of streamflow obd to fdc slope obd

    Args:
        df (`dataframe`): time series of streamflow observation data
        colnam (`str`): streamflow obd column name
        min_fdc (`int`): minimum value of exceedance range
        max_fdc (`int`): maximum value of exceedance range
        interval_num (`int`): number of intervals for observed points to be used for calibration
        plot_show (`boolean`, optional): show flow duration curve with slopes. Defaults to None.

    Returns:
        `dataframe`: simulated slopes from fdc stored in *.txt file
    """
    fdc_sim_files_ =[]
    for cha in channels:
        sim_stf = pd.read_csv(
                        rch_file,
                        delim_whitespace=True,
                        skiprows=9,
                        usecols=[0, 1, 8],
                        names=["idx", "sub", "str_sim"],
                        index_col=0)
        sim_stf = sim_stf.loc["REACH"]
        sim_stf_f = sim_stf.loc[sim_stf["sub"] == int(cha)]
        sim_stf_f = sim_stf_f.drop(['sub'], axis=1)
        if time_step is None:
            sim_stf_f.index = pd.date_range(start_day, periods=len(sim_stf_f.str_sim))
        if time_step == 'M':
            sim_stf_f.index = pd.date_range(start_day, periods=len(sim_stf_f.str_sim), freq='M')
        sim_stf_f = sim_stf_f[cali_start_day:cali_end_day]
        sort = np.sort(sim_stf_f.iloc[:, 0])[::-1]
        exceedence = np.arange(1.,len(sort)+1) / len(sort)
        exc_x = exceedence*100

        fdc_df = pd.DataFrame({'exceed': exc_x, 'flow': sort})
        # argument: how many intervals you want to get for slope?
        fdc_df = fdc_df.loc[(fdc_df['exceed']>=min_fdc) & (fdc_df['exceed']<max_fdc)]
        intervals = np.linspace(min_fdc, max_fdc, num=interval_num+1)
        slopes = []
        if plot_show is not None:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(fdc_df['exceed'], fdc_df['flow'])
            for i in range (len(intervals)-1):
                min_set = intervals[i]
                max_set = intervals[i+1]
                df2 = fdc_df.loc[(fdc_df['exceed']>=min_set) & (fdc_df['exceed']<max_set)]
                m, b = np.polyfit(df2['exceed'], df2['flow'], 1)
                ax.text(df2['exceed'].mean(), df2['flow'].mean(), '{0:.4f}'.format(m), fontsize=8)
                ax.plot(df2['exceed'], m*df2['exceed'] + b, color='red', alpha=0.5)
                slopes.append(m)
            plt.show()
        if plot_show is None:
            for i in range (len(intervals)-1):
                min_set = intervals[i]
                max_set = intervals[i+1]
                df2 = fdc_df.loc[(fdc_df['exceed']>=min_set) & (fdc_df['exceed']<max_set)]
                m, b = np.polyfit(df2['exceed'], df2['flow'], 1)
                slopes.append(m)
        slopes_df_ = pd.DataFrame({'cha_{:03d}'.format(cha):slopes})
        slopes_df_.to_csv('fdc_{:03d}.txt'.format(cha), sep='\t', encoding='utf-8', index=True, header=False, float_format='%.7e')
        print('fdc_{:03d}.txt file has been created...'.format(cha))
        fdc_sim_files_.append('fdc_{:03d}.txt'.format(cha))
    print('Finished ...')
    return fdc_sim_files_

def extract_salt_results(salt_subs, sim_start, cal_start, cal_end):
    """extract a simulated streamflow from the output.rch file,
       store it in each channel file.

    Args:
        - rch_file (`str`): the path and name of the existing output file
        - channels (`list`): channel number in a list, e.g. [9, 60]
        - start_day ('str'): simulation start day after warm period, e.g. '1/1/1985'
        - end_day ('str'): simulation end day e.g. '12/31/2005'

    Example:
        apexmf_pst_utils.extract_month_str('path', [9, 60], '1/1/1993', '1/1/1993', '12/31/2000')
    """
    if not os.path.exists('SALINITY/salt.output.channels'):
        raise Exception("'salt.output.channels' file not found")

    salt_df = pd.read_csv(
                        "SALINITY/salt.output.channels",
                        delim_whitespace=True,
                        skiprows=4,
                        header=0,
                        index_col=0,
                        )
    salt_df = salt_df.iloc[:, 5:] # only cols we need
    for i in salt_subs:
        salt_dff = salt_df.loc[i]
        salt_dff.index = pd.date_range(sim_start, periods=len(salt_dff))
        salt_dff = salt_dff[cal_start:cal_end]
        colnams = salt_dff.columns
        # print out daily
        for cn in colnams:
            sdf = salt_dff.loc[:, cn]
            sdf.to_csv('salt_{}_{:03d}_day.txt'.format(cn, i), sep='\t', encoding='utf-8', index=True, header=False, float_format='%.7e')
            print('salt_{}_{:03d}_day.txt'.format(cn, i))
            msdf = sdf.resample('M').mean()
            msdf.to_csv('salt_{}_{:03d}_mon.txt'.format(cn, i), sep='\t', encoding='utf-8', index=True, header=False, float_format='%.7e')
            print('salt_{}_{:03d}_mon.txt'.format(cn, i))
    print('Finished ...')    

def extract_salt_results2(salt_subs, sim_start, cal_start, cal_end):
    """extract a simulated streamflow from the output.rch file,
       store it in each channel file.

    Args:
        - rch_file (`str`): the path and name of the existing output file
        - channels (`list`): channel number in a list, e.g. [9, 60]
        - start_day ('str'): simulation start day after warm period, e.g. '1/1/1985'
        - end_day ('str'): simulation end day e.g. '12/31/2005'

    Example:
        apexmf_pst_utils.extract_month_str('path', [9, 60], '1/1/1993', '1/1/1993', '12/31/2000')
    """
    if not os.path.exists('SALINITY/salt.output.channels'):
        raise Exception("'salt.output.channels' file not found")
    salt_df = pd.read_csv(
                        "SALINITY/salt.output.channels",
                        delim_whitespace=True,
                        skiprows=4,
                        header=0,
                        index_col=0,
                        )
    
    for i in salt_subs:
        sdf = pd.DataFrame()
        sdf.index = pd.date_range(sim_start, cal_end)

        salt_dff = salt_df.loc[i]
        salt_dff['tempd'] = salt_dff.loc[:, 'year'].astype(str) + salt_dff['day'].map('{:03d}'.format)
        salt_dff['date'] = pd.to_datetime(salt_dff.year, format='%Y') + pd.to_timedelta(salt_dff.day - 1, unit='d')
        salt_dff.set_index('date', inplace=True)
        salt_dff.drop(['tempd'], axis=1, inplace=True)
        sdf = pd.concat([sdf, salt_dff], axis=1)
        sdf = sdf.iloc[:, 5:] # only cols we need
        sdf.fillna(0, inplace=True)
        sdf = sdf[cal_start:cal_end]
        colnams = sdf.columns
        # print out daily
        for cn in colnams:
            sdff = sdf.loc[:, cn]
            sdff.to_csv('salt_{}_{:03d}_day.txt'.format(cn, i), sep='\t', encoding='utf-8', index=True, header=False, float_format='%.7e')
            print('salt_{}_{:03d}_day.txt'.format(cn, i))
            msdf = sdff.resample('M').mean()
            msdf.to_csv('salt_{}_{:03d}_mon.txt'.format(cn, i), sep='\t', encoding='utf-8', index=True, header=False, float_format='%.7e')
            print('salt_{}_{:03d}_mon.txt'.format(cn, i))
    print('Finished ...')    



def modify_mf_tpl_path(pst_model_input):
    for i in range(len(pst_model_input)):
        if (
            pst_model_input.iloc[i, 0][:2] == 'hk' or 
            pst_model_input.iloc[i, 0][:2] == 'sy' or
            pst_model_input.iloc[i, 0][:2] == 'mf' or
            pst_model_input.iloc[i, 0][:2] == 'rt' or
            pst_model_input.iloc[i, 0][:4] == 'salt'
            ):
            pst_model_input.iloc[i, 0] =  "MODFLOW" +'\\'+ pst_model_input.iloc[i, 0]
            pst_model_input.iloc[i, 1] =  "MODFLOW" +'\\'+ pst_model_input.iloc[i, 1]

    return pst_model_input

def _remove_readonly(func, path, excinfo):
    """remove readonly dirs, apparently only a windows issue
    add to all rmtree calls: shutil.rmtree(**,onerror=remove_readonly), wk"""
    os.chmod(path, 128)  # stat.S_IWRITE==128==normal
    func(path)


# NOTE: Update description
def execute_beopest(
                main_dir, pst, num_workers=None, worker_root='..', port=4005, local=True,
                reuse_workers=None, copy_files=None, restart=None):
    """Execute BeoPEST and workers on the local machine

    Args:
        main_dir (str): 
        pst (str): [description]
        num_workers ([type], optional): [description]. Defaults to None.
        worker_root (str, optional): [description]. Defaults to '..'.
        port (int, optional): [description]. Defaults to 4005.
        local (bool, optional): [description]. Defaults to True.
        reuse_workers ([type], optional): [description]. Defaults to None.

    Raises:
        Exception: [description]
        Exception: [description]
        Exception: [description]
        Exception: [description]
        Exception: [description]
        Exception: [description]
    """

    if not os.path.isdir(main_dir):
        raise Exception("master dir '{0}' not found".format(main_dir))
    if not os.path.isdir(worker_root):
        raise Exception("worker root dir not found")
    if num_workers is None:
        num_workers = mp.cpu_count()
    else:
        num_workers = int(num_workers)
    if local:
        hostname = "localhost"
    else:
        hostname = socket.gethostname()

    base_dir = os.getcwd()
    port = int(port)
    cwd = os.chdir(main_dir)
    if restart is None:
        os.system("start cmd /k beopest64 {0} /h :{1}".format(pst, port))
    else:
        os.system("start cmd /k beopest64 {0} /r /h :{1}".format(pst, port))
    time.sleep(1.5) # a few cycles to let the master get ready
    
    tcp_arg = "{0}:{1}".format(hostname,port)
    worker_dirs = []
    for i in range(num_workers):
        new_worker_dir = os.path.join(worker_root,"worker_{0}".format(i))
        if os.path.exists(new_worker_dir) and reuse_workers is None:
            try:
                shutil.rmtree(new_worker_dir, onerror=_remove_readonly)#, onerror=del_rw)
            except Exception as e:
                raise Exception("unable to remove existing worker dir:" + \
                                "{0}\n{1}".format(new_worker_dir,str(e)))
            try:
                shutil.copytree(main_dir,new_worker_dir)
            except Exception as e:
                raise Exception("unable to copy files from worker dir: " + \
                                "{0} to new worker dir: {1}\n{2}".format(main_dir,new_worker_dir,str(e)))
        elif os.path.exists(new_worker_dir) and reuse_workers is True:
            try:
                shutil.copyfile(pst, os.path.join(new_worker_dir, pst))
            except Exception as e:
                raise Exception("unable to copy *.pst from main worker: " + \
                                "{0} to new worker dir: {1}\n{2}".format(main_dir,new_worker_dir,str(e)))
        else:
            try:
                shutil.copytree(main_dir,new_worker_dir)
            except Exception as e:
                raise Exception("unable to copy files from worker dir: " + \
                                "{0} to new worker dir: {1}\n{2}".format(main_dir,new_worker_dir,str(e)))
        if copy_files is not None and reuse_workers is True:
            try:
                for f in copy_files:
                    shutil.copyfile(f, os.path.join(new_worker_dir, f))
            except Exception as e:
                raise Exception("unable to copy *.pst from main worker: " + \
                                "{0} to new worker dir: {1}\n{2}".format(main_dir,new_worker_dir,str(e)))
        cwd = new_worker_dir
        os.chdir(cwd)
        os.system("start cmd /k beopest64 {0} /h {1}".format(pst, tcp_arg))


# TODO: copy pst / option to use an existing worker
def execute_workers(
            worker_rep, pst, host, num_workers=None,
            start_id=None, worker_root='..', port=4005, reuse_workers=None, copy_files=None):
    """[summary]

    Args:
        worker_rep ([type]): [description]
        pst ([type]): [description]
        host ([type]): [description]
        num_workers ([type], optional): [description]. Defaults to None.
        start_id ([type], optional): [description]. Defaults to None.
        worker_root (str, optional): [description]. Defaults to '..'.
        port (int, optional): [description]. Defaults to 4005.

    Raises:
        Exception: [description]
        Exception: [description]
        Exception: [description]
        Exception: [description]
    """

    if not os.path.isdir(worker_rep):
        raise Exception("master dir '{0}' not found".format(worker_rep))
    if not os.path.isdir(worker_root):
        raise Exception("worker root dir not found")
    if num_workers is None:
        num_workers = mp.cpu_count()
    else:
        num_workers = int(num_workers)
    if start_id is None:
        start_id = 0
    else:
        start_id = start_id

    hostname = host
    base_dir = os.getcwd()
    port = int(port)
    cwd = os.chdir(worker_rep)
    tcp_arg = "{0}:{1}".format(hostname,port)

    for i in range(start_id, num_workers + start_id):
        new_worker_dir = os.path.join(worker_root,"worker_{0}".format(i))
        if os.path.exists(new_worker_dir) and reuse_workers is None:
            try:
                shutil.rmtree(new_worker_dir, onerror=_remove_readonly)#, onerror=del_rw)
            except Exception as e:
                raise Exception("unable to remove existing worker dir:" + \
                                "{0}\n{1}".format(new_worker_dir,str(e)))
            try:
                shutil.copytree(worker_rep,new_worker_dir)
            except Exception as e:
                raise Exception("unable to copy files from worker dir: " + \
                                "{0} to new worker dir: {1}\n{2}".format(worker_rep,new_worker_dir,str(e)))
        elif os.path.exists(new_worker_dir) and reuse_workers is True:
            try:
                shutil.copyfile(pst, os.path.join(new_worker_dir, pst))
            except Exception as e:
                raise Exception("unable to copy *.pst from main worker: " + \
                                "{0} to new worker dir: {1}\n{2}".format(worker_rep,new_worker_dir,str(e)))
        else:
            try:
                shutil.copytree(worker_rep,new_worker_dir)
            except Exception as e:
                raise Exception("unable to copy files from worker dir: " + \
                                "{0} to new worker dir: {1}\n{2}".format(worker_rep,new_worker_dir,str(e)))
        if copy_files is not None and reuse_workers is True:
            try:
                for f in copy_files:
                    shutil.copyfile(f, os.path.join(new_worker_dir, f))
            except Exception as e:
                raise Exception("unable to copy *.pst from main worker: " + \
                                "{0} to new worker dir: {1}\n{2}".format(worker_rep, new_worker_dir,str(e)))

        cwd = new_worker_dir
        os.chdir(cwd)
        os.system("start cmd /k beopest64 {0} /h {1}".format(pst, tcp_arg))


