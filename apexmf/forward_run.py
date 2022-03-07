import os
from datetime import datetime
import pandas as pd
from apexmf import apexmf_pst_par, apexmf_utils
from apexmf import apexmf_pst_utils
import pyemu


def forward_run(
                sim_start, cal_start, cal_end,
                cha_file, subs,
                gw_level, grids,
                lai_file, lai_subs,
                riv_parm,  baseflow,
                time_step
                ):
    wd = os.getcwd()
    os.chdir(wd)
    print(wd)

    if riv_parm == 'y':
        time = datetime.now().strftime('[%m/%d/%y %H:%M:%S]')
        print('\n' + 30*'+ ')
        print(time + ' |  updating river parameters...')
        print(30*'+ ' + '\n')
        apexmf_pst_par.riv_par(wd)

    time = datetime.now().strftime('[%m/%d/%y %H:%M:%S]')
    print('\n' + 30*'+ ')
    print(time + ' |  running model...')
    print(30*'+ ' + '\n')
    # pyemu.os_utils.run('APEX-MODFLOW3.exe >_s+m.stdout', cwd='.')
    pyemu.os_utils.run('apexmf', cwd='.')
    time = datetime.now().strftime('[%m/%d/%y %H:%M:%S]')

    print('\n' + 35*'+ ')
    print(time + ' | simulation successfully completed | extracting simulated values...')
    print(35*'+ ' + '\n')


    if cha_file != 'n' and time_step == 'day':
        apexmf_pst_utils.extract_day_stf(subs, sim_start, cal_start, cal_end)
    elif cha_file != 'n' and time_step == 'month':
        apexmf_pst_utils.extract_month_stf(subs, sim_start, cal_start, cal_end)
    if gw_level == 'y':
        print('\n' + 35*'+ ')
        print(time + ' | simulation successfully completed | extracting depth to water values...')
        print(35*'+ ' + '\n')
    elif gw_level      
        apexmf_pst_utils.extract_depth_to_water(grids, sim_start, cal_end)
    if lai_file != 'n' and time_step == 'day':
        sao_df = apexmf_utils.read_sao(lai_file)
        apexmf_pst_utils.extract_day_lai(sao_df, lai_subs, sim_start, cal_start, cal_end)
    if lai_file != 'n' and time_step == 'month':
        sao_df = apexmf_utils.read_sao(lai_file)
        apexmf_pst_utils.extract_day_mon(sao_df, lai_subs, sim_start, cal_start, cal_end)
    if baseflow == 'y':
        print('\n' + 35*'+ ')
        print(time + ' | simulation successfully completed | calculating baseflow ratio...')
        print(35*'+ ' + '\n')
        apexmf_pst_utils.extract_month_baseflow(subs, sim_start, cal_start, cal_end)




    # extract_watertable_sim([5699, 5832], '1/1/1980', '12/31/2005')

if __name__ == '__main__':
    cwd = os.getcwd()
    os.chdir(cwd)
    apexmf_con = pd.read_csv('apexmf.con', sep='\t', names=['names', 'vals'], index_col=0, comment="#")
    wd = apexmf_con.loc['wd', 'vals']
    sim_start = apexmf_con.loc['sim_start','vals']
    cal_start = apexmf_con.loc['cal_start','vals']
    cal_end = apexmf_con.loc['cal_end','vals']
    cha_file = apexmf_con.loc['cha_file','vals']
    if cha_file != 'n':
        subs = apexmf_con.loc['subs','vals'].strip('][').split(', ')
        subs = [int(i) for i in subs]        
    gw_level = apexmf_con.loc['gw_level','vals']
    if gw_level == 'y':
        grids = apexmf_con.loc['grids','vals'].strip('][').split(', ')
        grids = [int(i) for i in grids]
    elif gw_level == 'n':
        grids = apexmf_con.loc['grids','vals']
    lai_file = apexmf_con.loc['lai_file','vals']
    if lai_file != 'n':
        lai_subs = apexmf_con.loc['lai_subs','vals'].strip('][').split(', ')
        lai_subs = [int(i) for i in lai_subs] 
    elif lai_file == 'n':
        lai_subs = apexmf_con.loc['lai_subs','vals']

    riv_parm = apexmf_con.loc['riv_parm','vals']
    baseflow = apexmf_con.loc['baseflow','vals']
    time_step = apexmf_con.loc['time_step','vals']

    forward_run(
                sim_start, cal_start, cal_end,
                cha_file, subs,
                gw_level, grids,
                lai_file, lai_subs,
                riv_parm,  baseflow,
                time_step
                )
