import os
import shutil
import glob
from datetime import datetime
import pandas as pd
from pyemu.pst.pst_utils import SFMT,IFMT,FFMT
import numpy as np


wd = os.getcwd()
os.chdir(wd)

def create_riv_par(wd, chns, chg_type=None, rivcd=None, rivbot=None, val=None):
    os.chdir(wd)
    if rivcd is None:
        rivcd =  ['rivcd_{}'.format(x) for x in chns]
    if rivbot is None:
        rivbot =  ['rivbot_{}'.format(x) for x in chns]
    if chg_type is None:
        chg_type = 'unfchg'
    if val is None:
        val = 0.001
    riv_f = rivcd + rivbot
    df = pd.DataFrame()
    df['parnme'] = riv_f
    df['chg_type'] = chg_type
    df['val'] = val
    df.index = df.parnme
    with open('mf_riv.par', 'w') as f:
        f.write("# modflow_par file.\n")
        f.write("NAME   CHG_TYPE    VAL\n")
        f.write(
            df.loc[:, ["parnme", "chg_type", "val"]].to_string(
                                                        col_space=0,
                                                        formatters=[SFMT, SFMT, SFMT],
                                                        index=False,
                                                        header=False,
                                                        justify="left")
            )
    print("'mf_riv.par' file has been exported to the MODFLOW working directory!")
    return df


def read_modflow_par(wd):
    os.chdir(wd)
    # read mf_riv_par.par
    riv_pars = pd.read_csv('mf_riv.par', sep=r'\s+', comment='#', index_col=0)
    # get parameter types and channel number
    riv_pars['par_type'] = [x.split('_')[0] for x in riv_pars.index]
    riv_pars['chn_no'] = [x.split('_')[1] for x in riv_pars.index]
    return riv_pars


def riv_par(wd):
    """change river parameters in *.riv file (river package).

    Args:
        - wd (`str`): the path and name of the existing output file
    Reqs:
        - 'modflow.par'
    Opts:
        - 'riv_package.org'
    Vars:
        - pctchg: provides relative percentage changes
        - absval: provides absolute values
        - unfchg: provides uniform changes
    """

    os.chdir(wd)
    riv_files = [f for f in glob.glob(wd + "/*.riv")]
    if len(riv_files) == 1:
        riv_f = os.path.basename(riv_files[0])
        # duplicate original riv file
        if not os.path.exists('riv_package.org'):
            shutil.copy(riv_f, 'riv_package.org')
            print('The original river package "{}" has been backed up...'.format(riv_f))
        else:
            print('The "riv_package.org" file already exists...')

        with open('riv_package.org') as f:
            line1 = f.readline()
            line2 = f.readline()
            line3 = f.readline()

        # read riv pacakge
        df_riv = pd.read_csv('riv_package.org', sep=r'\s+', skiprows=3, header=None)

        # read mf_riv_par.par
        riv_pars = read_modflow_par(wd)

        # Select rows based on channel number
        for i in range(len(riv_pars)):
            if riv_pars.iloc[i, 2] == 'rivcd':
                subdf = df_riv[df_riv.iloc[:, -1] == riv_pars.iloc[i, 3]]
                if riv_pars.iloc[i, 0] == 'pctchg':
                    new_rivcd = subdf.iloc[:, 4] + (subdf.iloc[:, 4] * float(riv_pars.iloc[i, 1]) / 100)
                elif riv_pars.iloc[i, 0] == 'unfchg':
                    new_rivcd = subdf.iloc[:, 4] + float(riv_pars.iloc[i, 1])
                else:
                    subdf.iloc[:, 4] = float(riv_pars.iloc[i, 1])
                    new_rivcd = subdf.iloc[:, 4]
                count = 0
                for j in range(len(df_riv)):
                    if df_riv.iloc[j, -1] == riv_pars.iloc[i, 3]:
                        df_riv.iloc[j, 4] = new_rivcd.iloc[count]
                        count += 1
            elif riv_pars.iloc[i, 2] == 'rivbot':
                subdf = df_riv[df_riv.iloc[:, -1] == riv_pars.iloc[i, 3]]
                if riv_pars.iloc[i, 0] == 'pctchg':
                    new_rivbot = subdf.iloc[:, 5] + (subdf.iloc[:, 4] * float(riv_pars.iloc[i, 1]) / 100)
                elif riv_pars.iloc[i, 0] == 'unfchg':
                    new_rivbot = subdf.iloc[:, 5] + float(riv_pars.iloc[i, 1])
                else:
                    subdf.iloc[:, 5] = float(riv_pars.iloc[i, 1])
                    new_rivbot = subdf.iloc[:, 5]
                count = 0
                for j in range(len(df_riv)):
                    if df_riv.iloc[j, -1] == riv_pars.iloc[i, 3]:
                        df_riv.iloc[j, 5] = new_rivbot.iloc[count]
                        count += 1

        df_riv.iloc[:, 4] = df_riv.iloc[:, 4].map(lambda x: '{:.10e}'.format(x))
        df_riv.iloc[:, 3] = df_riv.iloc[:, 3].map(lambda x: '{:.10e}'.format(x))
        df_riv.iloc[:, 5] = df_riv.iloc[:, 5].map(lambda x: '{:.10e}'.format(x))

        # ------------ Export Data to file -------------- #
        version = "version 1.2."
        time = datetime.now().strftime('- %m/%d/%y %H:%M:%S -')

        with open(riv_f, 'w') as f:
            f.write("# RIV: River package file is parameterized. " + version + time + "\n")
            f.write(line1)
            f.write(line2)
            f.write(line3)
            df_riv.to_csv(
                        f, sep='\t',
                        header=False,
                        index=False,
                        line_terminator='\n',
                        encoding='utf-8'
                        )
        print(os.path.basename(riv_f) + " file is overwritten successfully!")

    elif len(riv_files) > 1:
        print(
                "You have more than one River Package file!("+str(len(riv_files))+")"+"\n"
                + str(riv_files)+"\n"
                + "Solution: Keep only one file!")
    else:
        print("File Not Found! - We couldn't find your *.riv file.")

def riv_par_more_detail(wd):
    """change river parameters in *.riv file (river package).

    Args:
        - wd (`str`): the path and name of the existing output file
    Reqs:
        - 'modflow.par'
    Opts:
        - 'riv_package.org'
    Vars:
        - pctchg: provides relative percentage changes
        - absval: provides absolute values
        - unfchg: provides uniform changes
    """

    os.chdir(wd)
    riv_files = [f for f in glob.glob(wd + "/*.riv")]
    if len(riv_files) == 1:
        riv_f = os.path.basename(riv_files[0])
        # duplicate original riv file
        if not os.path.exists('riv_package.org'):
            shutil.copy(riv_f, 'riv_package.org')
            print('The original river package "{}" has been backed up...'.format(riv_f))
        else:
            print('The "riv_package.org" file already exists...')

        with open('riv_package.org') as f:
            line1 = f.readline()
            line2 = f.readline()
            line3 = f.readline()

        # read riv pacakge
        df_riv = pd.read_csv('riv_package.org', sep=r'\s+', skiprows=3, header=None)

        # read mf_riv_par.par
        riv_pars = read_modflow_par(wd)

        ''' this block used for whole change in river parameter
        # Change river conductance
        if riv_pars.loc['riv_cond', 'CHG_TYPE'].lower() == 'pctchg':
            riv_cond_v = riv_pars.loc['riv_cond', 'VAL']
            df_riv.iloc[:, 4] = df_riv.iloc[:, 4] + (df_riv.iloc[:, 4] * riv_cond_v / 100)
        else:
            riv_cond_v = riv_pars.loc['riv_cond', 'VAL']
            df_riv.iloc[:, 4] = riv_cond_v

        # Change river bottom elevation
        riv_bot_v = riv_pars.loc['riv_bot', 'VAL']
        if riv_pars.loc['riv_bot', 'CHG_TYPE'].lower() == 'pctchg':
            df_riv.iloc[:, 5] = df_riv.iloc[:, 5] + (df_riv.iloc[:, 5] * riv_bot_v / 100)
        elif riv_pars.loc['riv_bot', 'CHG_TYPE'].lower() == 'absval':
            df_riv.iloc[:, 5] = riv_bot_v
        else:
            df_riv.iloc[:, 5] = df_riv.iloc[:, 5] + riv_bot_v

        df_riv.iloc[:, 4] = df_riv.iloc[:, 4].map(lambda x: '{:.10e}'.format(x))
        df_riv.iloc[:, 3] = df_riv.iloc[:, 3].map(lambda x: '{:.10e}'.format(x))
        df_riv.iloc[:, 5] = df_riv.iloc[:, 5].map(lambda x: '{:.10e}'.format(x))
        '''
        # Select rows based on channel number
        for i in range(len(riv_pars)):
            if riv_pars.iloc[i, 2] == 'rivcd':
                if riv_pars.iloc[i, 3][0] != 'g':
                    subdf = df_riv[df_riv.iloc[:, -3] == int(riv_pars.iloc[i, 3])]
                else:
                    subdf = df_riv[df_riv.iloc[:, -1] == riv_pars.iloc[i, 3]]
                if riv_pars.iloc[i, 0] == 'pctchg':
                    new_rivcd = subdf.iloc[:, 4] + (subdf.iloc[:, 4] * float(riv_pars.iloc[i, 1]) / 100)
                elif riv_pars.iloc[i, 0] == 'unfchg':
                    new_rivcd = subdf.iloc[:, 4] + float(riv_pars.iloc[i, 1])
                else:
                    subdf.iloc[:, 4] = float(riv_pars.iloc[i, 1])
                    new_rivcd = subdf.iloc[:, 4]
                if riv_pars.iloc[i, 3][0] != 'g':         
                    count = 0
                    for j in range(len(df_riv)):
                        if df_riv.iloc[j, -3] == int(riv_pars.iloc[i, 3]):
                            df_riv.iloc[j, 4] = new_rivcd.iloc[count]
                            count += 1
                else:
                    count = 0
                    for j in range(len(df_riv)):
                        if df_riv.iloc[j, -1] == riv_pars.iloc[i, 3]:
                            df_riv.iloc[j, 4] = new_rivcd.iloc[count]
                            count += 1                
            elif riv_pars.iloc[i, 2] == 'rivbot':
                if riv_pars.iloc[i, 3][0] != 'g':
                    subdf = df_riv[df_riv.iloc[:, -3] == int(riv_pars.iloc[i, 3])]
                else:
                    subdf = df_riv[df_riv.iloc[:, -1] == riv_pars.iloc[i, 3]]
                if riv_pars.iloc[i, 0] == 'pctchg':
                    new_rivbot = subdf.iloc[:, 5] + (subdf.iloc[:, 4] * float(riv_pars.iloc[i, 1]) / 100)
                elif riv_pars.iloc[i, 0] == 'unfchg':
                    new_rivbot = subdf.iloc[:, 5] + float(riv_pars.iloc[i, 1])
                else:
                    subdf.iloc[:, 5] = float(riv_pars.iloc[i, 1])
                    new_rivbot = subdf.iloc[:, 5]
                if riv_pars.iloc[i, 3][0] != 'g':
                    count = 0
                    for j in range(len(df_riv)):
                        if df_riv.iloc[j, -3] == int(riv_pars.iloc[i, 3]):
                            df_riv.iloc[j, 5] = new_rivbot.iloc[count]
                            count += 1
                else:
                    count = 0
                    for j in range(len(df_riv)):
                        if df_riv.iloc[j, -1] == riv_pars.iloc[i, 3]:
                            df_riv.iloc[j, 5] = new_rivbot.iloc[count]
                            count += 1
                            
        df_riv.iloc[:, 4] = df_riv.iloc[:, 4].map(lambda x: '{:.10e}'.format(x))
        df_riv.iloc[:, 3] = df_riv.iloc[:, 3].map(lambda x: '{:.10e}'.format(x))
        df_riv.iloc[:, 5] = df_riv.iloc[:, 5].map(lambda x: '{:.10e}'.format(x))
        df_riv.iloc[:, -2] = df_riv.iloc[:, -2].map(lambda x: '{:.10e}'.format(x))

        # ------------ Export Data to file -------------- #
        version = "version 1.2."
        time = datetime.now().strftime('- %m/%d/%y %H:%M:%S -')

        with open(riv_f, 'w') as f:
            f.write("# RIV: River package file is parameterized. " + version + time + "\n")
            f.write(line1)
            f.write(line2)
            f.write(line3)
            df_riv.to_csv(
                        f, sep='\t',
                        header=False,
                        index=False,
                        line_terminator='\n',
                        encoding='utf-8'
                        )
        print(os.path.basename(riv_f) + " file is overwritten successfully!")

    elif len(riv_files) > 1:
        print(
                "You have more than one River Package file!("+str(len(riv_files))+")"+"\n"
                + str(riv_files)+"\n"
                + "Solution: Keep only one file!")
    else:
        print("File Not Found! - We couldn't find your *.riv file.")


def parm_to_tpl_file(parm_infile=None, parm_db=None, tpl_file=None):
    """write a template file for a APEX parameter file (PARM1501.DAT)

    Args:
        parm_infile (`str`, optional): path or name of the existing parm file. Defaults to None.
        parm_db (`str`, optional): DB for APEX parameters (apex.parm.xlsx). Defaults to None.
        tpl_file (`str`, optional): template file to write. If None, use
        `parm_infile` + ".tpl". Defaults to None.

    Returns:
        **pandas.DataFrame**: a dataFrame with template file information
    """
    if parm_infile is None:
        parm_infile = "PARM1501.DAT"
    if parm_db is None:
        parm_db = "apex.parm.xlsx"
    if tpl_file is None:
        tpl_file = parm_infile + ".tpl"
    parm_df = pd.read_excel(parm_db, usecols=[0, 7], index_col=0, comment="#", engine="openpyxl")
    parm_df = parm_df.loc[parm_df.index.notnull()]
    parm_df['temp_idx'] = parm_df.index
    parm_df['idx'] = 0
    for i in range(len(parm_df)):
        parm_df.iloc[i, 2] = int(parm_df.iloc[i, 1][1:]) 
    parm_df = parm_df.sort_values(by=['idx'])    

    parm_sel = parm_df[parm_df['flag'] == 'y'].index.tolist()
    with open(parm_infile, 'r') as f:
        content = f.readlines()
    upper_pars = [x.rstrip() for x in content[:35]] 
    core_pars = [x.strip() for x in content[35:46]]
    lower_pars = [x.rstrip() for x in content[46:]]

    # core_lst = [i for c in core_pars for i in c.split()]
    core_lst = []
    for i in core_pars:
        for j in i.split():
            core_lst.append(j)
    parm_df['nam'] = parm_df.index.tolist()
    parm_df['value'] = core_lst
    parm_df['tpl'] = np.where(
        parm_df['flag'] == 'n',
        parm_df['value'].apply(lambda x:"{0:>7s}".format(x)),
        parm_df.nam.apply(lambda x:"~{0:5s}~".format(x))
        )
    parm_lst = parm_df.tpl.tolist()
    parm_arr = np.reshape(parm_lst, (11, 10))
    parm_arr_df = pd.DataFrame(parm_arr)
    parm_arr_df.iloc[:, 0] = parm_arr_df.iloc[:, 0].apply(lambda x:"{0:>8s}".format(x))
    tpl_file = parm_infile + ".tpl"

    with open(tpl_file, 'w') as f:
        f.write("ptf ~\n")
        for row in upper_pars:
            f.write(row + '\n')
        f.write(parm_arr_df.to_string(
                                    col_space=0,
                                    # formatters=fmt,
                                    index=False,
                                    header=False,
                                    justify="right"
                                    ))
        f.write('\n')
        for row in lower_pars:
            f.write(row + '\n')
    return parm_arr_df


def export_pardb_pest(par):
    parm_df =  pd.read_excel(
        "apex.parm.xlsx", index_col=0, usecols=[x for x in range(8)],
        comment='#', engine='openpyxl', na_values=[-999, '']
        )
    parm_sel = parm_df[parm_df['flag'] == 'y']
    par_draft = pd.concat([par, parm_sel], axis=1)
    # filtering
    
    par_draft['parval1'] = np.where((par_draft.default_initial.isna()), par_draft.parval1, par_draft.default_initial)
    par_draft['parval1'] = np.where((par_draft.cali_initial.isna()), par_draft.parval1, par_draft.cali_initial)
    par_draft['parval1'] = np.where((par_draft.parval1 == 0), 0.00001, par_draft.parval1)
    par_draft['parlbnd'] = np.where((par_draft.cali_lower.isna()), par_draft.parlbnd, par_draft.cali_lower)
    par_draft['parlbnd'] = np.where((par_draft.absolute_lower.isna()), par_draft.parlbnd, par_draft.absolute_lower)
    par_draft['parlbnd'] = np.where((par_draft.parlbnd == 0), 0.00001, par_draft.parlbnd)
    par_draft['parubnd'] = np.where((par_draft.cali_upper.isna()), par_draft.parubnd, par_draft.cali_upper)
    par_draft['parubnd'] = np.where((par_draft.absolute_upper.isna()), par_draft.parubnd, par_draft.absolute_upper)
    par_f =  par_draft.dropna(axis=1)
    return par_f


def update_hk_pars(par):
    hk_df = pd.read_csv(
                        'MODFLOW/hk0pp.dat',
                        sep='\s+',
                        usecols=[4],
                        names=['hk_temp']
                        )
    hk_df.index = [f"hk{i:03d}" for i in range(len(hk_df))]
    par_draft = pd.concat([par, hk_df], axis=1)
    par_draft['parval1'] = np.where((par_draft.hk_temp.isna()), par_draft.parval1, par_draft.hk_temp)
    par_f =  par_draft.dropna(axis=1)
    return par_f

def update_sy_pars(par):
    sy_df = pd.read_csv(
                        'MODFLOW\sy0pp.dat',
                        sep='\s+',
                        usecols=[4],
                        names=['sy_temp']
                        )
    sy_df.index = [f"sy{i:03d}" for i in range(len(sy_df))]
    par_draft = pd.concat([par, sy_df], axis=1)
    par_draft['parval1'] = np.where((par_draft.sy_temp.isna()), par_draft.parval1, par_draft.sy_temp)
    par_f =  par_draft.dropna(axis=1)
    return par_f

def riv_par_to_template_file(riv_par_file, tpl_file=None):
    """write a template file for a SWAT parameter value file (model.in).

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
        tpl_file = riv_par_file + ".tpl"
    mf_par_df = pd.read_csv(
                        riv_par_file,
                        delim_whitespace=True,
                        header=None, skiprows=2,
                        names=["parnme", "chg_type", "parval1"])
    mf_par_df.index = mf_par_df.parnme
    mf_par_df.loc[:, "tpl"] = mf_par_df.parnme.apply(lambda x: " ~   {0:15s}   ~".format(x))
    with open(tpl_file, 'w') as f:
        f.write("ptf ~\n# modflow_par template file.\n")
        f.write("NAME   CHG_TYPE    VAL\n")
        f.write(mf_par_df.loc[:, ["parnme", "chg_type", "tpl"]].to_string(
                                                        col_space=0,
                                                        formatters=[SFMT, SFMT, SFMT],
                                                        index=False,
                                                        header=False,
                                                        justify="left"))
    return mf_par_df

def read_cropcom_par():
    crop_pars = pd.read_csv('CROPCOM.DAT', sep='\t', index_col=0)
    print(crop_pars)


def cropcom_col_spaces():
    S5s = lambda x: "{0:>5s}".format(str(x))
    S4s = lambda x: "{0:>4s}".format(str(x))
    F8s2 = lambda x: "{0:>7.2f}".format(float(x))
    F8s3 = lambda x: "{0:>7.3f}".format(float(x))
    F8s4 = lambda x: "{0:>7.4f}".format(float(x))
    I8s = lambda x: "{0:>7d}".format(int(x))
    return S5s, S4s, F8s2, F8s3, F8s4, I8s
    

def crop_par_df():
    # create cropcom template file
    crop_pars_ = pd.read_csv(
            'CROPCOM.DAT',
            skiprows=2, 
            header=None,
            sep=r'\s+',
            usecols=[i for i in range(58)]
            )
    # fix col names (adding ...)
    with open('CROPCOM.DAT', "r") as f:
        for line in f.readlines():
            if line.split()[0] == 'CROP':
                l = ['idx']+line.split()
            if line.split()[0] == '#':
                l2 = line.split()
    crop_pars_.loc[-2] = l
    crop_pars_.loc[-1] = l2
    crop_pars_.index = crop_pars_.index + 2  # shifting index
    crop_pars_.sort_index(inplace=True)
    crop_pars_.columns = l2

    return crop_pars_

def crop_pars_tpl():
    S5s, S4s, F8s2, F8s3, F8s4, I8s = cropcom_col_spaces()
    crop_pars = crop_par_df()
    formatters = ([S5s, S4s]+([F8s2]*10) + [I8s] + [F8s4]+([F8s2]*5)+([F8s4]*3)+([F8s2]*6)+([F8s4]*9)+([F8s3]*3)+ [I8s]+([F8s2]*17))
    for i in range(0, len(formatters)):
        crop_pars.iloc[2:, i] = crop_pars.iloc[2:, i].map(formatters[i])
    with open('crop.parm', "r") as f:
        pars = []
        crops = []
        for line in f.readlines():
            if line.strip() != "":
                pars.append(line.upper().strip().split("_")[0])
                crops.append(line.upper().strip().split("_")[1])
    count = 1
    for p, c in zip(pars, crops):
        c = "{0:>4s}".format(str(c))
        pc_nam =  "cp{0:02d}".format(count)
        crop_pars.loc[crop_pars['NAME'] == c, p] = crop_pars.loc[crop_pars['NAME'] == c, p].apply(
                                                                                            lambda x: "~{0:>5s}~".format(pc_nam))
        count += 1
    with open('CROPCOM.DAT.tpl', 'w') as f:
        f.write("ptf ~\n")
        f.write(crop_pars.to_string(
                                    col_space=0,
                                    index=False,
                                    header=False,
                                    justify="right")
            )
    print(crop_pars)



if __name__ == '__main__':
    wd = "D:\\Projects\\RegionalCalibration\\Autocalibration\\ani_apexmf_cal_v03"
    os.chdir(wd)
    read_cropcom_par()
