# from apexmf.salt.salt_handler import SaltAnalysis

import os
import pandas as pd
import numpy as np
from apexmf.utils import ObjFns
import matplotlib.pyplot as plt
# import ObjFns


class SaltAnalysis(object):

    def __init__(self, wd):
        os.chdir(wd)


    def load_salt_outlet_result(self):
        if not os.path.exists('SALINITY/salt.output.outlet'):
            raise Exception("'salt.output.outlet' file not found")
        salt_df = pd.read_csv(
                            "SALINITY/salt.output.outlet",
                            delim_whitespace=True,
                            skiprows=4,
                            header=0,
                            # index_col=0,
                            )
        salt_df = salt_df.iloc[:, 5:] # only cols we need
        return salt_df
    
    def get_tot_salt_outlet(self, df, sim_start, cal_start, cal_end=None):
        
        df.index = pd.date_range(sim_start, periods=len(df))
        if cal_end is None:
            df = df[cal_start:]
        else:
            df = df[cal_start:cal_end]
        df = df.iloc[:, :8]
        df['tot_salt_loads'] = df.iloc[:, :].sum(axis=1)
        df.drop(df.iloc[:, :-1], inplace=True, axis=1)
        return df

    def load_salt_cha_result(self):
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
        return salt_df
    
    def load_salt_budget_sub(self):
        if not os.path.exists('SALINITY/salt.output.budget_subarea'):
            raise Exception("'salt.output.budget_subarea' file not found")
        salt_df = pd.read_csv(
                            "SALINITY/salt.output.budget_subarea",
                            delim_whitespace=True,
                            skiprows=3,
                            header=0,
                            index_col=0,
                            )
        # salt_df = salt_df.iloc[:, 5:] # only cols we need
        return salt_df

    def load_salt_budget_watershed(self):
        if not os.path.exists('SALINITY/salt.output.budget_watershed'):
            raise Exception("'salt.output.budget_watershed' file not found")
        salt_df = pd.read_csv(
                            "SALINITY/salt.output.budget_watershed",
                            delim_whitespace=True,
                            skiprows=3,
                            header=0,
                            index_col=0,
                            )
        # salt_df = salt_df.iloc[:, 5:] # only cols we need
        return salt_df      

    def read_salt_sim_cha(self, df, sub_id, sim_start, cal_start, cal_end=None):
        salt_df = df.loc[sub_id]
        salt_df.index = pd.date_range(sim_start, periods=len(salt_df))
        if cal_end is None:
            salt_df = salt_df[cal_start:]
        else:
            salt_df = salt_df[cal_start:cal_end]
        return salt_df


    def read_salt_obd_cha(self, sub_id, time_step=None):
        if time_step is None:
            time_step = 'day'
        obd_file = f"salt_{sub_id:03d}_{time_step}.obd"
        if not os.path.exists(obd_file):
            raise Exception(f"'{obd_file}' file not found")
        obd_df = pd.read_csv(
                            obd_file, sep='\t', index_col=0, 
                            parse_dates=True, na_values=[-999, ""])
        return obd_df
    
    def sim_obd_df(self, sim_df, sim_col, obd_df, obd_col):
        df = pd.concat([sim_df.loc[:, sim_col], obd_df.loc[:, obd_col]], axis=1)
        return df

    # def get_stats(self):
    #     df_stat =self.sim_obd_df
    #     sim = df_stat.iloc[:, 0].to_numpy()
    #     obd = df_stat.iloc[:, 1].to_numpy()
        
    #     # df_nse = evaluator(nse, sim, obd)
    #     # df_rmse = evaluator(rmse, sim, obd)
    #     # df_pibas = evaluator(pbias, sim, obd)
    #     # r_squared = (
    #     #     ((sum((obd - obd.mean())*(sim-sim.mean())))**2)/
    #     #     ((sum((obd - obd.mean())**2)* (sum((sim-sim.mean())**2))))
    #     #     )        
    #     df_nse = ObjFns.nse(sim, obd)
    #     df_rmse = ObjFns.rmse(sim, obd)
    #     df_pibas = ObjFns.pbias(sim, obd)
    #     r_squared = ObjFns.rsq(sim, obd)
    #     return df_nse, df_rmse, df_pibas, r_squared


font = {"family": "calibri", "weight": "normal", "size": 12}
class SaltViz(object):

    def __init__(self, df):
        self.df = df

    def plot_hist(self, ax, ion_nam, label=None):
        
        plt.rc("font", **font)
        ax.hist(
            self.df[ion_nam], 
            bins=np.linspace(self.df[ion_nam].min(), self.df[ion_nam].max(), 20), 
            alpha=0.5, label=label)
        ax.set_ylabel("Density")
        ax.set_xlabel(ion_nam)

        
    def hydro_sim_obd02(self, ax, height=None):
        if height is None:
            height = 3
        # plot
        
        ax.grid(True)
        # cali
        ax.plot(self.df.index, self.df.iloc[:, 0], label='Calibrated', color='k', alpha=0.7)
        # ax.plot(self.df.index, self.df.iloc[:, 1], 'k--', label='Validated' , alpha=0.7)
        ax.scatter(
            self.df.index, self.df.iloc[:, 1], label='Observed',
            color='red',
            # facecolors="None", edgecolors='red',
            lw=1.5,
            alpha=0.4,
            zorder=2,
            )
        # ax.set_ylim(0, 1000)
        ax.tick_params(axis='both', labelsize=12)


    def hydro_sim_obd(self, height=None):
        if height is None:
            height = 3
        # plot
        fig, ax = plt.subplots(figsize=(16, height))

        ax.grid(True)
        # cali
        ax.plot(self.df.index, self.df.iloc[:, 0], label='Calibrated', color='k', alpha=0.7)
        # ax.plot(self.df.index, self.df.iloc[:, 1], 'k--', label='Validated' , alpha=0.7)
        ax.scatter(
            self.df.index, self.df.iloc[:, 1], label='Observed',
            color='red',
            # facecolors="None", edgecolors='red',
            lw=1.5,
            alpha=0.4,
            zorder=2,
            )
        # ax.set_ylim(0, 1000)
        ax.tick_params(axis='both', labelsize=12)
        fig.tight_layout()
        plt.show()

    def reg_line02(self, ax, fmin=None, fmax=None):

        dmin = self.df.min().min()
        dmax = self.df.max().max()

        if fmin is None:
            fmin = dmin
        if fmax is None:
            fmax = dmax

        ax.plot([fmin, fmax], [fmin, fmax], 'k--', alpha=0.2)
        ax.scatter(
            self.df.iloc[:, 0], self.df.iloc[:, 1],
            alpha=0.4,
            # zorder=2,
            )
        ax.tick_params(axis='both', labelsize=12)
        ax.set_ylim(fmin, fmax)
        ax.set_xlim(fmin, fmax)
        ax.set_ylabel("Observed",fontsize=12)
        ax.set_xlabel("Simulated",fontsize=12)


    def reg_line(self, fmin=None, fmax=None):
        fig, ax = plt.subplots(figsize=(4, 3.5))
        dmin = self.df.min().min()
        dmax = self.df.max().max()

        if fmin is None:
            fmin = dmin
        if fmax is None:
            fmax = dmax
        ax.plot([fmin, fmax], [fmin, fmax], 'k--', alpha=0.2)
        ax.scatter(
            self.df.iloc[:, 0], self.df.iloc[:, 1],
            alpha=0.4,
            # zorder=2,
            )
        ax.tick_params(axis='both', labelsize=12)
        ax.set_ylim(fmin, fmax)
        ax.set_xlim(fmin, fmax)
        ax.set_ylabel("Observed",fontsize=12)
        ax.set_xlabel("Simulated",fontsize=12)
        fig.tight_layout()
        # plt.savefig('1to1.jpg', dpi=600, bbox_inches="tight")
        plt.show() 

def get_stats(df_stat):
    df_stat = df_stat.dropna()
    sim = df_stat.iloc[:, 0].to_numpy()
    obd = df_stat.iloc[:, 1].to_numpy()
    df_nse = ObjFns.nse(sim, obd)
    df_rmse = ObjFns.rmse(sim, obd)
    df_pibas = ObjFns.pbias(sim, obd)
    r_squared = ObjFns.rsq(sim, obd)
    return df_nse, df_rmse, df_pibas, r_squared



def get_vars_95ppu(data_list, l95p=2.5, u95p=97.5):
    # getting data of the histogram
    count, bins_count = np.histogram(data_list, bins=1000)
    # finding the PDF of the histogram using count values
    pdf = count / sum(count)
    # using numpy np.cumsum to calculate the CDF
    # We can also find using the PDF values by looping and adding
    cdf = np.cumsum(pdf)
    l95ppu = bins_count[int((l95p)*10)]
    u95ppu = bins_count[int((u95p)*10)]
    m95ppu = bins_count[int((50)*10)]
    return l95ppu, u95ppu, m95ppu

def get_95ppu_df(input_df, l95p=2.5, u95p=97.5):
    df_95ppu = pd.DataFrame(columns=['l95ppu','u95ppu','m95ppu'])
    for i in range(len(input_df.columns)):
        l, u, m = get_vars_95ppu(input_df.iloc[:, i], l95p=l95p, u95p=u95p)
        df_95ppu.loc[i] = [l, u, m]
    return df_95ppu