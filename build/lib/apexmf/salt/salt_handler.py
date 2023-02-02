import os
import pandas as pd
from utils import ObjFns
import matplotlib.pyplot as plt
# import ObjFns


class SaltAnalysis(object):

    def __init__(self, wd):
        os.chdir(wd)

    def read_salt_sim_cha(self, sub_id, sim_start, cal_start, cal_end=None):
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
        salt_dff = salt_df.loc[sub_id]
        salt_dff.index = pd.date_range(sim_start, periods=len(salt_dff))
        if cal_end is None:
            salt_dff = salt_dff[cal_start:]
        else:
            salt_dff = salt_dff[cal_start:cal_end]
        return salt_dff

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


class SaltViz(object):

    def __init__(self, df):
        self.df = df

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

