# from apexmf_pkgs.apexmf_pst_par import riv_par
import os
from datetime import datetime
import pyemu
import subprocess
from apexmf_pst_utils import extract_month_str, extract_month_sed, extract_depth_to_water
from apexmf_pst_par import riv_par

wd = os.getcwd()
os.chdir(wd)
print(wd)

mf_wd = wd + "\MODFLOW"

# file path
rch_file = 'SITE75.RCH'
# reach numbers that are used for calibration
subs = [12, 57, 72, 75]
grid_ids = [5895, 6273]


time = datetime.now().strftime('[%m/%d/%y %H:%M:%S]')
print('\n' + 30*'+ ')
print(time + ' |  modifying parameters...')
print(30*'+ ' + '\n')

riv_par(mf_wd)

os.chdir(mf_wd)
# modify MODFLOW parameters
time = datetime.now().strftime('[%m/%d/%y %H:%M:%S]')
print(time + ' |  modifying MODFLOW HK, VHK, and SY parameters...')
data_fac = ['hk0pp.dat', 'sy0pp.dat']
for i in data_fac:
    outfile = i + '.ref'
    pyemu.utils.geostats.fac2real(i, factors_file=i+'.fac', out_file=outfile)
#     # Create vertical k file
#     # if i[:2] == 'hk':
#     #     vk = np.loadtxt(outfile)
#     #     np.savetxt('v' + outfile, vk/10, fmt='%.12e', delimiter='\t')

# Run model
os.chdir(wd)
time = datetime.now().strftime('[%m/%d/%y %H:%M:%S]')
print('\n' + 30*'+ ')
print(time + ' |  running model...')
print(30*'+ ' + '\n')
# pyemu.os_utils.run('SWAT-MODFLOW3.exe >_s+m.stdout', cwd='.')
p = subprocess.Popen('apexmf.exe' , cwd = '.')
p.wait()
# pyemu.os_utils.run('APEX-MODFLOW.exe', cwd=wd)

time = datetime.now().strftime('[%m/%d/%y %H:%M:%S]')


print('\n' + 35*'+ ')
print(time + ' | simulation successfully completed | extracting simulated values...')
print(35*'+ ' + '\n')
extract_month_str(rch_file, subs, '1/1/1987', '1/1/1992', '12/31/2011')
extract_month_sed(rch_file, subs, '1/1/1987', '1/1/1992', '12/31/2011')
extract_depth_to_water(grid_ids, '1/1/1987', '10/30/2003')
print(time + ' | Complete ...')


