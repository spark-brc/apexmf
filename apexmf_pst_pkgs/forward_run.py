import os
from datetime import datetime
import pyemu
from apex_pst_utils import extract_month_str


wd = os.getcwd()
os.chdir(wd)
print(wd)

# file path
rch_file = 'SITE75.RCH'
# reach numbers that are used for calibration
subs = [57, 72, 75]

time = datetime.now().strftime('[%m/%d/%y %H:%M:%S]')
print('\n' + 30*'+ ')
print(time + ' |  modifying parameters...')
print(30*'+ ' + '\n')

time = datetime.now().strftime('[%m/%d/%y %H:%M:%S]')
print('\n' + 30*'+ ')
print(time + ' |  running model...')
print(30*'+ ' + '\n')
# pyemu.os_utils.run('SWAT-MODFLOW3.exe >_s+m.stdout', cwd='.')
pyemu.os_utils.run('APEX-MODFLOW_ani.exe', cwd='.')
time = datetime.now().strftime('[%m/%d/%y %H:%M:%S]')

print('\n' + 35*'+ ')
print(time + ' | simulation successfully completed | extracting simulated values...')
print(35*'+ ' + '\n')
extract_month_str(rch_file, subs, '1/1/2000', '1/1/2003', '12/31/2010')

# extract_watertable_sim([5699, 5832], '1/1/1980', '12/31/2005')


