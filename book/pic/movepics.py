import pyradi.ryfiles as ryfiles
import os

for ch in [2]:

    filenames = ryfiles.listFiles('.',f'p05c{ch:02d}*')
    for filename in filenames:
        newname = filename
        newname = newname.replace(f'p05c{ch:02d}',f'p05c{ch+2:02d}')
        cmnd = f'git mv {filename} {newname}'

        print(cmnd)

        os.system(cmnd)

