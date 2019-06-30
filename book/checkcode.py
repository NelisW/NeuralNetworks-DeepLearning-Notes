

import pyradi.ryfiles as ryfiles

filenames = ryfiles.listFiles('.','*.tex',recurse=0)

check = False
for filename in filenames:
    if filename[0]=='p':
        print(filename)
        
        with open(filename) as fin:
            lines = fin.readlines()
            for icnt,line in enumerate(lines):
                if 'begin{lstlisting}' in line:
                    check = True
                if 'end{lstlisting}' in line:
                    check = False

                if check:
                    if '``' in line:
                        print(f'{icnt+1}:  {line}',end='')
                

