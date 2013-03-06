#! coding:utf-8

import os
from pylearn2.utils.shell import run_shell_command

# I personally just write python scripts to emit lots of yaml ﬁles and generate 
# commands like: 
# jobdispatch /data/lisatmp/username/my_super_script.py 
# /data/lisatmp/ift6266h13/${USER}/conﬁg_ﬁle”{{1,2,3}}”.yaml

expdir = '/data/lisatmp/ift6266h13/berniergtmp/ift6266/experiment1'
namelist = os.listdir(expdir)
names = [expdir + '/' + name for name in namelist if name[-5:] == '.yaml']
print names
command = 'jobdispatch --env=THEANO_FLAGS=device=cpu,ﬂoatX=ﬂoat32,force_device=True' + \
        ' --whitespace python /data/lisatmp/ift6266h13/berniergtmp/ift6266/train.py -i '
command += ' "{{'
command += ', '.join(names)
command += '}}" '
output, rc = run_shell_command(command)
print 'output', output
print 'rc', rc
