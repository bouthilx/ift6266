import os
from pylearn2.utils.shell import run_shell_command

# I personally just write python scripts to emit lots of yaml ﬁles and generate 
# commands like: 
# jobdispatch /data/lisatmp/username/my_super_script.py 
# /data/lisatmp/ift6266h13/${USER}/conﬁg_ﬁle”{{1,2,3}}”.yaml

expdir = '/data/lisatmp'
names = os.listdir(expdir)
dirs = [expdir + '/' + name for name in names]
dirs = [d for d in dirs if not os.path.exists(d + '/cluster_info.txt')]
command = 'jobdispatch --torque --
env=THEANO_FLAGS=device=gpu,ﬂoatX=ﬂoat32,force_device=True' + \
        ' --duree=48:00:00 --whitespace ' + \
        '--gpu bash /RQexec/goodfell/galatea/mlp/experiment_6/worker.sh '
command += ' "{{'
command += ', '.join(dirs)
command += '}}" '
output, rc = run_shell_command(command)