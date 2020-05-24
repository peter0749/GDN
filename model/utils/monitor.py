import psutil
import gpustat


def write_hwstat(root, outname='hwstat.out'):
    cpu_usage = psutil.cpu_percent()
    mem_usage = psutil.virtual_memory()._asdict()['percent']
    with open(root+'/'+'hwstat.out', 'w') as fp:
        fp.write('CPU: %.1f%% MEM: %.1f%%\n' % (cpu_usage, mem_usage))
        for line in gpustat.new_query():
            fp.write(str(line)+'\n')
