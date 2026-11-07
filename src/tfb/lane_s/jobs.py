"""Owned resumable-job receipt and the lane's RAM ceiling."""
from contextlib import contextmanager
from datetime import datetime,timezone,timedelta
import json,os,sys,threading,time
import psutil
from tfb.paths import project_root

@contextmanager
def owned_job(number,checkpoint,eta_hours=3,ram_gib=7):
    root=project_root();path=root/'artifacts/jobs'/f'{number}.json';path.parent.mkdir(parents=True,exist_ok=True)
    start=datetime.now(timezone.utc); stop=threading.Event();process=psutil.Process()
    receipt={'id':number,'owner':'S','pid':os.getpid(),'command':[sys.executable,*sys.argv],'start_utc':start.isoformat(),'eta':(start+timedelta(hours=eta_hours)).isoformat(),'checkpoint_path':str(checkpoint),'status':'RUNNING','ram_limit_gib':ram_gib,'peak_rss_gib':0.}
    lock=threading.Lock()
    def save():
        temp=path.with_suffix('.tmp');temp.write_text(json.dumps(receipt,indent=2)+'\n')
        for attempt in range(10):
            try:os.replace(temp,path);return
            except PermissionError:
                if attempt==9:raise
                time.sleep(.02*(attempt+1))
    def monitor():
        while not stop.wait(2):
            rss=process.memory_info().rss/1024**3
            with lock:
                receipt['peak_rss_gib']=max(receipt['peak_rss_gib'],rss);receipt['updated_utc']=datetime.now(timezone.utc).isoformat()
                if rss>ram_gib:
                    receipt['status']='FAILED_RAM_LIMIT'
                    try:save()
                    finally:os._exit(88)
                try:save()
                except PermissionError:
                    sys.stderr.write('S receipt temporarily locked; RAM monitor remains active\n')
    save();thread=threading.Thread(target=monitor,daemon=True);thread.start()
    try:yield receipt
    except BaseException as error:
        with lock:receipt['status']='FAILED';receipt['error']=repr(error)
        raise
    else:
        with lock:receipt['status']='COMPLETE'
    finally:
        stop.set();thread.join();receipt['end_utc']=datetime.now(timezone.utc).isoformat();save()
