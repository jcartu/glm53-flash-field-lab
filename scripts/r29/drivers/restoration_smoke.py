#!/usr/bin/env python3
"""Real Docker stop-interruption and cleanup proof using CPU-only fixtures."""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid
from unittest.mock import patch

sys.path.insert(0, '/home/josh/omp-workspace/glm53-flash-field-lab/scripts/r26')
import runtime as rt
import run_qualification as coordinator

ROOT = Path(__file__).resolve().parent / 'restoration-smoke'
IMAGE = 'localinferencelab/vllm@sha256:e44e07e615287605f87bd4db916d683e39066e72a1ba94cf4149089c1ec21b49'
LABEL = 'r29-restoration-smoke'
SERVER = '''import http.server,json,signal,time
class Handler(http.server.BaseHTTPRequestHandler):
 def do_GET(self):
  body=json.dumps({'data':[{'id':'cpu-restoration-fixture'}]}).encode()
  self.send_response(200);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
 def log_message(self,*args): pass
def stop(signum,frame):
 print('STOP_ACCEPTED',flush=True);time.sleep(8);raise SystemExit(0)
signal.signal(signal.SIGTERM,stop)
server=http.server.HTTPServer(('0.0.0.0',8080),Handler)
print('FIXTURE_READY',flush=True)
server.serve_forever(poll_interval=.1)
'''

def command(args, **kwargs):
 return subprocess.run(['docker', *args], capture_output=True, text=True, check=True, timeout=90, **kwargs)

def inspect(container):
 return json.loads(command(['inspect', container]).stdout)[0]

def wait_for(predicate, timeout=20):
 deadline=time.monotonic()+timeout
 while time.monotonic()<deadline:
  if predicate(): return
  time.sleep(.1)
 raise RuntimeError('Fixture condition did not become true before deadline')

def run():
 print('R29 RESTORATION SMOKE: CPU-only Docker fixtures; actual production is untouched', flush=True)
 ROOT.mkdir(exist_ok=False)
 before=inspect('glm53-prod')
 names='r29-restore-proof-'+uuid.uuid4().hex[:12]
 owned=[]
 stop_client=None
 get=coordinator.requests.get
 checks=[]
 try:
  def launch(name, code, test=False):
   args=['run','-d','--pull=never','--runtime','runc','--name',name,'--label','field-lab.component='+LABEL,
    '--memory','128m','--memory-swap','128m','--cpus','.25','--pids-limit','16','-e','NVIDIA_VISIBLE_DEVICES=void']
   if test: args+=['--network','none','--label','field-lab.battery=r26']
   else: args+=['--publish','127.0.0.1::8080']
   identifier=command(args+['--entrypoint','python3',IMAGE,'-u','-c',code]).stdout.strip()
   owned.append(identifier)
   state=inspect(identifier)
   assert not state['HostConfig'].get('DeviceRequests'), 'CPU fixture exposes GPU requests'
   assert state['HostConfig']['Runtime']=='runc'
   return identifier
  production_id=launch(names,SERVER)
  def origin():
   state=inspect(production_id)
   port=state['NetworkSettings']['Ports']['8080/tcp'][0]['HostPort']
   return 'http://127.0.0.1:'+port
  def healthy():
   try: return get(origin()+'/health', timeout=.5).status_code==200
   except coordinator.requests.RequestException: return False
  wait_for(healthy)
  started_at=inspect(production_id)['State']['StartedAt']
  stop_client=subprocess.Popen(['docker','stop','--timeout','60',production_id],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
  wait_for(lambda:'STOP_ACCEPTED' in command(['logs',production_id]).stdout)
  assert inspect(production_id)['State']['Running'], 'Slow stop completed before client interruption'
  stop_client.kill()
  stop_client.wait(timeout=5)
  assert stop_client.returncode!=0
  def fixture_get(url, **kwargs):
   assert url.startswith('http://127.0.0.1:5001/'), url
   return get(url.replace('http://127.0.0.1:5001',origin(),1),**kwargs)
  with patch.object(coordinator,'PRODUCTION',names), patch.object(rt,'NAME',names+'-test'), patch.object(rt,'ROOT',ROOT), patch.object(coordinator.requests,'get',side_effect=fixture_get):
   coordinator.restore_production(production_id)
   state=inspect(production_id)
   assert state['State']['Running'] and state['State']['StartedAt']!=started_at
   assert state['Id']==production_id and state['Image']==inspect(IMAGE)['Id']
   assert get(origin()+'/v1/models',timeout=2).json()['data'][0]['id']=='cpu-restoration-fixture'
   checks.append({'case':'accepted-stop-client-killed','passed':True,'client_exit_code':stop_client.returncode,'restored_same_container_id':production_id})
   (ROOT/'production-restored.json').rename(ROOT/'interrupted-stop-restored.json')
   command(['stop','--timeout','60',production_id])
   test_id=launch(names+'-test','import time;time.sleep(120)',test=True)
   with patch.object(rt,'capture',side_effect=subprocess.TimeoutExpired(['diagnostic-fixture'],20)):
    coordinator.restore_production(production_id)
   assert not command(['ps','-a','--no-trunc','--filter','id='+test_id,'--format','{{.ID}}']).stdout.strip()
   assert inspect(production_id)['State']['Running'] and healthy()
   checks.append({'case':'diagnostic-timeout-before-owned-test-removal','passed':True,'test_container_removed':test_id,'original_fixture_restored':production_id})
   (ROOT/'production-restored.json').rename(ROOT/'diagnostic-failure-restored.json')
  after=inspect('glm53-prod')
  for key in ('Id','Image'):
   assert before[key]==after[key], ('Real production identity changed',key)
  assert before['State']['StartedAt']==after['State']['StartedAt'], 'Real production was restarted'
  assert before['State']['Running']==after['State']['Running']
  result={'passed':True,'scope':'Real Docker daemon transitions and actual HTTP health/models on CPU-only owned fixtures. Only restoration HTTP origins were redirected; real production was not stopped or restarted.', 'checks':checks,'real_production_unchanged':True}
  (ROOT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
  print(json.dumps(result,indent=2),flush=True)
 finally:
  if stop_client is not None and stop_client.poll() is None:
   stop_client.kill();stop_client.wait(timeout=5)
  for identifier in reversed(owned):
   remaining=command(['ps','-a','--no-trunc','--filter','id='+identifier,'--format','{{.ID}}']).stdout.splitlines()
   if identifier not in remaining: continue
   state=inspect(identifier)
   assert state['Config'].get('Labels',{}).get('field-lab.component')==LABEL
   command(['rm','-f',identifier])
  if owned:
   assert not command(['ps','-a','--filter','label=field-lab.component='+LABEL,'--format','{{.ID}}']).stdout.strip()

if __name__=='__main__':
 run()
