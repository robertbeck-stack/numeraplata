"""Vefþjónn og HTML viðmót fyrir Numby ALPR."""
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
import cv2
import numpy as np

from .scanner import make_alpr
from .storage import load_logs, LOG_FILE, CAPTURES_DIR, is_safe_path
from .pipeline import process_frame

HTML = b"""<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Numby ALPR (Python)</title>
<style>
body{font:16px system-ui;margin:0;background:#121212;color:#eee}
video{width:100%;max-height:42vh;background:#000;display:block;object-fit:cover}
.bar{display:flex;justify-content:space-between;align-items:center;padding:10px 14px;background:#1e1e1e}
button,a.btn{background:#bb86fc;color:#000;border:0;padding:8px 14px;border-radius:18px;font-weight:700;text-decoration:none;cursor:pointer}
.day{background:#262626;padding:6px 14px;font-weight:700;font-size:14px;display:flex;justify-content:space-between}
.row{display:flex;justify-content:space-between;align-items:center;padding:10px 14px;border-bottom:1px solid #2a2a2a;cursor:pointer}
.row img{height:48px;width:92px;object-fit:cover;border-radius:4px}
dialog{background:#1e1e1e;color:#eee;border:1px solid #444;border-radius:12px;max-width:92vw;padding:16px}
dialog img{max-width:85vw;max-height:65vh;display:block;margin:10px 0}
#err{background:#b62324;color:#fff;padding:8px 12px;display:none;font-size:14px}
</style>
<div id=err></div>
<video id=v autoplay playsinline></video><canvas id=c hidden></canvas>
<div class=bar><b id=cnt>Atvik (0)</b><div><a class=btn href="/plates.csv" download="plates.csv">Deila CSV</a></div></div>
<div id=list></div>
<dialog id=dlg onclick="if(event.target===this)this.close()">
  <h3 id=d_title style="margin:0"></h3><small id=d_time></small>
  <img id=d_img><div style="display:flex;gap:8px;justify-content:flex-end">
  <a id=d_dl class=btn download>S\xc3\xa6kja mynd</a><button onclick="dlg.close()">Loka</button></div>
</dialog>
<script>
const v=document.getElementById('v'),c=document.getElementById('c'),list=document.getElementById('list'),cnt=document.getElementById('cnt'),dlg=document.getElementById('dlg'),err=document.getElementById('err');
navigator.mediaDevices.getUserMedia({video:{facingMode:'environment',width:{ideal:1920},height:{ideal:1080}}})
  .then(s=>v.srcObject=s)
  .catch(e=>{err.textContent='Gat ekki opna\xc3\xb0 myndav\xc3\xa9l. Athuga\xc3\xb0u a\xc3\xb0gangsheimildir og a\xc3\xb0 s\xc3\xad\xc3\xb0an s\xc3\xa9 \xc3\xa1 HTTPS e\xc3\xb0a localhost.';err.style.display='block';});
let events=[];
function render(){
  cnt.textContent=`Atvik (${events.length})`;
  const byDay={};
  events.forEach(e=>{(byDay[e.time.split(' ')[0]]??=[]).push(e)});
  list.innerHTML=Object.entries(byDay).map(([d,arr])=>
    `<div class=day><span>${d}</span><span>${arr.length} atvik</span></div>`+
    arr.map((x,i)=>`<div class=row onclick='show(${JSON.stringify(x)})'><div><span style="color:#bb86fc;margin-right:12px">${x.time.split(' ')[1]}</span><b style="font-size:18px">${x.plate}</b></div>${x.img?`<img src="${x.img}">`:''}</div>`).join('')
  ).join('');
}
function show(x){
  document.getElementById('d_title').textContent=x.plate;
  document.getElementById('d_time').textContent=x.time;
  const img=document.getElementById('d_img'),dl=document.getElementById('d_dl');
  if(x.img){img.src=x.img;img.hidden=false;dl.href=x.img;dl.hidden=false;}else{img.hidden=true;dl.hidden=true;}
  dlg.showModal();
}
let scanning=false;
function scanLoop(){
  if(!v.videoWidth||scanning){setTimeout(scanLoop,600);return;}
  scanning=true;
  c.width=v.videoWidth;c.height=v.videoHeight;
  c.getContext('2d').drawImage(v,0,0);
  c.toBlob(b=>{
    if(!b){scanning=false;setTimeout(scanLoop,600);return;}
    fetch('/scan',{method:'POST',body:b})
      .then(r=>r.json())
      .then(all=>{if(all&&all.length){events=all;render();}})
      .catch(()=>{})
      .finally(()=>{scanning=false;setTimeout(scanLoop,600);});
  },'image/jpeg',0.92);
}
setTimeout(scanLoop,1000);
</script>"""

DASHBOARD_HTML = """<!doctype html><html lang=is><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Numby ALPR</title>
<style>
:root{--bg:#0f1115;--panel:#171a21;--line:#262a33;--text:#e8eaed;--muted:#8b919c;--accent:#7c9cff;--ok:#3ecf8e}
*{box-sizing:border-box}
html,body{height:100%;margin:0}
body{font:15px/1.4 system-ui,-apple-system,"Segoe UI",sans-serif;background:var(--bg);color:var(--text);display:flex;flex-direction:column}
header{display:flex;align-items:center;gap:14px;padding:12px 20px;border-bottom:1px solid var(--line);background:var(--panel)}
header h1{font-size:17px;margin:0;font-weight:650;letter-spacing:.2px}
.status{display:flex;align-items:center;gap:7px;color:var(--muted);font-size:13px}
.dot{width:8px;height:8px;border-radius:50%;background:var(--ok);box-shadow:0 0 0 3px rgba(62,207,142,.18)}
.dot.off{background:#e5534b;box-shadow:0 0 0 3px rgba(229,83,75,.18)}
.spacer{flex:1}
.btn{background:transparent;color:var(--text);border:1px solid var(--line);padding:7px 14px;border-radius:8px;font:inherit;font-size:13px;text-decoration:none;cursor:pointer}
.btn:hover{border-color:var(--accent);color:var(--accent)}
main{flex:1;display:grid;grid-template-columns:1fr 380px;min-height:0}
.video{padding:20px;display:flex;align-items:center;justify-content:center;min-height:0;position:relative}
.video img{width:100%;height:100%;object-fit:contain;border-radius:12px;cursor:crosshair}
.zoom{display:flex;align-items:center;gap:10px;color:var(--muted);font-size:13px}
.zoom input{accent-color:var(--accent);width:150px}
.zoom output{min-width:38px;color:var(--text);font-variant-numeric:tabular-nums}
.hint{position:absolute;left:50%;transform:translateX(-50%);bottom:30px;white-space:nowrap;background:rgba(0,0,0,.55);color:#ddd;font-size:12px;padding:5px 10px;border-radius:6px;pointer-events:none}
aside{border-left:1px solid var(--line);background:var(--panel);display:flex;flex-direction:column;min-height:0}
.ahead{display:flex;justify-content:space-between;align-items:baseline;padding:16px 18px 10px}
.ahead h2{font-size:14px;margin:0;text-transform:uppercase;letter-spacing:.8px;color:var(--muted);font-weight:600}
.ahead span{font-size:13px;color:var(--muted)}
#list{overflow-y:auto;flex:1;padding:0 10px 12px}
.day{font-size:12px;color:var(--muted);padding:12px 8px 6px;display:flex;justify-content:space-between}
.ev{display:flex;gap:12px;align-items:center;padding:8px;border-radius:10px;cursor:pointer;position:relative}
.ev:hover{background:#1f232c}
.ev.new{animation:flash 1.6s ease-out}
@keyframes flash{from{background:rgba(124,156,255,.25)}to{background:transparent}}
.ev img{width:104px;height:60px;object-fit:cover;border-radius:6px;background:#000;flex:none}
.meta{display:flex;flex-direction:column;gap:5px;min-width:0}
.sub{font-size:12px;color:var(--muted)}
.plate{display:inline-flex;align-items:stretch;border:2px solid #111;border-radius:5px;background:#fff;color:#111;overflow:hidden;width:max-content}
.plate .is{background:#1d4fa8;color:#fff;font-size:9px;font-weight:700;display:flex;align-items:flex-end;padding:0 3px 2px}
.plate b{font:700 19px/1 "Segoe UI",system-ui,sans-serif;letter-spacing:2px;padding:4px 8px}
.plate.big b{font-size:34px;padding:6px 14px}.plate.big .is{font-size:12px;padding:0 5px 4px}
.people{display:flex;justify-content:space-between;align-items:center;gap:12px;margin:14px 14px 4px;padding:12px 14px;border:1px solid var(--line);border-left:3px solid #ffd200;border-radius:10px;background:#1b1f27}
.people.cars{border-left-color:#e62828;margin-top:8px}
.plabel{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.8px;font-weight:600}
.pnow{font-size:30px;font-weight:700;line-height:1.1;font-variant-numeric:tabular-nums}
.pside{text-align:right;display:flex;flex-direction:column;gap:4px}
.pside a{font-size:12px;color:var(--accent);text-decoration:none}.pside a:hover{text-decoration:underline}
.empty{color:var(--muted);text-align:center;padding:40px 20px;font-size:14px}
dialog{background:var(--panel);color:var(--text);border:1px solid var(--line);border-radius:14px;padding:18px;max-width:min(92vw,1100px)}
dialog::backdrop{background:rgba(0,0,0,.6)}
dialog img{max-width:100%;max-height:70vh;display:block;border-radius:8px;margin:14px 0}
.drow{display:flex;justify-content:space-between;align-items:center;gap:16px}
.dact{display:flex;gap:8px;justify-content:flex-end}
.btn.danger{color:#ff7b72;border-color:#5a2a2a;margin-right:auto}
.btn.danger:hover,.btn.danger.confirm{color:#fff;background:#b62324;border-color:#b62324}
.ev .del{position:absolute;right:8px;top:50%;transform:translateY(-50%);opacity:0;background:transparent;border:1px solid var(--line);color:var(--muted);border-radius:6px;width:30px;height:30px;cursor:pointer;font-size:15px;line-height:1}
.ev:hover .del{opacity:1}.ev .del:hover{color:#fff;background:#b62324;border-color:#b62324}
.ev .del.confirm{opacity:1;width:auto;padding:0 10px;font-size:12px;color:#fff;background:#b62324;border-color:#b62324}
@media (hover:none){.ev .del{opacity:1}}
.btn.shot{color:var(--accent);border-color:var(--accent)}
.btn.shot:disabled{opacity:.5;cursor:wait}
.flash{position:fixed;inset:0;background:#fff;opacity:0;pointer-events:none;transition:opacity .35s}
.flash.on{opacity:.6;transition:none}
.toast{position:fixed;left:50%;bottom:24px;transform:translateX(-50%) translateY(20px);background:var(--panel);border:1px solid var(--line);color:var(--text);padding:10px 16px;border-radius:10px;font-size:14px;opacity:0;transition:.25s;display:flex;gap:12px;align-items:center;z-index:10}
.toast.show{opacity:1;transform:translateX(-50%)}
.toast a{color:var(--accent)}
@media (max-width:860px){main{grid-template-columns:1fr;grid-template-rows:auto 1fr}aside{border-left:0;border-top:1px solid var(--line)}.video{padding:12px}}
</style>
<header>
  <h1>Numby ALPR <span style="font-size:12px;color:#7c9cff;font-weight:400;margin-left:6px;vertical-align:middle;background:#1b1f27;padding:2px 6px;border-radius:4px;border:1px solid #2d3340">v2.3.0</span></h1>
  <div class=status><span id=dot class=dot></span><span id=stat>Tengist…</span></div>
  <div class=spacer></div>
  <label class=zoom>Aðdráttur <input id=zr type=range min=1 max=4 step=0.1 value=1><output id=zv>1.0×</output></label>
  <button class=btn id=zreset>Endurstilla</button>
  <button class="btn shot" id=shot title="Vistar það sem sést, í fullri upplausn, með öllum kössum">📷 Taka mynd</button>
  <a class=btn href="/plates.csv" download="plates.csv">Sækja CSV</a>
</header>
<main>
  <section class=video><img id=stream src="/stream" alt="Myndavél"><div class=hint>Smelltu á myndina til að miðja · skrunaðu til að þysja</div></section>
  <aside>
    <div class=people>
      <div><div class=plabel>Fólk á svæðinu</div><div class=pnow id=pnow>–</div></div>
      <div class=pside><div class=sub id=ppeak>Mest í dag: –</div><a href="/people.csv" download="people.csv">Sækja talningu</a></div>
    </div>
    <div class="people cars">
      <div><div class=plabel>Bílar framhjá í dag</div><div class=pnow id=tnow>–</div></div>
      <div class=pside><div class=sub id=tdir>← 0 · 0 →</div><a href="/traffic.csv" download="traffic.csv">Sækja umferð</a></div>
    </div>
    <div class=ahead><h2>Skráð númer</h2><span id=cnt>0</span></div>
    <div id=list><div class=empty>Engin númer skráð enn.<br>Þau birtast hér um leið og plata er lesin.</div></div>
  </aside>
</main>
<dialog id=dlg>
  <div class=drow><div id=d_plate></div><div class=sub id=d_time></div></div>
  <img id=d_img alt="">
  <div class=dact><button class="btn danger" id=d_del>Eyða færslu</button><a id=d_dl class=btn download>Sækja mynd</a><button class=btn onclick="dlg.close()">Loka</button></div>
</dialog>
<script>
const $=id=>document.getElementById(id);
const escapeHtml=str=>String(str).replace(/[&<>'"]/g,tag=>({ '&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;' }[tag]||tag));
const plateHtml=(p,big)=>`<span class="plate${big?' big':''}"><span class=is>IS</span><b>${escapeHtml(p)}</b></span>`;
let last='',seen=new Set(),first=true;
$('dlg').addEventListener('click',e=>{if(e.target===$('dlg'))$('dlg').close()});
let cur=null;
async function del(e){
  const r=await fetch('/delete',{method:'POST',body:JSON.stringify({time:e.time,plate:e.plate})});
  if(r.ok){last='';await poll()}
}
function armDelete(btn,label,onConfirm){
  if(btn.classList.contains('confirm')){onConfirm();return}
  const orig=btn.textContent;btn.classList.add('confirm');btn.textContent=label;
  setTimeout(()=>{btn.classList.remove('confirm');btn.textContent=orig},3000);
}
$('d_del').onclick=()=>armDelete($('d_del'),'Smelltu aftur til að eyða',()=>{$('dlg').close();del(cur)});
function show(e){cur=e;$('d_del').classList.remove('confirm');$('d_del').textContent='Eyða færslu';
  $('d_plate').innerHTML=plateHtml(e.plate,true);$('d_time').textContent=`${e.time} · ${e.conf}% öryggi`;
  $('d_img').src=e.img;$('d_dl').href=e.img;$('dlg').showModal()}
function render(evs){
  $('cnt').textContent=evs.length+' atvik';
  if(!evs.length){$('list').innerHTML='<div class=empty>Engin númer skráð enn.<br>Þau birtast hér um leið og plata er lesin.</div>';return}
  const byDay={};evs.forEach(e=>(byDay[e.time.split(' ')[0]]??=[]).push(e));
  const list=$('list');list.innerHTML='';
  for(const[d,arr]of Object.entries(byDay)){
    const h=document.createElement('div');h.className='day';h.innerHTML=`<span>${d}</span><span>${arr.length}</span>`;list.append(h);
    for(const e of arr){
      const key=e.time+e.plate,el=document.createElement('div');
      el.className='ev'+(!first&&!seen.has(key)?' new':'');seen.add(key);
      el.innerHTML=`<img src="${e.img}" loading=lazy alt=""><div class=meta>${plateHtml(e.plate)}<span class=sub>${e.time.split(' ')[1]} · ${e.conf}%</span></div>`;
      const b=document.createElement('button');b.className='del';b.title='Eyða færslu';b.textContent='✕';
      b.onclick=ev=>{ev.stopPropagation();armDelete(b,'Eyða?',()=>del(e))};
      el.append(b);el.onclick=()=>show(e);list.append(el);
    }
  }
  first=false;
}
async function poll(){
  try{
    const r=await (await fetch('/events')).json();
    $('dot').className='dot'+(r.live?'':' off');
    $('stat').textContent=r.live?`Í gangi · ${r.res} · ${r.fps} greiningar/s`:'Myndavél ekki tengd';
    if(r.people){$('pnow').textContent=r.live?r.people.now:'–';
      $('ppeak').textContent=r.people.peak?`Mest í dag: ${r.people.peak} kl. ${r.people.peak_time}`:'Mest í dag: 0'}
    if(r.traffic){$('tnow').textContent=r.traffic.left+r.traffic.right;
      $('tdir').textContent=`← ${r.traffic.left} · ${r.traffic.right} →`}
    const s=JSON.stringify(r.events);if(s!==last){last=s;render(r.events)}
  }catch{$('dot').className='dot off';$('stat').textContent='Þjónn ekki svarar'}
}
let Z={zoom:1,cx:.5,cy:.5},dragging=false,zt;
function showZoom(){$('zr').value=Z.zoom;$('zv').textContent=Number(Z.zoom).toFixed(1)+'×'}
function sendZoom(){clearTimeout(zt);zt=setTimeout(async()=>{
  Z=await (await fetch('/zoom',{method:'POST',body:JSON.stringify(Z)})).json();showZoom()},60)}
$('zr').addEventListener('input',e=>{dragging=true;Z.zoom=+e.target.value;showZoom();sendZoom()});
$('zr').addEventListener('change',()=>dragging=false);
$('zreset').onclick=()=>{Z={zoom:1,cx:.5,cy:.5};showZoom();sendZoom()};
const img=$('stream');
function imgPos(ev){
  const r=img.getBoundingClientRect(),ar=img.naturalWidth/img.naturalHeight||16/9;
  let w=r.width,h=r.height;if(w/h>ar)w=h*ar;else h=w/ar;
  const x=(ev.clientX-r.left-(r.width-w)/2)/w,y=(ev.clientY-r.top-(r.height-h)/2)/h;
  return x<0||x>1||y<0||y>1?null:[x,y]}
img.addEventListener('click',ev=>{const p=imgPos(ev);if(!p)return;
  Z.cx+=(p[0]-.5)/Z.zoom;Z.cy+=(p[1]-.5)/Z.zoom;sendZoom()});
img.addEventListener('wheel',ev=>{ev.preventDefault();const p=imgPos(ev);
  const nz=Math.min(4,Math.max(1,Math.round((Z.zoom*(ev.deltaY<0?1.15:1/1.15))*10)/10));
  if(p){const fx=Z.cx+(p[0]-.5)/Z.zoom,fy=Z.cy+(p[1]-.5)/Z.zoom;
    Z.cx=fx-(p[0]-.5)/nz;Z.cy=fy-(p[1]-.5)/nz}
  Z.zoom=nz;showZoom();sendZoom()},{passive:false});
fetch('/zoom').then(r=>r.json()).then(z=>{Z=z;showZoom()});
const flash=document.createElement('div');flash.className='flash';document.body.append(flash);
const toast=document.createElement('div');toast.className='toast';document.body.append(toast);
let tt;function say(html){toast.innerHTML=html;toast.classList.add('show');clearTimeout(tt);tt=setTimeout(()=>toast.classList.remove('show'),6000)}
async function takePhoto(){
  const b=$('shot');if(b.disabled)return;b.disabled=true;
  try{
    const r=await fetch('/photo',{method:'POST'});
    if(!r.ok)throw 0;
    const p=await r.json();
    flash.classList.add('on');requestAnimationFrame(()=>requestAnimationFrame(()=>flash.classList.remove('on')));
    say(`Mynd vistuð (${p.res}) <a href="${p.img}" target=_blank>Opna</a><a href="${p.img}" download="${p.name}">Sækja</a>`);
  }catch{say('Gat ekki tekið mynd – er myndavélin tengd?')}
  b.disabled=false;
}
$('shot').onclick=takePhoto;
document.addEventListener('keydown',e=>{if(e.key===' '&&!e.target.closest('input,button,a,dialog')){e.preventDefault();takePhoto()}});
poll();setInterval(poll,1000);
</script></html>"""


def _send_file(handler, path: str, ctype: str) -> bool:
    """Sendir skrá á öruggan hátt ef hún er til."""
    if not os.path.isfile(path):
        return False
    handler.send_response(200)
    handler.send_header("Content-Type", ctype)
    handler.end_headers()
    with open(path, "rb") as f:
        handler.wfile.write(f.read())
    return True


def serve(port: int = 8000, host: str = "0.0.0.0") -> None:
    """Vefþjónn fyrir vefmyndavél í síma/vafra (--web)."""
    alpr = make_alpr()
    logs = load_logs()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/plates.csv" and os.path.exists(LOG_FILE):
                _send_file(self, LOG_FILE, "text/csv; charset=utf-8")
                return
            clean_path = self.path.split("?")[0].lstrip("/\\")
            if clean_path.startswith(CAPTURES_DIR) and is_safe_path(CAPTURES_DIR, clean_path):
                if _send_file(self, clean_path, "image/jpeg"):
                    return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML)

        def do_POST(self):
            buf = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            frame = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
            updated = process_frame(alpr, frame, logs, alpr.predict(frame, everything=True)) if frame is not None else []
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            payload = [{k: v for k, v in e.items() if k != "ts_sec"} for e in updated]
            self.wfile.write(json.dumps(payload).encode())

        def log_message(self, *args):
            pass

    print(f"Vefviðmót í gangi: http://{host}:{port}")
    HTTPServer((host, port), Handler).serve_forever()
