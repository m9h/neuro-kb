#!/usr/bin/env python3
"""Render the OKF wiki bundle as a single self-contained interactive HTML
visualizer (force-directed concept graph + reading pane). No external assets,
matching OKF's static-visualizer reference tool. Output: okf-visualizer.html.

Usage: python scripts/build_viz.py
"""
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WIKI = ROOT / "wiki"
OUT = ROOT / "okf-visualizer.html"
RESERVED = {"index.md", "log.md"}

def split_fm(text):
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end == -1:
        return {}, text
    return text[4:end], text[end + 5:]

def parse_fm(block):
    """Minimal frontmatter parse: scalar strings + flow lists (tags/related/implementations)."""
    fm = {}
    for ln in block.split("\n"):
        m = re.match(r"([A-Za-z_][\w]*):\s*(.*)$", ln)
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip()
        if val.startswith("[") and val.endswith("]"):
            items = [x.strip().strip('"').strip("'") for x in val[1:-1].split(",") if x.strip()]
            fm[key] = items
        else:
            fm[key] = val.strip('"')
    return fm

def main():
    slugs = {p.stem for p in WIKI.glob("*.md") if p.name not in RESERVED}
    nodes, edges, seen_edge = [], [], set()
    for p in sorted(WIKI.glob("*.md")):
        if p.name in RESERVED:
            continue
        text = p.read_text()
        block, body = split_fm(text)
        fm = parse_fm(block)
        slug = p.stem
        # edges: union of `related:` frontmatter + body markdown links to local slugs
        targets = set()
        for r in fm.get("related", []):
            targets.add(r[:-3] if r.endswith(".md") else r)
        for t in re.findall(r"\]\(([a-z0-9][a-z0-9-]*)\.md\)", body):
            targets.add(t)
        for t in targets:
            if t in slugs and t != slug:
                key = tuple(sorted((slug, t)))
                if key not in seen_edge:
                    seen_edge.add(key)
                    edges.append({"s": key[0], "t": key[1]})
        nodes.append({
            "id": slug,
            "title": fm.get("title", slug),
            "type": fm.get("type", "concept"),
            "tags": fm.get("tags", []),
            "desc": fm.get("description", ""),
            "impl": fm.get("implementations", []),
            "body": body.strip(),
        })
    # degree for sizing
    deg = {n["id"]: 0 for n in nodes}
    for e in edges:
        deg[e["s"]] += 1; deg[e["t"]] += 1
    for n in nodes:
        n["deg"] = deg[n["id"]]
    data = {"nodes": nodes, "edges": edges,
            "generated": "neuro-kb OKF bundle", "counts": {"nodes": len(nodes), "edges": len(edges)}}
    html = TEMPLATE.replace("/*__DATA__*/", json.dumps(data, ensure_ascii=False))
    OUT.write_text(html)
    print(f"wrote {OUT}  ({len(nodes)} nodes, {len(edges)} edges, {OUT.stat().st_size//1024} KB)")

TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>neuro-kb · OKF visualizer</title>
<style>
:root{
  --bg:#0e1116; --panel:#161b22; --panel2:#1c232d; --line:#2a323d; --edge:#303947;
  --fg:#e6edf3; --dim:#8b98a5; --accent:#4aa8ff;
  --t-modality:#4aa8ff; --t-physics:#c98bff; --t-tissue:#ff8f6b;
  --t-method:#5ad19b; --t-head-model:#ffd166; --t-concept:#9aa7b4; --t-coordinate-system:#f472b6;
}
@media (prefers-color-scheme: light){
  :root{--bg:#f6f8fa;--panel:#fff;--panel2:#f0f3f6;--line:#d0d7de;--edge:#c4ccd6;
        --fg:#1f2328;--dim:#57606a;}
}
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:var(--bg);color:var(--fg);
  font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
#app{display:flex;height:100vh;overflow:hidden}
#left{flex:1 1 60%;position:relative;min-width:0}
#right{flex:1 1 40%;max-width:560px;min-width:320px;background:var(--panel);
  border-left:1px solid var(--line);overflow-y:auto;padding:20px 24px}
svg{width:100%;height:100%;display:block;cursor:grab}
svg:active{cursor:grabbing}
line.edge{stroke:var(--edge);stroke-width:1}
line.edge.hi{stroke:var(--accent);stroke-width:2}
circle.node{cursor:pointer;stroke:var(--bg);stroke-width:1.5}
circle.node.dim{opacity:.18}
text.lbl{fill:var(--dim);font-size:9px;pointer-events:none;user-select:none}
text.lbl.hi{fill:var(--fg);font-weight:600}
#top{position:absolute;top:12px;left:12px;right:12px;display:flex;gap:10px;
  align-items:center;flex-wrap:wrap;z-index:5;pointer-events:none}
#top>*{pointer-events:auto}
#search{background:var(--panel);border:1px solid var(--line);color:var(--fg);
  border-radius:8px;padding:7px 11px;width:220px;outline:none}
#search:focus{border-color:var(--accent)}
.legend{display:flex;gap:6px;flex-wrap:wrap}
.chip{display:inline-flex;align-items:center;gap:5px;background:var(--panel);
  border:1px solid var(--line);border-radius:20px;padding:4px 10px;font-size:12px;
  cursor:pointer;user-select:none;color:var(--dim)}
.chip.off{opacity:.4}
.chip .dot{width:9px;height:9px;border-radius:50%}
.meta{color:var(--dim);font-size:12px;margin-left:auto;background:var(--panel);
  border:1px solid var(--line);border-radius:8px;padding:6px 10px}
#right h1{font-size:20px;margin:.1em 0 .1em}
#right .type-badge{display:inline-block;font-size:11px;padding:2px 9px;border-radius:20px;
  color:#0b0e12;font-weight:700;margin-bottom:10px}
#right .desc{color:var(--dim);font-style:italic;margin:0 0 14px}
#right .tags{display:flex;gap:6px;flex-wrap:wrap;margin:0 0 16px}
#right .tag{font-size:11px;background:var(--panel2);border:1px solid var(--line);
  border-radius:6px;padding:2px 8px;color:var(--dim)}
#body h2{font-size:15px;border-bottom:1px solid var(--line);padding-bottom:4px;margin:20px 0 8px}
#body h3{font-size:13px;margin:14px 0 6px}
#body p{margin:8px 0}
#body code{background:var(--panel2);padding:1px 5px;border-radius:4px;font-size:12.5px}
#body pre{background:var(--panel2);border:1px solid var(--line);border-radius:8px;
  padding:12px;overflow-x:auto;font-size:12.5px}
#body pre code{background:none;padding:0}
#body table{border-collapse:collapse;width:100%;font-size:12.5px;margin:10px 0;display:block;overflow-x:auto}
#body th,#body td{border:1px solid var(--line);padding:5px 8px;text-align:left}
#body th{background:var(--panel2)}
#body a{color:var(--accent);text-decoration:none;cursor:pointer}
#body a:hover{text-decoration:underline}
#body ul{margin:8px 0;padding-left:22px}
#body li{margin:3px 0}
.hint{color:var(--dim);text-align:center;margin-top:40vh}
.xref{margin-top:22px;padding-top:14px;border-top:1px solid var(--line)}
.xref b{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--dim)}
.xref a{display:inline-block;margin:4px 8px 0 0;color:var(--accent);cursor:pointer;font-size:13px}
</style>
</head>
<body>
<div id="app">
  <div id="left">
    <div id="top">
      <input id="search" placeholder="Search concepts…" autocomplete="off">
      <div class="legend" id="legend"></div>
      <div class="meta" id="meta"></div>
    </div>
    <svg id="svg"><g id="scene"><g id="edges"></g><g id="nodes"></g><g id="labels"></g></g></svg>
  </div>
  <div id="right"><div class="hint">Click a node to read its page.<br>Drag to pan · scroll to zoom · drag a node to pin.</div></div>
</div>
<script>
const DATA = /*__DATA__*/;
const TYPES = ["modality","physics","tissue","method","head-model","concept","coordinate-system"];
const color = t => getComputedStyle(document.documentElement).getPropertyValue('--t-'+t).trim() || '#9aa7b4';
const byId = Object.fromEntries(DATA.nodes.map(n=>[n.id,n]));
const active = new Set(TYPES);

// ---- layout (force-directed, velocity Verlet) ----
const W=1200,H=800;
const N=DATA.nodes; const E=DATA.edges;
N.forEach((n,i)=>{const a=i/N.length*6.283; n.x=W/2+Math.cos(a)*260+(i%7)*7; n.y=H/2+Math.sin(a)*260+(i%5)*7; n.vx=0; n.vy=0; n.pin=false;});
const adj=Object.fromEntries(N.map(n=>[n.id,new Set()]));
E.forEach(e=>{adj[e.s].add(e.t); adj[e.t].add(e.s);});
function tick(){
  for(let i=0;i<N.length;i++){for(let j=i+1;j<N.length;j++){
    const a=N[i],b=N[j];let dx=a.x-b.x,dy=a.y-b.y;let d2=dx*dx+dy*dy||1;let d=Math.sqrt(d2);
    let f=2600/d2; let fx=dx/d*f,fy=dy/d*f; a.vx+=fx;a.vy+=fy;b.vx-=fx;b.vy-=fy;}}
  E.forEach(e=>{const a=byId[e.s],b=byId[e.t];let dx=b.x-a.x,dy=b.y-a.y;let d=Math.sqrt(dx*dx+dy*dy)||1;
    let f=(d-90)*0.015; let fx=dx/d*f,fy=dy/d*f; a.vx+=fx;a.vy+=fy;b.vx-=fx;b.vy-=fy;});
  N.forEach(n=>{n.vx+=(W/2-n.x)*0.003; n.vy+=(H/2-n.y)*0.003;
    if(!n.pin){n.x+=n.vx*=0.82; n.y+=n.vy*=0.82;}});
}
for(let k=0;k<420;k++) tick();

// ---- render ----
const svg=document.getElementById('svg'),scene=document.getElementById('scene');
const gE=document.getElementById('edges'),gN=document.getElementById('nodes'),gL=document.getElementById('labels');
const NS='http://www.w3.org/2000/svg';
const el=(t,a)=>{const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);return e;};
const lineEls={},nodeEls={},lblEls={};
E.forEach((e,i)=>{const l=el('line',{class:'edge','data-s':e.s,'data-t':e.t});gE.appendChild(l);lineEls[i]=l;});
N.forEach(n=>{
  const r=5+Math.min(n.deg,10)*1.1;
  const c=el('circle',{class:'node',r,cx:n.x,cy:n.y,fill:color(n.type),'data-id':n.id});
  c.addEventListener('click',ev=>{ev.stopPropagation();select(n.id);});
  c.addEventListener('mouseenter',()=>hover(n.id,true));
  c.addEventListener('mouseleave',()=>hover(n.id,false));
  c.addEventListener('mousedown',ev=>startDrag(ev,n));
  gN.appendChild(c);nodeEls[n.id]=c;
  const t=el('text',{class:'lbl',x:n.x,y:n.y-r-3,'text-anchor':'middle'});t.textContent=n.title;
  gL.appendChild(t);lblEls[n.id]=t;
});
function redraw(){
  E.forEach((e,i)=>{const a=byId[e.s],b=byId[e.t];const l=lineEls[i];
    l.setAttribute('x1',a.x);l.setAttribute('y1',a.y);l.setAttribute('x2',b.x);l.setAttribute('y2',b.y);});
  N.forEach(n=>{const c=nodeEls[n.id];c.setAttribute('cx',n.x);c.setAttribute('cy',n.y);
    const t=lblEls[n.id];t.setAttribute('x',n.x);t.setAttribute('y',n.y-(+c.getAttribute('r'))-3);});
}
redraw();

// ---- pan/zoom ----
let vb={x:0,y:0,w:W,h:H};function applyVB(){svg.setAttribute('viewBox',`${vb.x} ${vb.y} ${vb.w} ${vb.h}`);}applyVB();
svg.addEventListener('wheel',e=>{e.preventDefault();const s=e.deltaY>0?1.1:0.9;
  const pt=cursor(e);vb.x=pt.x-(pt.x-vb.x)*s;vb.y=pt.y-(pt.y-vb.y)*s;vb.w*=s;vb.h*=s;applyVB();},{passive:false});
let pan=null;
svg.addEventListener('mousedown',e=>{if(e.target===svg||e.target.id==='scene')pan={x:e.clientX,y:e.clientY,vx:vb.x,vy:vb.y};});
window.addEventListener('mousemove',e=>{
  if(pan){const k=vb.w/svg.clientWidth;vb.x=pan.vx-(e.clientX-pan.x)*k;vb.y=pan.vy-(e.clientY-pan.y)*k;applyVB();}
});
window.addEventListener('mouseup',()=>{pan=null;drag=null;});
function cursor(e){const r=svg.getBoundingClientRect();return{x:vb.x+(e.clientX-r.left)/r.width*vb.w,y:vb.y+(e.clientY-r.top)/r.height*vb.h};}

// ---- node drag ----
let drag=null;
function startDrag(e,n){e.stopPropagation();drag=n;n.pin=true;}
window.addEventListener('mousemove',e=>{if(drag){const p=cursor(e);drag.x=p.x;drag.y=p.y;drag.vx=drag.vy=0;anim();}});

let raf=null;function anim(){if(raf)return;let f=0;const step=()=>{tick();redraw();if(++f<60){raf=requestAnimationFrame(step);}else raf=null;};raf=requestAnimationFrame(step);}

// ---- hover highlight ----
function hover(id,on){
  const nb=adj[id];
  E.forEach((e,i)=>{const rel=(e.s===id||e.t===id);lineEls[i].classList.toggle('hi',on&&rel);});
  N.forEach(n=>{const near=on?(n.id===id||nb.has(n.id)):true;
    nodeEls[n.id].classList.toggle('dim',on&&!near&&visible(n));
    lblEls[n.id].classList.toggle('hi',on&&(n.id===id||nb.has(n.id)));});
}

// ---- filter/search ----
function visible(n){return active.has(n.type);}
function applyFilter(q){
  q=(q||'').toLowerCase();
  N.forEach(n=>{
    const hit=!q||n.title.toLowerCase().includes(q)||n.id.includes(q)||n.tags.join(' ').includes(q)||n.desc.toLowerCase().includes(q);
    const show=visible(n)&&hit;
    nodeEls[n.id].style.display=show?'':'none';
    lblEls[n.id].style.display=show?'':'none';
  });
  E.forEach((e,i)=>{const on=visible(byId[e.s])&&visible(byId[e.t]);lineEls[i].style.display=on?'':'none';});
}
document.getElementById('search').addEventListener('input',e=>applyFilter(e.target.value));

// ---- legend ----
const legend=document.getElementById('legend');
TYPES.forEach(t=>{
  const c=DATA.nodes.filter(n=>n.type===t).length;if(!c)return;
  const chip=document.createElement('span');chip.className='chip';
  chip.innerHTML=`<span class="dot" style="background:${color(t)}"></span>${t} <span style="opacity:.6">${c}</span>`;
  chip.addEventListener('click',()=>{active.has(t)?active.delete(t):active.add(t);chip.classList.toggle('off');applyFilter(document.getElementById('search').value);});
  legend.appendChild(chip);
});
document.getElementById('meta').textContent=`${DATA.counts.nodes} concepts · ${DATA.counts.edges} links`;

// ---- reading pane + tiny markdown ----
function esc(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function inline(s){
  return esc(s)
    .replace(/`([^`]+)`/g,'<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g,'<b>$1</b>')
    .replace(/\[([^\]]+)\]\(([a-z0-9][a-z0-9-]*)\.md\)/g,'<a data-nav="$2">$1</a>')
    .replace(/\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g,'<a href="$2" target="_blank" rel="noopener">$1</a>');
}
function md(src){
  const lines=src.split('\n');let html='',i=0;
  while(i<lines.length){
    let ln=lines[i];
    if(ln.startsWith('```')){let buf=[];i++;while(i<lines.length&&!lines[i].startsWith('```')){buf.push(esc(lines[i]));i++;}i++;html+=`<pre><code>${buf.join('\n')}</code></pre>`;continue;}
    if(/^\|.*\|/.test(ln)){let rows=[];while(i<lines.length&&/^\|/.test(lines[i])){rows.push(lines[i]);i++;}
      const cells=r=>r.replace(/^\||\|$/g,'').split('|').map(c=>c.trim());
      let out='<table>';rows.forEach((r,ri)=>{if(/^\|[\s:|-]+\|?$/.test(r))return;const tag=ri===0?'th':'td';
        out+='<tr>'+cells(r).map(c=>`<${tag}>${inline(c)}</${tag}>`).join('')+'</tr>';});html+=out+'</table>';continue;}
    let m;
    if(m=ln.match(/^(#{1,3})\s+(.*)/)){const h=m[1].length+1;html+=`<h${h}>${inline(m[2])}</h${h}>`;i++;continue;}
    if(/^[-*]\s+/.test(ln)){let items=[];while(i<lines.length&&/^[-*]\s+/.test(lines[i])){items.push(`<li>${inline(lines[i].replace(/^[-*]\s+/,''))}</li>`);i++;}html+=`<ul>${items.join('')}</ul>`;continue;}
    if(ln.trim()===''){i++;continue;}
    let para=[];while(i<lines.length&&lines[i].trim()!==''&&!/^(#|\||```|[-*]\s)/.test(lines[i])){para.push(lines[i]);i++;}
    html+=`<p>${inline(para.join(' '))}</p>`;
  }
  return html;
}
function select(id){
  const n=byId[id];if(!n)return;
  N.forEach(x=>nodeEls[x.id].setAttribute('stroke-width',x.id===id?'3':'1.5'));
  nodeEls[id].setAttribute('stroke',getComputedStyle(document.documentElement).getPropertyValue('--accent'));
  const nb=[...adj[id]].map(t=>byId[t]).sort((a,b)=>a.title.localeCompare(b.title));
  const impl=n.impl.length?`<div class="xref"><b>Implementations</b><br>${n.impl.map(x=>`<code>${esc(x)}</code>`).join('&nbsp; ')}</div>`:'';
  const xref=nb.length?`<div class="xref"><b>Linked concepts</b><br>${nb.map(x=>`<a data-nav="${x.id}">${esc(x.title)}</a>`).join('')}</div>`:'';
  document.getElementById('right').innerHTML=
    `<span class="type-badge" style="background:${color(n.type)}">${n.type}</span>`+
    `<h1>${esc(n.title)}</h1>`+
    (n.desc?`<p class="desc">${esc(n.desc)}</p>`:'')+
    (n.tags.length?`<div class="tags">${n.tags.map(t=>`<span class="tag">${esc(t)}</span>`).join('')}</div>`:'')+
    `<div id="body">${md(n.body)}</div>`+impl+xref;
  document.getElementById('right').scrollTop=0;
  document.querySelectorAll('#right [data-nav]').forEach(a=>a.addEventListener('click',()=>select(a.dataset.nav)));
}
window.location.hash&&select(decodeURIComponent(window.location.hash.slice(1)));
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
