import React from 'react';
import {AbsoluteFill, continueRender, delayRender, Easing, interpolate, useCurrentFrame} from 'remotion';
import data from './data.json';

const C = {
  ink: '#242039', muted: '#79748D', line: '#ECE9F2',
  purple: '#8C68E8', purpleDark: '#6842C1', gold: '#EDB943',
  goldDark: '#BF8821', rose: '#E27B9A', roseDark: '#B64167',
};
const progress = (frame: number, start: number, duration: number) =>
  interpolate(frame, [start, start + duration], [0, 1],
    {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: Easing.bezier(0.2, 0.8, 0.2, 1)});
const fix = (n: number) => n.toFixed(2);
const Label: React.FC<React.PropsWithChildren<{style?: React.CSSProperties}>> = ({children, style}) =>
  <div style={{fontSize: 16, fontWeight: 700, letterSpacing: 1.45, color: C.muted, textTransform: 'uppercase', ...style}}>{children}</div>;
const Note: React.FC<React.PropsWithChildren<{style?: React.CSSProperties}>> = ({children, style}) =>
  <div style={{fontSize: 17, lineHeight: 1.45, color: C.muted, ...style}}>{children}</div>;
const Pill: React.FC<React.PropsWithChildren<{color?: string; bg?: string; style?: React.CSSProperties}>> = ({children, color = C.purpleDark, bg = '#F0EAFB', style}) =>
  <div style={{display: 'inline-flex', alignItems: 'center', borderRadius: 10, padding: '8px 12px', background: bg, color, fontSize: 17, fontWeight: 700, ...style}}>{children}</div>;

const Card: React.FC<React.PropsWithChildren<{x: number; y: number; w: number; h: number; order: number; style?: React.CSSProperties}>> = ({x,y,w,h,order,style,children}) => {
  const f = useCurrentFrame();
  const p = progress(f, order * 3, 28);
  return <section style={{position:'absolute',left:x,top:y,width:w,height:h,
    padding:24,boxSizing:'border-box',borderRadius:24,background:'#FFFFFF',
    border:'1px solid rgba(255,255,255,0.9)',boxShadow:'0 8px 25px rgba(64,46,94,0.045)',
    opacity:0.35 + 0.65*p,transform:`translateY(${10*(1-p)}px)`,...style}}>{children}</section>;
};

const Legend: React.FC<{label: string; color: string; style?: React.CSSProperties}> = ({label,color,style}) =>
  <div style={{display:'flex',alignItems:'center',gap:8,fontSize:16,fontWeight:600,color:C.muted,...style}}>
    <span style={{width:9,height:9,borderRadius:3,background:color}}/>{label}
  </div>;

const Chip: React.FC = () => {
  const f = useCurrentFrame();
  const p = progress(f, 10, 100);
  return <svg width="134" height="134" viewBox="0 0 160 160" style={{overflow:'visible',transform:`rotate(${-8+3*p}deg)`}}>
    <defs>
      <linearGradient id="chip" x1="0" y1="0" x2="1" y2="1"><stop stopColor="#C0A2FF"/><stop offset=".5" stopColor="#8F67DC"/><stop offset="1" stopColor="#624396"/></linearGradient>
      <linearGradient id="chipFace" x1="0" y1="0" x2="1" y2="1"><stop stopColor="#FAF8FF"/><stop offset="1" stopColor="#DDD0F6"/></linearGradient>
      <filter id="chipShadow"><feDropShadow dx="0" dy="10" stdDeviation="8" floodColor="#705697" floodOpacity=".18"/></filter>
    </defs>
    {Array.from({length:7},(_,i)=> <g key={i} fill="#CEC3E0">
      <rect x={40+i*13} y="12" width="6" height="20" rx="2"/>
      <rect x={40+i*13} y="128" width="6" height="20" rx="2"/>
      <rect x="12" y={40+i*13} width="20" height="6" rx="2"/>
      <rect x="128" y={40+i*13} width="20" height="6" rx="2"/>
    </g>)}
    <rect x="26" y="29" width="108" height="110" rx="23" fill="#664593" filter="url(#chipShadow)"/>
    <rect x="26" y="23" width="108" height="108" rx="23" fill="url(#chip)" stroke="#E5D6FC" strokeWidth="2"/>
    <rect x="39" y="36" width="82" height="81" rx="15" fill="url(#chipFace)"/>
    <text x="80" y="83" textAnchor="middle" fill="#715195" fontSize="29" fontWeight="800">M5</text>
    <text x="80" y="102" textAnchor="middle" fill="#9174B7" fontSize="11" letterSpacing="3" fontWeight="700">PRO</text>
    <path d="M42 41H112" stroke="white" strokeWidth="2" opacity=".6"/>
  </svg>;
};

const Goodput: React.FC = () => {
  const f = useCurrentFrame();
  const p = progress(f, 45, 110);
  const values = [data.paired_result.fixed_mean_goodput, data.paired_result.adaptive_mean_goodput];
  return <svg width="474" height="221" viewBox="0 0 474 221" style={{display:'block',overflow:'visible'}}>
    <defs>
      <linearGradient id="purpleBar" x1="0" y1="0" x2="1" y2=".3"><stop stopColor="#B191F4"/><stop offset=".55" stopColor="#966FE4"/><stop offset="1" stopColor="#8055D0"/></linearGradient>
      <linearGradient id="goldBar" x1="0" y1="0" x2="1" y2=".3"><stop stopColor="#F7D775"/><stop offset=".55" stopColor="#F0C354"/><stop offset="1" stopColor="#D9A32A"/></linearGradient>
      <filter id="barShadow" x="-.5" y="-.5" width="2" height="2"><feDropShadow dx="0" dy="6" stdDeviation="5" floodColor="#503572" floodOpacity=".10"/></filter>
    </defs>
    {[0,10,20,30,40].map(v=> <g key={v}>
      <line x1="38" y1={172-v*3.05} x2="465" y2={172-v*3.05} stroke={C.line} strokeWidth="1"/>
      <text x="26" y={178-v*3.05} textAnchor="end" fontSize="13" fill="#A49DB2">{v}</text>
    </g>)}
    {values.map((v,i)=>{
      const x=109+i*204, h=v*3.05*p, y=172-h, color=i?'#FFDF83':'#CCB3FF',side=i?'#C9972B':'#764DBD';
      return <g key={i}>
        <text x={x+44} y={y-24} textAnchor="middle" fill={C.ink} fontSize="31" fontWeight="800">{fix(v)}</text>
        <g filter="url(#barShadow)" opacity={p}>
          <path d={`M${x} ${y}h90v${h}h-90Z`} fill={`url(#${i?'goldBar':'purpleBar'})`}/>
          <path d={`M${x} ${y}l11 -9h90l-11 9Z`} fill={color}/>
          <path d={`M${x+90} ${y}l11 -9v${h}l-11 9Z`} fill={side}/>
          <path d={`M${x+2} ${y+2}h86`} stroke="white" strokeWidth="2" opacity=".45"/>
        </g>
        <text x={x+48} y="205" textAnchor="middle" fill={C.muted} fontSize="17" fontWeight="600">{i?'Adaptive':'Fixed'}</text>
      </g>;
    })}
  </svg>;
};

const Interval: React.FC = () => {
  const f=useCurrentFrame();
  const p=progress(f,300,65);
  const map=(v:number)=>44+(v+1)/4.5*347;
  const [lo,hi]=data.paired_result.difference_95ci;
  const mid=data.paired_result.difference_goodput;
  return <svg width="428" height="110" viewBox="0 0 428 110">
    <rect x="20" y="6" width="389" height="82" rx="14" fill="#F8F6FC"/>
    <line x1={map(0)} y1="13" x2={map(0)} y2="83" stroke="#BDB2CE" strokeDasharray="4 4"/>
    <line x1={map(lo)} y1="44" x2={map(hi)} y2="44" stroke="#DCD0F0" strokeWidth="7" strokeLinecap="round"/>
    <line x1={map(lo)} y1="44" x2={map(lo)+(map(hi)-map(lo))*p} y2="44" stroke={C.purple} strokeWidth="7" strokeLinecap="round"/>
    {[lo,hi].map((v,i)=><g key={i}><line x1={map(v)} y1="35" x2={map(v)} y2="53" stroke={C.purpleDark} strokeWidth="2"/><text x={map(v)} y="75" fill={C.muted} fontSize="16" textAnchor="middle">{v>0?'+':''}{fix(v)}</text></g>)}
    <circle cx={map(mid)} cy="44" r="8" fill={C.purpleDark} stroke="white" strokeWidth="3"/>
    <text x={map(0)} y="105" textAnchor="middle" fill="#A69BB5" fontSize="14">0</text>
  </svg>;
};

const Rail: React.FC<{amount:number;max:number;color:string;p:number;height?:number}> = ({amount,max,color,p,height=11}) =>
  <div style={{height,background:'#F0EDF5',borderRadius:7,overflow:'hidden'}}>
    <div style={{height:'100%',width:`${100*amount/max*p}%`,borderRadius:7,background:`linear-gradient(90deg, ${color}BB, ${color})`}}/>
  </div>;

const LatencyRow: React.FC<{label:string;values:number[];max:number;start:number}> = ({label,values,max,start}) => {
  const f=useCurrentFrame();const p=progress(f,start,65);
  return <div style={{marginTop:15}}>
    <div style={{display:'flex',justifyContent:'space-between',alignItems:'baseline',marginBottom:8}}>
      <div style={{fontSize:18,fontWeight:600}}>{label}</div>
      <div style={{fontSize:21,fontWeight:700,fontVariantNumeric:'tabular-nums'}}>
        <span style={{color:C.purpleDark}}>{fix(values[0])}</span>
        <span style={{color:'#B9B2C6',padding:'0 9px'}}>→</span>
        <span style={{color:C.goldDark}}>{fix(values[1])} s</span>
      </div>
    </div>
    <Rail amount={values[0]} max={max} p={p} color={C.purple}/>
    <div style={{height:5}}/>
    <Rail amount={values[1]} max={max} p={p} color={C.gold}/>
  </div>;
};

export const Dashboard: React.FC = () => {
  const [readyHandle] = React.useState(() => delayRender('Wait for fonts and initial compositor paint'));
  React.useEffect(() => {
    let timer: ReturnType<typeof setTimeout>;
    document.fonts.ready.then(() => {
      // Allow the renderer's resized viewport and composited layers to settle.
      timer = setTimeout(() => continueRender(readyHandle), 700);
    });
    return () => clearTimeout(timer);
  }, [readyHandle]);
  const f=useCurrentFrame();
  const filled=progress(f,185,65);
  const kv=progress(f,285,65);
  const reveal=progress(f,325,40);
  const titleP=progress(f,0,28);
  const short=data.solo_short;
  return <AbsoluteFill style={{background:'#F2F0F7',color:C.ink,fontFamily:'Inter, Arial, sans-serif',fontVariantNumeric:'tabular-nums'}}>
    <div style={{position:'absolute',inset:0,background:'radial-gradient(ellipse at 82% 0%, #E9E2F7 0%, transparent 55%)'}}/>
    <div style={{position:'absolute',left:34,top:28,right:34,opacity:.65+.35*titleP}}>
      <div style={{display:'flex',alignItems:'center',gap:10}}>
        <span style={{width:10,height:10,background:C.purple,borderRadius:3,transform:'rotate(-12deg)'}}/>
        <Label>Mac Inference Lab</Label>
        <span style={{fontSize:14,color:'#A29AAE',paddingLeft:10}}>EXPERIMENT 001</span>
        <div style={{marginLeft:'auto',fontSize:15,fontWeight:600,color:'#8D819F'}}>MEASURED ON APPLE SILICON</div>
      </div>
      <h1 style={{fontSize:46,letterSpacing:-1.8,lineHeight:1.15,fontWeight:800,margin:'14px 0 10px'}}>Which inference settings actually help on a Mac?</h1>
      <div style={{fontSize:20,color:C.muted}}>{data.hardware.chip} · {data.hardware.memory_gib} GiB · {data.model} · {data.weight_bits}-bit weights · MLX-LM</div>
    </div>

    <Card x={32} y={174} w={340} h={342} order={0}>
      <Label>The test bench</Label>
      <div style={{display:'flex',alignItems:'center',justifyContent:'space-between',margin:'16px 0 6px'}}>
        <div><div style={{fontSize:39,fontWeight:800,letterSpacing:-1.6}}>{data.hardware.chip}</div><div style={{fontSize:21,color:C.muted,marginTop:5}}>{data.hardware.memory_gib} GiB memory</div></div>
        <Chip/>
      </div>
      <div style={{height:1,background:C.line,margin:'8px 0 19px'}}/>
      <div style={{display:'flex',gap:30}}>
        <div><div style={{fontSize:35,fontWeight:800}}>{data.configs}</div><Note style={{fontSize:15}}>configurations</Note></div>
        <div><div style={{fontSize:35,fontWeight:800}}>{data.total_model_requests}</div><Note style={{fontSize:15}}>model requests</Note></div>
      </div>
      <Note style={{fontSize:15,marginTop:14}}>Full pilot · warm, resident models</Note>
    </Card>

    <Card x={388} y={174} w={524} h={342} order={1}>
      <div style={{display:'flex',justifyContent:'space-between',alignItems:'center'}}>
        <Label>Held-out goodput</Label>
        <Pill color="#8F6A1D" bg="#FFF6D9">+{data.paired_result.relative_change_percent.toFixed(1)}% observed</Pill>
      </div>
      <Note style={{marginTop:7}}>Correct + on-time responses per minute</Note>
      <Goodput/>
      <Note style={{fontSize:15,marginTop:0}}>Balanced mean across {data.paired_result.blocks} paired workload blocks</Note>
    </Card>

    <Card x={928} y={174} w={480} h={342} order={2}>
      <Label>How strong is the evidence?</Label>
      <div style={{marginTop:17,fontSize:37,fontWeight:800,letterSpacing:-1.2}}>Inconclusive.</div>
      <Note style={{fontSize:19,marginTop:5,color:'#625773'}}>The 95% interval includes zero.</Note>
      <div style={{marginTop:16}}><Interval/></div>
      <div style={{display:'flex',alignItems:'baseline',gap:10,marginTop:3}}>
        <span style={{fontSize:25,fontWeight:800,color:C.purpleDark}}>+{fix(data.paired_result.difference_goodput)}</span>
        <span style={{fontSize:17,color:C.muted}}>responses/min difference</span>
      </div>
      <Note style={{fontSize:15,marginTop:7}}>Paired bootstrap · overall gain not established</Note>
    </Card>

    <Card x={32} y={532} w={340} h={286} order={3}>
      <Label>Same test accuracy</Label>
      <div style={{display:'flex',alignItems:'baseline',gap:8,margin:'16px 0 4px'}}>
        <span style={{fontSize:51,fontWeight:800,letterSpacing:-2}}>{data.correct.fixed}</span>
        <span style={{fontSize:28,color:'#A49BAF'}}>/ {data.test_unique_prompts}</span>
        <span style={{fontSize:16,color:C.muted,marginLeft:5}}>each</span>
      </div>
      <div style={{display:'flex',justifyContent:'space-between',margin:'14px 0 7px'}}><Legend label="Fixed" color={C.purple}/><span style={{fontSize:15,color:C.muted}}>96.25%</span></div>
      <Rail amount={data.correct.fixed} max={data.test_unique_prompts} color={C.purple} p={filled}/>
      <div style={{display:'flex',justifyContent:'space-between',margin:'12px 0 7px'}}><Legend label="Adaptive" color={C.gold}/><span style={{fontSize:15,color:C.muted}}>96.25%</span></div>
      <Rail amount={data.correct.adaptive} max={data.test_unique_prompts} color={C.gold} p={filled}/>
      <Note style={{fontSize:15,marginTop:16}}>{data.test_unique_prompts} unique prompts · {data.test_executions} executions</Note>
    </Card>

    <Card x={388} y={532} w={524} h={286} order={4}>
      <Label>The short-request tradeoff</Label>
      <div style={{fontSize:23,fontWeight:700,letterSpacing:-.5,marginTop:10}}>Finish sooner. First token later.</div>
      <LatencyRow label="Completion" values={[short.fixed.mean_completion_s,short.adaptive.mean_completion_s]} max={2} start={185}/>
      <LatencyRow label="First token" values={[short.fixed.mean_ttft_s,short.adaptive.mean_ttft_s]} max={0.5} start={220}/>
      <div style={{display:'flex',gap:18,alignItems:'center',marginTop:16}}>
        <Legend label="Fixed" color={C.purple}/><Legend label="Adaptive" color={C.gold}/>
        <Note style={{fontSize:14,marginLeft:'auto'}}>{short.fixed.requests} paired short cases</Note>
      </div>
    </Card>

    <Card x={928} y={532} w={480} h={286} order={5}>
      <Label>KV precision changed quality</Label>
      <Note style={{fontSize:18,marginTop:10}}>Exact answers · calibration tasks</Note>
      <div style={{display:'flex',gap:28,marginTop:20,alignItems:'flex-start'}}>
        <div style={{flex:1}}>
          <div style={{fontSize:38,fontWeight:800,letterSpacing:-1,color:C.purpleDark}}>{data.calibration.serial.correct}<span style={{fontSize:23,fontWeight:500,color:'#B7A8CB'}}>/{data.calibration.serial.requests}</span></div>
          <div style={{margin:'8px 0 10px',fontSize:16,color:C.muted}}>Ordinary KV</div>
          <Rail amount={data.calibration.serial.correct} max={30} color={C.purple} p={kv} height={12}/>
        </div>
        <div style={{flex:1}}>
          <div style={{fontSize:38,fontWeight:800,letterSpacing:-1,color:C.roseDark}}>{data.calibration.kv4.correct}<span style={{fontSize:23,fontWeight:500,color:'#C6A5B2'}}>/{data.calibration.kv4.requests}</span></div>
          <div style={{margin:'8px 0 10px',fontSize:16,color:C.muted}}>4-bit KV · start 0</div>
          <Rail amount={0} max={30} color={C.rose} p={kv} height={12}/>
        </div>
      </div>
      <Pill color={C.roseDark} bg="#FDEDF2" style={{fontSize:15,marginTop:17,opacity:.5+.5*reveal}}>Excluded by the quality gate</Pill>
      <Note style={{fontSize:14,marginTop:9}}>This model + runtime + task set; not a universal result.</Note>
    </Card>

    <Card x={32} y={834} w={880} h={166} order={6}>
      <Label>What the controller actually chose</Label>
      <div style={{display:'flex',gap:22,marginTop:16}}>
        <div style={{flex:1,borderRadius:13,background:'#F6F2FC',padding:'14px 16px',display:'flex',alignItems:'center',gap:16}}>
          <div style={{width:33,height:33,borderRadius:10,background:'#E9DDFB',color:C.purpleDark,display:'grid',placeItems:'center',fontSize:19}}>1</div>
          <div><div style={{fontSize:17,fontWeight:600}}>Single + short</div><Note style={{fontSize:16,marginTop:4}}>→ 2-token speculation</Note></div>
        </div>
        <div style={{flex:1,borderRadius:13,background:'#FAF7EF',padding:'14px 16px',display:'flex',alignItems:'center',gap:16}}>
          <div style={{width:33,height:33,borderRadius:10,background:'#F4E8C4',color:C.goldDark,display:'grid',placeItems:'center',fontSize:19}}>3</div>
          <div><div style={{fontSize:17,fontWeight:600}}>Other calibrated states</div><Note style={{fontSize:16,marginTop:4}}>→ batch path · 256-token chunks</Note></div>
        </div>
      </div>
    </Card>

    <Card x={928} y={834} w={480} h={166} order={7} style={{background:'#292238',borderColor:'#392F49',color:'#F7F3FD',boxShadow:'0 9px 23px #35284018'}}>
      <Label style={{color:'#B8A4D8'}}>Open code. Raw traces. Real limits.</Label>
      <div style={{fontSize:25,fontWeight:700,letterSpacing:-.6,marginTop:16}}>mac-inference-lab</div>
      <div style={{fontSize:16,color:'#D3C7E4',marginTop:7}}>github.com/zeeshan8281</div>
      <div style={{position:'absolute',right:25,bottom:25,fontSize:25,color:'#D3BDF3'}}>↗</div>
    </Card>

    <div style={{position:'absolute',left:36,right:36,top:1023,display:'flex',justifyContent:'space-between',alignItems:'center'}}>
      <span style={{fontSize:16,color:'#8C819E'}}>Controlled extraction pilot · finite bursts · warm models · no general speedup claim</span>
      <span style={{fontSize:14,fontWeight:600,color:'#A093B0'}}>PILOT-002 / MLX-LM {data.hardware.mlx_lm}</span>
    </div>
    <div style={{position:'absolute',bottom:0,left:0,height:3,width:`${100*Math.min(f/390,1)}%`,background:'linear-gradient(90deg,#B191F4,#EAB8CE,#EBD084)',opacity:.8}}/>
  </AbsoluteFill>;
};
