let prep=null, clf=null, reg=null;
async function loadJSON(u){const r=await fetch(u); if(!r.ok) throw new Error(u); return r.json();}
function makeInputs(cols){const el=document.getElementById('inputs'); el.innerHTML=''; cols.forEach(c=>{
  const d=document.createElement('div'); d.innerHTML=`${c}: <input id="f_${c}" type="number" step="any" value="0">`; el.appendChild(d);
});}
function getX(){const x=new Float32Array(prep.feature_order.length);
  for(let i=0;i<prep.feature_order.length;i++){const k=prep.feature_order[i];
    const v=parseFloat(document.getElementById('f_'+k).value||'0'); x[i]=(v-prep.mean[i])/prep.scale[i];}
  return x;}
async function run(){const out=document.getElementById('out'); out.textContent='Running…';
  const x=getX(); const input={'input': new ort.Tensor('float32', x, [1,x.length])};
  const task=document.querySelector('input[name="task"]:checked').value;
  if(task==='clf'){const r=await clf.run(input); const logit=r.logits.data[0]; const p=1/(1+Math.exp(-logit));
    const thr=parseFloat(document.getElementById('thr').value); const idx=p>=thr?1:0; const name=(prep.classes||['bad','good'])[idx];
    out.textContent=`P(${(prep.classes||['bad','good'])[1]})=${p.toFixed(3)} | threshold=${thr.toFixed(2)} → ${name}`;
  } else {const r=await reg.run(input); out.textContent=`Predicted Weight = ${r.weight_pred.data[0].toFixed(4)}`;}}
(async()=>{
  prep=await loadJSON('assets/preprocessing.json'); makeInputs(prep.feature_order);
  clf=await ort.InferenceSession.create('models/apple_quality_classifier.onnx');
  reg=await ort.InferenceSession.create('models/apple_weight_regressor.onnx');
  document.getElementById('btnPredict').onclick=run;
  const thr=document.getElementById('thr'), thv=document.getElementById('thv'); thr.oninput=()=>thv.textContent=parseFloat(thr.value).toFixed(2);
  document.getElementById('out').textContent='Models loaded. Enter features and click Predict.';
})();