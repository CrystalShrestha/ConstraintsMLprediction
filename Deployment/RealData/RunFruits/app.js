// // One-page runner for: apple_reg, apple_clf, banana_clf
// let assets = {
//   apple: null, // {feature_order, mean[], scale[], classes?}
//   banana: null,
// };
// let sessions = {
//   apple_reg: null,
//   apple_clf: null,
//   banana_clf: null,
// };
// let current = "apple_reg";

// async function getJSON(url) {
//   const r = await fetch(url);
//   if (!r.ok) throw new Error(url);
//   return r.json();
// }

// function buildInputs(cols) {
//   const root = document.getElementById("inputs");
//   root.innerHTML = "";
//   cols.forEach((c) => {
//     const div = document.createElement("div");
//     div.className = "inputbox";
//     div.innerHTML = `<label>${c}</label><input id="f_${c}" type="number" step="any" value="0">`;
//     root.appendChild(div);
//   });
// }

// function readStandardized(prep) {
//   const cols = prep.feature_order;
//   const x = new Float32Array(cols.length);
//   for (let i = 0; i < cols.length; i++) {
//     const el = document.getElementById("f_" + cols[i]);
//     const v = parseFloat(el.value || "0");
//     x[i] = (v - prep.mean[i]) / prep.scale[i];
//   }
//   return x;
// }

// function toggleThreshold() {
//   const thrwrap = document.getElementById("thrwrap");
//   thrwrap.style.display = current.endsWith("_clf") ? "inline-flex" : "none";
// }

// async function predict() {
//   const out = document.getElementById("out");
//   out.textContent = "Running…";

//   if (current === "apple_reg") {
//     const prep = assets.apple;
//     const x = readStandardized(prep);
//     const feeds = { input: new ort.Tensor("float32", x, [1, x.length]) };
//     const res = await sessions.apple_reg.run(feeds);
//     const y = res.weight_pred.data[0];
//     out.textContent = `Predicted Weight = ${y.toFixed(4)}`;
//     return;
//   }

//   // Classification (apple_clf or banana_clf)
//   const isApple = current === "apple_clf";
//   const prep = isApple ? assets.apple : assets.banana;
//   const sess = isApple ? sessions.apple_clf : sessions.banana_clf;

//   const x = readStandardized(prep);
//   const feeds = { input: new ort.Tensor("float32", x, [1, x.length]) };
//   const res = await sess.run(feeds);
//   const logit = res.logits.data[0];
//   const prob = 1 / (1 + Math.exp(-logit));
//   const thr = parseFloat(document.getElementById("thr").value);
//   const idx = prob >= thr ? 1 : 0;
//   const name =
//     prep.classes && prep.classes.length === 2
//       ? prep.classes[idx]
//       : idx
//       ? "positive"
//       : "negative";
//   const positiveName = prep.classes?.[1] ?? "positive";
//   out.textContent = `P(${positiveName}) = ${prob.toFixed(
//     3
//   )}  |  threshold=${thr.toFixed(2)}  →  Predicted: ${name}`;
// }

// function resetInputs() {
//   const cols = current.startsWith("apple")
//     ? assets.apple.feature_order
//     : assets.banana.feature_order;
//   cols.forEach((c) => {
//     document.getElementById("f_" + c).value = 0;
//   });
//   document.getElementById("out").textContent = "Cleared inputs.";
// }

// async function handleTaskChange() {
//   const sel = document.getElementById("task");
//   current = sel.value;

//   const prep = current.startsWith("apple") ? assets.apple : assets.banana;
//   buildInputs(prep.feature_order);
//   toggleThreshold();

//   // Restore threshold label display
//   const thr = document.getElementById("thr"),
//     thv = document.getElementById("thv");
//   thv.textContent = parseFloat(thr.value).toFixed(2);
//   document.getElementById("out").textContent = "Ready.";
// }

// async function main() {
//   // Load assets
//   assets.apple = await getJSON("assets/apple.json");
//   assets.banana = await getJSON("assets/banana.json");

//   // Create sessions
//   sessions.apple_reg = await ort.InferenceSession.create(
//     "models/apple_weight_regressor.onnx"
//   );
//   sessions.apple_clf = await ort.InferenceSession.create(
//     "models/apple_quality_classifier.onnx"
//   );
//   sessions.banana_clf = await ort.InferenceSession.create(
//     "models/banana_quality_classifier.onnx"
//   );

//   // Build default inputs (apple_reg)
//   buildInputs(assets.apple.feature_order);
//   toggleThreshold();

//   // Wire UI
//   document.getElementById("task").onchange = handleTaskChange;
//   document.getElementById("btnPredict").onclick = predict;
//   document.getElementById("btnReset").onclick = resetInputs;

//   const thr = document.getElementById("thr"),
//     thv = document.getElementById("thv");
//   thr.oninput = () => (thv.textContent = parseFloat(thr.value).toFixed(2));
//   thv.textContent = parseFloat(thr.value).toFixed(2);

//   document.getElementById("out").textContent =
//     "Models loaded. Enter features and click Predict.";
// }
// main();
let sessions = { apple_reg: null, banana_clf: null };
let io = { apple_reg: {}, banana_clf: {} }; // {inputName, outputName}
let current = "apple_reg";

/* === EDIT THESE to match training feature order & z-score stats === */
const order = {
  apple_reg: [
    "Size",
    "Sweetness",
    "Crunchiness",
    "Juiciness",
    "Ripeness",
    "Acidity",
  ],
  banana_clf: ["Diameter", "Length", "Firmness", "Sugar", "Acidity"],
};
const stats = {
  apple_reg: {
    mean: [6.12, 6.98, 5.41, 5.88, 3.02, 3.5],
    scale: [1.15, 1.2, 1.0, 1.05, 0.9, 0.4],
  },
  banana_clf: {
    mean: [3.6, 18.2, 42.0, 15.5, 4.2],
    scale: [0.4, 2.1, 8.0, 3.0, 0.5],
    classes: ["bad", "good"],
  },
};
/* ================================================================ */

function byId(id) {
  return document.getElementById(id);
}

function activeForm() {
  return current === "apple_reg" ? byId("form_apple") : byId("form_banana");
}

function validateForm() {
  const ok = activeForm().checkValidity();
  byId("btnPredict").disabled = !ok;
}

function readVector() {
  const form = activeForm();
  const names = order[current];
  const x = new Float32Array(names.length);

  // read values by data-name
  const inputs = Array.from(form.querySelectorAll("input[type='number']"));
  const map = {};
  for (const inp of inputs) {
    const key = inp.getAttribute("data-name");
    map[key] = parseFloat(inp.value);
  }

  // standardize
  const mu = stats[current].mean,
    sc = stats[current].scale;
  if (names.length !== mu.length || mu.length !== sc.length) {
    throw new Error(
      `Length mismatch: order=${names.length}, mean=${mu.length}, scale=${sc.length}`
    );
  }
  for (let i = 0; i < names.length; i++) {
    const v = map[names[i]];
    if (!Number.isFinite(v))
      throw new Error(`Non-numeric value for "${names[i]}"`);
    x[i] = (v - mu[i]) / sc[i];
  }
  return x;
}

function toggleTask() {
  current = byId("task").value;
  byId("form_apple").style.display = current === "apple_reg" ? "" : "none";
  byId("form_banana").style.display = current === "banana_clf" ? "" : "none";
  byId("thrwrap").style.display =
    current === "banana_clf" ? "inline-flex" : "none";
  byId("out").textContent = "Ready.";
  validateForm();
}

async function predict() {
  const out = byId("out");
  out.textContent = "Running…";
  try {
    const x = readVector();

    // use model's real input name
    const inputName = io[current].inputName;
    if (!inputName)
      throw new Error("Model not loaded yet (missing inputName).");

    const feeds = {};
    feeds[inputName] = new ort.Tensor("float32", x, [1, x.length]);

    const session = sessions[current];
    const res = await session.run(feeds);

    // take first output if we don't have a chosen name
    const outName = io[current].outputName || Object.keys(res)[0];
    const data = res[outName]?.data;
    if (!data || data.length === 0)
      throw new Error(`No output data for "${outName}"`);

    if (current === "apple_reg") {
      const y = data[0]; // numeric regression
      out.textContent = `Predicted Weight = ${(+y).toFixed(4)}`;
    } else {
      const logit = data[0]; // single logit
      const p = 1 / (1 + Math.exp(-logit)); // sigmoid
      const thr = parseFloat(byId("thr").value);
      const idx = p >= thr ? 1 : 0;
      const names = stats.banana_clf.classes || ["negative", "positive"];
      out.textContent = `P(${names[1]}) = ${p.toFixed(
        3
      )} | threshold=${thr.toFixed(2)} → Predicted: ${names[idx]}`;
    }
  } catch (err) {
    console.error(err);
    out.textContent = `Error: ${err.message}`;
  }
}

function resetForm() {
  activeForm().reset();
  byId("out").textContent = "Cleared inputs.";
  validateForm();
}

async function main() {
  try {
    byId("btnPredict").disabled = true;

    // load sessions
    sessions.apple_reg = await ort.InferenceSession.create(
      "models/apple_weight_regressor.onnx"
    );
    sessions.banana_clf = await ort.InferenceSession.create(
      "models/banana_quality_classifier.onnx"
    );

    // discover real IO names
    io.apple_reg = {
      inputName: sessions.apple_reg.inputNames?.[0],
      outputName: sessions.apple_reg.outputNames?.[0],
    };
    io.banana_clf = {
      inputName: sessions.banana_clf.inputNames?.[0],
      outputName: sessions.banana_clf.outputNames?.[0],
    };

    // log them for debugging
    console.log("apple_reg IO:", io.apple_reg);
    console.log("banana_clf IO:", io.banana_clf);

    // quick sanity: feature count must match model input
    const expectedAppleIn =
      sessions.apple_reg.session?.inputNamesLength || order.apple_reg.length;
    if (order.apple_reg.length !== order.apple_reg.length) {
      /* no-op, placeholder for custom checks */
    }

    // wire events
    byId("task").onchange = toggleTask;
    byId("btnPredict").onclick = predict;
    byId("btnReset").onclick = resetForm;
    for (const id of ["form_apple", "form_banana"]) {
      byId(id).addEventListener("input", validateForm);
    }
    const thr = byId("thr"),
      thv = byId("thv");
    thr.oninput = () => (thv.textContent = parseFloat(thr.value).toFixed(2));
    thv.textContent = parseFloat(thr.value).toFixed(2);

    toggleTask();
    byId("out").textContent = "Models loaded. Enter values and click Predict.";
  } catch (err) {
    console.error(err);
    byId("out").textContent = `Error loading models: ${err.message}`;
  }
}
main();
