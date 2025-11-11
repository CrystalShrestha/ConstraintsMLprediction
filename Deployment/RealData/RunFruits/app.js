let sessions = { apple_reg: null, banana_clf: null };
let io = { apple_reg: {}, banana_clf: {} };
let current = "apple_reg";

/* === EDIT THESE when you know the exact banana 6 features & stats ===
   Keep names in the SAME ORDER you used during training. */
const order = {
  apple_reg: [
    "Size",
    "Sweetness",
    "Crunchiness",
    "Juiciness",
    "Ripeness",
    "Acidity",
  ],
  // Temporary: include a placeholder 6th name ("Feature6") to match model's 6 inputs
  banana_clf: [
    "Diameter",
    "Length",
    "Firmness",
    "Sugar",
    "Acidity",
    "HarvestTime",
  ],
};

const stats = {
  apple_reg: {
    mean: [6.12, 6.98, 5.41, 5.88, 3.02, 3.5],
    scale: [1.15, 1.2, 1.0, 1.05, 0.9, 0.4],
  },
  // For banana, if these don't match length 6 or exact order, code will use identity scaling (mean=0, scale=1)
  banana_clf: {
    mean: [3.6, 18.2, 42.0, 15.5, 4.2, 1.0],
    scale: [0.4, 2.1, 8.0, 3.0, 0.5, 1.0],
    classes: ["bad", "good"],
  },
};
/* ==================================================================== */

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

function getUINamesAndMap() {
  const form = activeForm();
  const inputs = Array.from(form.querySelectorAll("input[type='number']"));
  const uiNames = inputs.map((i) => i.getAttribute("data-name"));
  const map = {};
  for (const inp of inputs) {
    const k = inp.getAttribute("data-name");
    map[k] = Number(inp.value);
  }
  return { uiNames, map };
}

function getExpectedLenSafe(session, task) {
  // Default to declared order length
  let expected = (order[task] && order[task].length) || 0;
  try {
    const meta = session.inputMetadata || {};
    const inName = Object.keys(meta)[0] || session.inputNames?.[0];
    const dims = meta[inName]?.dimensions;
    if (Array.isArray(dims)) {
      const last = dims[dims.length - 1];
      if (typeof last === "number" && Number.isFinite(last)) expected = last;
    }
  } catch (e) {
    console.warn(
      `[${task}] Could not read inputMetadata; falling back to order length.`,
      e
    );
  }
  if (!Number.isFinite(expected) || expected <= 0) {
    throw new Error(
      `[${task}] Could not determine expected feature length. Check your model and order.${task}.`
    );
  }
  return expected;
}

function buildVector(task, expectedLen) {
  const namesDeclared = (order[task] || []).slice(); // copy
  const { uiNames, map } = getUINamesAndMap();

  const notes = [];
  if (expectedLen !== namesDeclared.length) {
    notes.push(
      `[${task}] Model expects ${expectedLen} features but order.${task} has ${namesDeclared.length}. Padded/truncated.`
    );
  }
  if (expectedLen !== uiNames.length) {
    const missing = namesDeclared.filter((n) => !uiNames.includes(n));
    const extra = uiNames.filter((n) => !namesDeclared.includes(n));
    notes.push(
      `[${task}] UI has ${uiNames.length} inputs. Missing in UI: ${
        missing.join(", ") || "unknown"
      }; Extra in UI: ${extra.join(", ") || "none"}. Padded/truncated.`
    );
  }

  // Stats or identity if mismatch
  let mu = stats[task]?.mean || [];
  let sc = stats[task]?.scale || [];
  if (mu.length !== expectedLen || sc.length !== expectedLen) {
    notes.push(
      `[${task}] Stats length mismatch (mean=${mu.length}, scale=${sc.length}, expected=${expectedLen}). Using identity scaling.`
    );
    mu = Array(expectedLen).fill(0);
    sc = Array(expectedLen).fill(1);
  }

  // Build final name list to expectedLen
  const names = namesDeclared.slice(0, expectedLen);
  while (names.length < expectedLen) names.push(`__pad_${names.length}__`);

  // Build standardized vector
  const x = new Float32Array(expectedLen);
  for (let i = 0; i < expectedLen; i++) {
    const name = names[i];
    const raw = name in map ? map[name] : 0; // missing -> 0
    if (!Number.isFinite(raw))
      throw new Error(`Non-numeric value for "${name}"`);
    x[i] = (raw - mu[i]) / sc[i];
  }
  return { x, notes };
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
    const session =
      current === "apple_reg" ? sessions.apple_reg : sessions.banana_clf;

    if (!io[current].inputName) {
      io[current].inputName = session.inputNames?.[0];
      io[current].outputName = session.outputNames?.[0];
    }
    const inputName = io[current].inputName;
    if (!inputName) throw new Error("Could not resolve model input name.");

    // Determine expected feature length robustly
    const expectedLen = getExpectedLenSafe(session, current);

    // Build vector (pads/truncates, identity scaling if needed)
    const { x, notes } = buildVector(current, expectedLen);

    const feeds = {};
    feeds[inputName] = new ort.Tensor("float32", x, [1, x.length]);
    const res = await session.run(feeds);

    const outName = io[current].outputName || Object.keys(res)[0];
    const data = res[outName]?.data;
    if (!data || data.length === 0)
      throw new Error(`No output data for "${outName}"`);

    if (current === "apple_reg") {
      const y = Math.max(0, Number(data[0])); // clamp display at 0
      out.textContent = `Predicted Weight = ${y.toFixed(4)}`;
    } else {
      const logit = Number(data[0]);
      const p = 1 / (1 + Math.exp(-logit));
      const thr = parseFloat(byId("thr").value);
      const idx = p >= thr ? 1 : 0;
      const names = stats.banana_clf.classes || ["negative", "positive"];
      out.textContent = `P(${names[1]}) = ${p.toFixed(
        3
      )} | threshold=${thr.toFixed(2)} → Predicted: ${names[idx]}`;
    }

    if (notes.length) {
      out.textContent += `\n\nNote:\n- ` + notes.join("\n- ");
      console.warn(...notes);
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

    // Load sessions
    sessions.apple_reg = await ort.InferenceSession.create(
      "models/apple_weight_regressor.onnx"
    );
    sessions.banana_clf = await ort.InferenceSession.create(
      "models/banana_quality_classifier.onnx"
    );

    // IO names
    io.apple_reg = {
      inputName: sessions.apple_reg.inputNames?.[0],
      outputName: sessions.apple_reg.outputNames?.[0],
    };
    io.banana_clf = {
      inputName: sessions.banana_clf.inputNames?.[0],
      outputName: sessions.banana_clf.outputNames?.[0],
    };
    console.log("apple_reg IO:", io.apple_reg);
    console.log("banana_clf IO:", io.banana_clf);

    // Wire UI
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
