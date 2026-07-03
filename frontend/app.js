// ------------------------------------------------------------------ //
// DermaScan frontend logic
// ------------------------------------------------------------------ //
const $ = (id) => document.getElementById(id);

const fileInput = $("fileInput");
const dropzone = $("dropzone");
const previewHolder = $("previewHolder");
const previewImg = $("previewImg");
const dzEmpty = $("dzEmpty");
const clearBtn = $("clearBtn");
const analyzeBtn = $("analyzeBtn");

const resultEmpty = $("resultEmpty");
const resultBody = $("resultBody");
const resultError = $("resultError");
const verdictLabel = $("verdictLabel");
const gaugeFill = $("gaugeFill");
const confNum = $("confNum");
const benignBar = $("benignBar");
const malignantBar = $("malignantBar");
const benignVal = $("benignVal");
const malignantVal = $("malignantVal");
const modelStatus = $("modelStatus");

const GAUGE_CIRC = 2 * Math.PI * 52; // r = 52
let selectedFile = null;

// ---- Model status ------------------------------------------------- //
async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    if (data.model_loaded) {
      modelStatus.className = "status-pill ok";
      modelStatus.innerHTML = `<span class="dot"></span>Model ready · ${data.device.toUpperCase()}`;
    } else {
      modelStatus.className = "status-pill bad";
      modelStatus.innerHTML = `<span class="dot"></span>Model not loaded`;
    }
  } catch {
    modelStatus.className = "status-pill bad";
    modelStatus.innerHTML = `<span class="dot"></span>Backend offline`;
  }
}
checkHealth();

// ---- File selection ----------------------------------------------- //
function setFile(file) {
  if (!file || !file.type.startsWith("image/")) return;
  selectedFile = file;
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  previewHolder.hidden = false;
  dzEmpty.hidden = true;
  analyzeBtn.disabled = false;
}

function clearFile() {
  selectedFile = null;
  fileInput.value = "";
  previewImg.src = "";
  previewHolder.hidden = true;
  dzEmpty.hidden = false;
  analyzeBtn.disabled = true;
  resultBody.hidden = true;
  resultError.hidden = true;
  resultEmpty.hidden = false;
}

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
});
fileInput.addEventListener("change", (e) => setFile(e.target.files[0]));

clearBtn.addEventListener("click", (e) => { e.stopPropagation(); clearFile(); });

["dragenter", "dragover"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("dragging"); })
);
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("dragging"); })
);
dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) setFile(file);
});

// ---- Prediction --------------------------------------------------- //
analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  analyzeBtn.classList.add("loading");
  analyzeBtn.disabled = true;
  resultError.hidden = true;

  const form = new FormData();
  form.append("file", selectedFile);

  try {
    const res = await fetch("/api/predict", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Prediction failed.");
    renderResult(data);
  } catch (err) {
    showError(err.message);
  } finally {
    analyzeBtn.classList.remove("loading");
    analyzeBtn.disabled = false;
  }
});

function renderResult(data) {
  const pMal = data.probability_malignant;      // P(malignant)
  const pBen = 1 - pMal;
  const isMal = data.prediction === "Malignant";
  const conf = Math.round(data.confidence * 100);

  resultEmpty.hidden = true;
  resultError.hidden = true;
  resultBody.hidden = false;

  // Verdict pill
  verdictLabel.textContent = data.prediction;
  verdictLabel.className = "verdict-label " + (isMal ? "malignant" : "benign");

  // Gauge reflects confidence in the chosen class
  const color = isMal ? getVar("--malignant") : getVar("--benign");
  gaugeFill.style.stroke = color;

  // animate from empty
  gaugeFill.style.strokeDashoffset = GAUGE_CIRC;
  confNum.textContent = "0%";
  requestAnimationFrame(() => {
    gaugeFill.style.strokeDashoffset = GAUGE_CIRC * (1 - conf / 100);
  });
  animateNumber(confNum, conf);

  // Bars
  const malPct = Math.round(pMal * 100);
  const benPct = 100 - malPct;
  benignBar.style.width = benPct + "%";
  malignantBar.style.width = malPct + "%";
  benignVal.textContent = benPct + "%";
  malignantVal.textContent = malPct + "%";
}

function showError(msg) {
  resultEmpty.hidden = true;
  resultBody.hidden = true;
  resultError.hidden = false;
  resultError.textContent = "⚠ " + msg;
}

// ---- Small helpers ------------------------------------------------ //
function getVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function animateNumber(el, target) {
  const start = performance.now();
  const dur = 900;
  function tick(now) {
    const t = Math.min((now - start) / dur, 1);
    const eased = 1 - Math.pow(1 - t, 3);
    el.textContent = Math.round(eased * target) + "%";
    if (t < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}
