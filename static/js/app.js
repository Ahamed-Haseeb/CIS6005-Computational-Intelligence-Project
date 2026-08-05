const form = document.querySelector("#predictionForm");
const submitButton = form.querySelector('button[type="submit"]');
const resetButton = document.querySelector("#resetButton");
const formMessage = document.querySelector("#formMessage");
const resultPlaceholder = document.querySelector("#resultPlaceholder");
const resultContent = document.querySelector("#resultContent");
const historyBody = document.querySelector("#historyBody");
const historyEmpty = document.querySelector("#historyEmpty");
const historyTable = document.querySelector("#historyTable");
const clearHistory = document.querySelector("#clearHistory");
const menuToggle = document.querySelector(".menu-toggle");
const sidebar = document.querySelector(".sidebar");

const numericFields = new Set([
  "age", "balance", "day", "duration", "campaign", "pdays", "previous"
]);
let history = [];

function serializeForm() {
  return Object.fromEntries(
    [...new FormData(form).entries()].map(([key, value]) => [
      key,
      numericFields.has(key) ? Number(value) : value,
    ])
  );
}

function clearErrors() {
  formMessage.textContent = "";
  form.querySelectorAll(".field.invalid").forEach((field) => {
    field.classList.remove("invalid");
    field.querySelector(".field-error").textContent = "";
  });
}

function showFieldErrors(errors = {}) {
  Object.entries(errors).forEach(([name, message]) => {
    const input = form.elements[name];
    if (!input) return;
    const field = input.closest(".field");
    field.classList.add("invalid");
    field.querySelector(".field-error").textContent = message;
  });
  const firstInvalid = form.querySelector(".field.invalid input, .field.invalid select");
  firstInvalid?.focus();
}

function validateClientSide() {
  const errors = {};
  [...form.elements].forEach((input) => {
    if (!(input instanceof HTMLInputElement || input instanceof HTMLSelectElement)) return;
    if (!input.checkValidity()) {
      errors[input.name] = input.validity.rangeOverflow || input.validity.rangeUnderflow
        ? `Enter a value from ${input.min} to ${input.max}.`
        : "This field is required.";
    }
  });
  return errors;
}

function setLoading(isLoading) {
  submitButton.disabled = isLoading;
  submitButton.classList.toggle("loading", isLoading);
  submitButton.querySelector("span:first-child").textContent = isLoading
    ? "Loading model…"
    : "Run prediction";
}

function renderResult(result) {
  const percent = result.probability * 100;
  const positive = result.prediction === 1 || result.prediction === "1";
  const gauge = document.querySelector("#probabilityGauge");
  gauge.style.setProperty("--probability", percent.toFixed(1));
  gauge.style.setProperty("--green", percent >= 65 ? "#1d7254" : percent >= 35 ? "#d09a31" : "#73817b");
  document.querySelector("#probabilityValue").textContent = `${percent.toFixed(1)}%`;
  document.querySelector("#verdictBadge").textContent = result.segment;
  document.querySelector("#verdictTitle").textContent = positive
    ? "Likely to subscribe"
    : "Unlikely to subscribe";
  document.querySelector("#verdictCopy").textContent = positive
    ? "This profile shows a positive propensity for the term deposit offer."
    : "The model estimates a lower conversion propensity for this profile.";
  document.querySelector("#classValue").textContent = String(result.prediction);
  document.querySelector("#confidenceValue").textContent = `${(result.confidence * 100).toFixed(1)}%`;
  document.querySelector("#modelValue").textContent = result.model.replace("Classifier", "");
  resultPlaceholder.hidden = true;
  resultContent.hidden = false;
}

function renderHistory() {
  historyEmpty.hidden = history.length > 0;
  historyTable.hidden = history.length === 0;
  clearHistory.hidden = history.length === 0;
  historyBody.innerHTML = history.map((item) => `
    <tr>
      <td>${item.time}</td>
      <td>${item.age} · ${item.job}</td>
      <td>${item.contact}</td>
      <td><strong>${(item.probability * 100).toFixed(1)}%</strong></td>
      <td>${item.segment}</td>
    </tr>
  `).join("");
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearErrors();
  const clientErrors = validateClientSide();
  if (Object.keys(clientErrors).length) {
    formMessage.textContent = "Please correct the highlighted fields.";
    showFieldErrors(clientErrors);
    return;
  }

  const payload = serializeForm();
  setLoading(true);
  formMessage.textContent = "The first prediction may take a moment while the model loads.";
  formMessage.style.color = "#66716d";

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) {
      throw Object.assign(new Error(data.error || "Prediction failed."), { fields: data.fields });
    }
    formMessage.textContent = "";
    renderResult(data);
    history.unshift({
      ...data,
      age: payload.age,
      job: payload.job.replaceAll("-", " "),
      contact: payload.contact,
      time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    });
    history = history.slice(0, 8);
    renderHistory();
  } catch (error) {
    formMessage.style.color = "#b74338";
    formMessage.textContent = error.message;
    showFieldErrors(error.fields);
  } finally {
    setLoading(false);
  }
});

form.addEventListener("input", (event) => {
  const field = event.target.closest(".field");
  if (field) {
    field.classList.remove("invalid");
    field.querySelector(".field-error").textContent = "";
  }
});

resetButton.addEventListener("click", () => {
  form.reset();
  clearErrors();
});

clearHistory.addEventListener("click", () => {
  history = [];
  renderHistory();
});

menuToggle.addEventListener("click", () => {
  const open = sidebar.classList.toggle("open");
  menuToggle.setAttribute("aria-expanded", String(open));
});

document.addEventListener("click", (event) => {
  if (window.innerWidth <= 860 && sidebar.classList.contains("open") &&
      !sidebar.contains(event.target) && !menuToggle.contains(event.target)) {
    sidebar.classList.remove("open");
    menuToggle.setAttribute("aria-expanded", "false");
  }
});
