// Asistente paso a paso del tamizaje rápido.
(function () {
  const form = document.getElementById("wizard");
  if (!form) return;
  const steps = [...form.querySelectorAll(".step")];
  const counter = form.querySelector(".step-count b");
  const bar = form.querySelector(".progress span");
  const prev = form.querySelector("[data-prev]");
  const next = form.querySelector("[data-next]");
  const submit = form.querySelector("[data-submit]");
  const errorBox = form.querySelector(".step-error");
  const painDetail = form.querySelector(".pain-detail");
  let current = 0;

  const val = (name) => {
    const el = form.querySelector(`[name="${name}"]:checked`) || form.querySelector(`[name="${name}"]:not([type=radio])`);
    return el ? el.value : "";
  };
  const checked = (name) => !!form.querySelector(`[name="${name}"]:checked`);

  // Mostrar las preguntas de detalle solo si hay dolor.
  form.querySelectorAll('[name="has_pain"]').forEach((r) =>
    r.addEventListener("change", () => { painDetail.hidden = val("has_pain") !== "si"; }));

  // "No sé" deshabilita el campo numérico.
  form.querySelectorAll(".nose input").forEach((cb) => {
    const input = cb.closest(".inline").querySelector("input[type=number]");
    cb.addEventListener("change", () => { input.disabled = cb.checked; if (cb.checked) input.value = ""; });
  });

  function inRange(name, min, max, optional) {
    if (optional && checked(`${name}_nose`)) return true;
    const v = parseFloat(val(name));
    return !isNaN(v) && v >= min && v <= max;
  }

  function validate(i) {
    if (i === 0) {
      if (!inRange("age", 18, 100)) return "Ingresa una edad entre 18 y 100 años.";
      if (!checked("sex")) return "Selecciona el sexo.";
    }
    if (i === 1) {
      if (!checked("has_pain")) return "Responde si has sentido dolor en el pecho.";
      if (val("has_pain") === "si" && !["central", "exertion", "relief"].every(checked))
        return "Responde las tres preguntas sobre el dolor.";
    }
    if (i === 2) {
      if (!inRange("bp", 70, 250, true)) return "Presión: escribe un valor entre 70 y 250, o marca “No sé”.";
      if (!inRange("chol", 80, 700, true)) return "Colesterol: escribe un valor entre 80 y 700, o marca “No sé”.";
      if (!checked("fbs")) return "Responde la pregunta de la glucosa.";
    }
    return "";
  }

  function fillReview() {
    const yn = { si: "Sí", no: "No", nose: "No sé" };
    const rows = [
      ["Edad", `${val("age")} años`],
      ["Sexo", val("sex") === "M" ? "Hombre" : "Mujer"],
      ["Dolor en el pecho", yn[val("has_pain")]],
    ];
    if (val("has_pain") === "si") {
      rows.push(["En el centro del pecho", yn[val("central")]]);
      rows.push(["Aparece con esfuerzo", yn[val("exertion")]]);
      rows.push(["Se quita al descansar", yn[val("relief")]]);
    }
    rows.push(["Presión sistólica", checked("bp_nose") || !val("bp") ? "No sé" : `${val("bp")} mmHg`]);
    rows.push(["Colesterol total", checked("chol_nose") || !val("chol") ? "No sé" : `${val("chol")} mg/dL`]);
    rows.push(["Glucosa > 120 / diabetes", yn[val("fbs")]]);
    document.getElementById("review").innerHTML =
      rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
  }

  function show(i) {
    steps.forEach((s, j) => s.classList.toggle("on", j === i));
    counter.textContent = i + 1;
    bar.style.width = `${((i + 1) / steps.length) * 100}%`;
    prev.disabled = i === 0;
    const last = i === steps.length - 1;
    next.hidden = last;
    submit.hidden = !last;
    if (last) fillReview();
    errorBox.textContent = "";
    current = i;
  }

  next.addEventListener("click", () => {
    const err = validate(current);
    if (err) { errorBox.textContent = err; return; }
    show(current + 1);
  });
  prev.addEventListener("click", () => show(current - 1));
  form.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && current < steps.length - 1) { e.preventDefault(); next.click(); }
  });
  form.addEventListener("submit", () => { submit.disabled = true; submit.textContent = "Calculando…"; });
})();
