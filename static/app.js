const input = document.querySelector("#image-input");
const uploadArea = document.querySelector(".upload-area");
const previewSection = document.querySelector("#preview-section");
const preview = document.querySelector("#image-preview");
const fileName = document.querySelector("#file-name");
const fileSize = document.querySelector("#file-size");
const analyseButton = document.querySelector("#analyse-button");
const errorMessage = document.querySelector("#error-message");
const resultCard = document.querySelector("#result-card");
const resultFilename = document.querySelector("#result-filename");
const scoreValue = document.querySelector("#score-value");
const verdict = document.querySelector("#verdict");
const meter = document.querySelector(".meter");
const meterFill = document.querySelector("#meter-fill");

let selectedFile = null;
let previewUrl = null;

function showError(message) {
  errorMessage.textContent = message;
  errorMessage.hidden = !message;
}

function formatBytes(bytes) {
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function selectFile(file) {
  showError("");
  resultCard.hidden = true;

  if (!file || !file.type.startsWith("image/")) {
    selectedFile = null;
    previewSection.hidden = true;
    analyseButton.disabled = true;
    showError("Choose a supported image file.");
    return;
  }

  if (file.size > 15 * 1024 * 1024) {
    selectedFile = null;
    previewSection.hidden = true;
    analyseButton.disabled = true;
    showError("The image is larger than the 15 MB limit.");
    return;
  }

  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
  }

  selectedFile = file;
  previewUrl = URL.createObjectURL(file);
  preview.src = previewUrl;
  fileName.textContent = file.name;
  fileSize.textContent = formatBytes(file.size);
  previewSection.hidden = false;
  analyseButton.disabled = false;
}

input.addEventListener("change", () => selectFile(input.files[0]));

["dragenter", "dragover"].forEach((eventName) => {
  uploadArea.addEventListener(eventName, (event) => {
    event.preventDefault();
    uploadArea.classList.add("is-dragging");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  uploadArea.addEventListener(eventName, (event) => {
    event.preventDefault();
    uploadArea.classList.remove("is-dragging");
  });
});

uploadArea.addEventListener("drop", (event) => {
  selectFile(event.dataTransfer.files[0]);
});

analyseButton.addEventListener("click", async () => {
  if (!selectedFile) {
    showError("Choose an image first.");
    return;
  }

  showError("");
  resultCard.hidden = true;
  analyseButton.disabled = true;
  analyseButton.textContent = "Analysing…";

  const formData = new FormData();
  formData.append("image", selectedFile);

  try {
    const response = await fetch("/api/analyse", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.error || "The image could not be analysed.");
    }

    resultFilename.textContent = payload.filename;
    scoreValue.textContent = Number(payload.score).toFixed(1);
    verdict.textContent = payload.verdict;
    meter.setAttribute("aria-valuenow", payload.score);
    meterFill.style.width = `${payload.score}%`;
    resultCard.hidden = false;
    resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (error) {
    showError(error.message);
  } finally {
    analyseButton.disabled = false;
    analyseButton.textContent = "Analyse image";
  }
});
