// Helper to format model object keys into clean display names
function formatModelName(key) {
  const names = {
    logistic_regression: "Logistic Regression",
    decision_tree: "Decision Tree",
    knn: "K-Nearest Neighbors",
    naive_bayes: "Naive Bayes",
    random_forest: "Random Forest"
  };
  return names[key] || key;
}

// Fetch and render preview of training data
async function peekData() {
  const previewDiv = document.getElementById("dataPreview");
  previewDiv.classList.remove("hidden");
  previewDiv.innerHTML = `<p class="p-4 text-slate-400">Loading preview data...</p>`;

  try {
    const response = await fetch("/api/peek-data");
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Failed to fetch data preview.");
    }

    if (Array.isArray(data) && data.length > 0) {
      const headers = Object.keys(data[0]);
      let tableHtml = `
        <table class="w-full text-xs text-left text-slate-300 border-collapse">
          <thead class="bg-slate-700 text-slate-100 uppercase sticky top-0">
            <tr>${headers.map(h => `<th class="p-3 border-b border-slate-600 font-semibold">${h}</th>`).join('')}</tr>
          </thead>
          <tbody class="divide-y divide-slate-700">
            ${data.map(row => `
              <tr class="hover:bg-slate-750 transition-colors">
                ${headers.map(h => `<td class="p-3 whitespace-nowrap">${row[h]}</td>`).join('')}
              </tr>
            `).join('')}
          </tbody>
        </table>
      `;
      previewDiv.innerHTML = tableHtml;
    } else {
      previewDiv.innerHTML = `<p class="p-4 text-slate-400">No preview data returned.</p>`;
    }
  } catch (err) {
    previewDiv.innerHTML = `<p class="p-4 text-red-400">Error: ${err.message}</p>`;
  }
}

// Upload test CSV and render metrics table
async function runPrediction() {
  const fileInput = document.getElementById("csvFile");
  const outputContainer = document.getElementById("outputContainer");
  const outputStatus = document.getElementById("outputStatus");
  const outputContent = document.getElementById("outputContent");

  outputContainer.classList.remove("hidden");

  if (!fileInput.files || fileInput.files.length === 0) {
    outputStatus.className = "font-semibold text-lg text-red-400";
    outputStatus.textContent = "Validation Error";
    outputContent.textContent = "Please select a test CSV file before clicking Predict.";
    return;
  }

  outputStatus.className = "font-semibold text-lg text-indigo-400";
  outputStatus.textContent = "Processing...";
  outputContent.innerHTML = `<p class="text-slate-400">Uploading dataset and calculating metrics across all 5 models...</p>`;

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      body: formData
    });

    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.detail || "Prediction request failed.");
    }

    outputStatus.className = "font-semibold text-lg text-emerald-400";
    outputStatus.textContent = "Evaluation Benchmark Results";

    // Build side-by-side comparison table from metrics JSON
    const modelKeys = Object.keys(result);
    if (modelKeys.length > 0) {
      let tableHtml = `
        <div class="overflow-x-auto border border-slate-700 rounded-lg mt-2">
          <table class="w-full text-sm text-left text-slate-200 border-collapse">
            <thead class="bg-slate-800 text-indigo-300 font-semibold uppercase text-xs">
              <tr>
                <th class="p-3 border-b border-slate-700">Model</th>
                <th class="p-3 border-b border-slate-700 text-center">Accuracy</th>
                <th class="p-3 border-b border-slate-700 text-center">AUC Score</th>
                <th class="p-3 border-b border-slate-700 text-center">Precision</th>
                <th class="p-3 border-b border-slate-700 text-center">Recall</th>
                <th class="p-3 border-b border-slate-700 text-center">F1 Score</th>
                <th class="p-3 border-b border-slate-700 text-center">MCC Score</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-slate-700">
      `;

      modelKeys.forEach(key => {
        const m = result[key];
        tableHtml += `
          <tr class="hover:bg-slate-800/50 transition-colors">
            <td class="p-3 font-medium text-slate-100">${formatModelName(key)}</td>
            <td class="p-3 text-center text-emerald-400 font-mono">${m.accuracy}</td>
            <td class="p-3 text-center font-mono">${m.auc_score}</td>
            <td class="p-3 text-center font-mono">${m.precision}</td>
            <td class="p-3 text-center font-mono">${m.recall}</td>
            <td class="p-3 text-center font-mono">${m.f1_score}</td>
            <td class="p-3 text-center font-mono">${m.mcc_score}</td>
          </tr>
        `;
      });

      tableHtml += `
            </tbody>
          </table>
        </div>
      `;

      outputContent.innerHTML = tableHtml;
    }
  } catch (err) {
    outputStatus.className = "font-semibold text-lg text-red-400";
    outputStatus.textContent = "Error Encountered";
    outputContent.textContent = err.message;
  }
}