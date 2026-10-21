/**
 * NeuroScan AI: Main Client-side Controller
 * Professional Navy & Teal Diagnostic Interface
 * Handles drag-and-drop uploads, sample selection, asynchronous inference,
 * Chart.js probability visualizers, and dynamic UI updates.
 */

document.addEventListener('DOMContentLoaded', function () {
    // DOM Elements
    const dropZone = document.getElementById('dropZone');
    const mriFileInput = document.getElementById('mriFileInput');
    const dropzoneDefaultState = document.getElementById('dropzoneDefaultState');
    const dropzonePreviewState = document.getElementById('dropzonePreviewState');
    const localPreviewImg = document.getElementById('localPreviewImg');
    const selectedFileName = document.getElementById('selectedFileName');
    const selectedFileInfo = document.getElementById('selectedFileInfo');
    const clearSelectedFileBtn = document.getElementById('clearSelectedFileBtn');
    const patientNameInput = document.getElementById('patientName');
    const scanningLine = document.getElementById('scanningLine');
    const totalScansCounter = document.getElementById('totalScansCounter');
    
    const sampleCards = document.querySelectorAll('.sample-card');
    const resetSamplesBtn = document.getElementById('resetSamplesBtn');
    
    const analyzeBtn = document.getElementById('analyzeBtn');
    const analyzeBtnText = document.getElementById('analyzeBtnText');
    const analyzeSpinner = document.getElementById('analyzeSpinner');
    const analyzeIcon = document.getElementById('analyzeIcon');
    
    // Results DOM elements
    const emptyResultsState = document.getElementById('emptyResultsState');
    const activeResultsState = document.getElementById('activeResultsState');
    const inferenceStatusBadge = document.getElementById('inferenceStatusBadge');
    
    const resultScanImg = document.getElementById('resultScanImg');
    const resultSeverityBadge = document.getElementById('resultSeverityBadge');
    const resultCaseId = document.getElementById('resultCaseId');
    const resultPredictedTitle = document.getElementById('resultPredictedTitle');
    const resultConfidence = document.getElementById('resultConfidence');
    const probabilitiesContainer = document.getElementById('probabilitiesContainer');
    const resultDescription = document.getElementById('resultDescription');
    const resultMriFeatures = document.getElementById('resultMriFeatures');
    const resultPrecautions = document.getElementById('resultPrecautions');
    const recentScansTableBody = document.getElementById('recentScansTableBody');

    // Grad-CAM DOM elements
    const gradcamSection = document.getElementById('gradcamSection');
    const gradcamOverlayImg = document.getElementById('gradcamOverlayImg');
    const gradcamRawImg = document.getElementById('gradcamRawImg');
    const gradcamStatusBadge = document.getElementById('gradcamStatusBadge');
    const tumorDetectedBox = document.getElementById('tumorDetectedBox');
    const anatomicalLocationBox = document.getElementById('anatomicalLocationBox');
    const noTumorLocBox = document.getElementById('noTumorLocBox');

    // Chart.js instance holder
    let chartInstance = null;

    // State Variables
    let selectedFile = null;
    let selectedSampleFilename = null;

    // Helper: Reset selection state
    function resetSelection() {
        selectedFile = null;
        selectedSampleFilename = null;
        if (mriFileInput) mriFileInput.value = '';
        
        if (dropzoneDefaultState && dropzonePreviewState) {
            dropzoneDefaultState.classList.remove('d-none');
            dropzonePreviewState.classList.add('d-none');
        }
        
        sampleCards.forEach(c => c.classList.remove('active'));
        if (analyzeBtn) analyzeBtn.disabled = true;
        if (scanningLine) scanningLine.classList.add('d-none');
    }

    // Render Grad-CAM localization results
    function renderGradCAM(gradcamData, predictedClass) {
        if (!gradcamSection) return;

        // Show the Grad-CAM section
        gradcamSection.classList.remove('d-none');

        if (!gradcamData) {
            gradcamStatusBadge.innerHTML = '<i class="fa-solid fa-circle-xmark me-1"></i> Unavailable';
            gradcamStatusBadge.className = 'badge bg-dark-subtle text-muted border border-subtle font-mono';
            if (tumorDetectedBox) tumorDetectedBox.innerHTML = '<div class="text-muted small text-center py-1">Grad-CAM data not available for this analysis.</div>';
            if (gradcamOverlayImg) gradcamOverlayImg.style.display = 'none';
            if (gradcamRawImg) gradcamRawImg.style.display = 'none';
            if (anatomicalLocationBox) anatomicalLocationBox.classList.add('d-none');
            if (noTumorLocBox) noTumorLocBox.classList.add('d-none');
            return;
        }

        // Show heatmap images
        if (gradcamData.heatmap_overlay && gradcamOverlayImg) {
            gradcamOverlayImg.src = gradcamData.heatmap_overlay;
            gradcamOverlayImg.style.display = '';
        }
        if (gradcamData.heatmap_raw && gradcamRawImg) {
            gradcamRawImg.src = gradcamData.heatmap_raw;
            gradcamRawImg.style.display = '';
        }

        if (gradcamData.tumor_detected && gradcamData.anatomical_location) {
            // Tumor detected - show location details
            gradcamStatusBadge.innerHTML = '<i class="fa-solid fa-location-dot me-1"></i> Tumor Located';
            gradcamStatusBadge.className = 'badge bg-danger-subtle text-danger border border-danger-subtle font-mono';

            // Status badge
            if (tumorDetectedBox) {
                tumorDetectedBox.innerHTML = `
                    <div class="d-flex align-items-center gap-2 p-2 rounded bg-danger bg-opacity-10 border border-danger-subtle">
                        <i class="fa-solid fa-triangle-exclamation text-danger"></i>
                        <div class="small">
                            <span class="fw-bold text-danger">Focal Lesion Detected</span>
                            <span class="text-muted"> — Spatial activation confirms tumor presence in the scan.</span>
                        </div>
                    </div>
                `;
            }

            // Show anatomical location box
            if (anatomicalLocationBox) {
                anatomicalLocationBox.classList.remove('d-none');
                const loc = gradcamData.anatomical_location;
                document.getElementById('locHemisphere').textContent = loc.hemisphere || '—';
                document.getElementById('locVertical').textContent = loc.vertical_level || '—';
                document.getElementById('locRegion').textContent = loc.region || '—';
                document.getElementById('locDescription').textContent = loc.description || '—';
            }

            // Fill bounding box data
            if (gradcamData.bounding_box) {
                const bb = gradcamData.bounding_box;
                document.getElementById('bboxXMin').textContent = bb.x_min;
                document.getElementById('bboxYMin').textContent = bb.y_min;
                document.getElementById('bboxWidth').textContent = `${bb.width}px`;
                document.getElementById('bboxHeight').textContent = `${bb.height}px`;
                document.getElementById('bboxCenter').textContent = `(${bb.center_x}, ${bb.center_y})`;
                document.getElementById('bboxArea').textContent = `${bb.area_percent}%`;
            }

            // Hide no tumor message
            if (noTumorLocBox) noTumorLocBox.classList.add('d-none');

        } else {
            // No tumor detected or healthy scan
            gradcamStatusBadge.innerHTML = '<i class="fa-solid fa-shield-check me-1"></i> Clear';
            gradcamStatusBadge.className = 'badge bg-success-subtle text-success border border-success-subtle font-mono';

            if (tumorDetectedBox) {
                tumorDetectedBox.innerHTML = '';
            }

            if (anatomicalLocationBox) anatomicalLocationBox.classList.add('d-none');
            if (noTumorLocBox) noTumorLocBox.classList.remove('d-none');
        }
    }

    // Handle Local File Selection
    function handleFile(file) {
        if (!file) return;

        const validTypes = ['image/jpeg', 'image/png', 'image/webp'];
        if (!validTypes.includes(file.type)) {
            alert('Please select a valid MRI image format (.jpg, .jpeg, .png, or .webp).');
            return;
        }

        selectedFile = file;
        selectedSampleFilename = null;
        sampleCards.forEach(c => c.classList.remove('active'));

        const reader = new FileReader();
        reader.onload = function (e) {
            localPreviewImg.src = e.target.result;
            selectedFileName.textContent = file.name;
            const sizeInKb = (file.size / 1024).toFixed(1);
            if (selectedFileInfo) {
                selectedFileInfo.textContent = `${sizeInKb} KB • Uploaded Scan Ready`;
            }
            dropzoneDefaultState.classList.add('d-none');
            dropzonePreviewState.classList.remove('d-none');
            if (analyzeBtn) analyzeBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    // Drag and Drop Event Listeners
    if (dropZone) {
        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropZone.classList.add('dragover');
            }, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropZone.classList.remove('dragover');
            }, false);
        });

        dropZone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files.length > 0) {
                handleFile(files[0]);
            }
        });
    }

    if (mriFileInput) {
        mriFileInput.addEventListener('change', function () {
            if (this.files && this.files.length > 0) {
                handleFile(this.files[0]);
            }
        });
    }

    if (clearSelectedFileBtn) {
        clearSelectedFileBtn.addEventListener('click', function (e) {
            e.stopPropagation();
            resetSelection();
        });
    }

    // Handle Sample Card Selection
    sampleCards.forEach(card => {
        card.addEventListener('click', function () {
            const sampleName = this.getAttribute('data-sample');
            sampleCards.forEach(c => c.classList.remove('active'));
            this.classList.add('active');

            selectedSampleFilename = sampleName;
            selectedFile = null;
            if (mriFileInput) mriFileInput.value = '';

            const img = this.querySelector('img');
            if (img && localPreviewImg) {
                localPreviewImg.src = img.src;
                selectedFileName.textContent = `Sample: ${sampleName}`;
                if (selectedFileInfo) {
                    selectedFileInfo.textContent = `Pre-packaged Evaluation Scan`;
                }
                dropzoneDefaultState.classList.add('d-none');
                dropzonePreviewState.classList.remove('d-none');
            }

            if (analyzeBtn) analyzeBtn.disabled = false;
        });
    });

    if (resetSamplesBtn) {
        resetSamplesBtn.addEventListener('click', resetSelection);
    }

    // Render Probabilities Helper
    function renderProbabilities(probabilities, predictedClass) {
        probabilitiesContainer.innerHTML = '';

        const classThemeMap = {
            'Glioma': { bar: 'progress-bar-danger', text: 'text-danger' },
            'Meningioma': { bar: 'progress-bar-warning', text: 'text-warning' },
            'Pituitary': { bar: 'progress-bar-info', text: 'text-info' },
            'No Tumor': { bar: 'progress-bar-success', text: 'text-success' }
        };

        for (const [className, prob] of Object.entries(probabilities)) {
            const isHighest = (className === predictedClass);
            const theme = classThemeMap[className] || { bar: 'progress-bar-teal', text: 'text-teal' };

            const wrapper = document.createElement('div');
            wrapper.className = `prob-bar-wrapper ${isHighest ? 'winner' : ''}`;
            wrapper.innerHTML = `
                <div class="d-flex justify-content-between align-items-center small mb-1">
                    <span class="fw-semibold ${isHighest ? 'text-teal' : 'text-light'}">
                        ${isHighest ? '<i class="fa-solid fa-circle-check text-teal me-1"></i>' : '<i class="fa-regular fa-circle text-muted me-1"></i>'}
                        ${className}
                        ${isHighest ? '<span class="badge bg-teal-subtle text-teal ms-1 font-mono fs-8">Top Diagnosis</span>' : ''}
                    </span>
                    <span class="font-mono ${isHighest ? 'fw-bold text-teal fs-6' : 'text-muted'}">${prob}%</span>
                </div>
                <div class="progress" style="height: 8px;">
                    <div class="progress-bar ${isHighest ? 'progress-bar-teal' : theme.bar}" role="progressbar" style="width: ${prob}%;" aria-valuenow="${prob}" aria-valuemin="0" aria-valuemax="100"></div>
                </div>
            `;
            probabilitiesContainer.appendChild(wrapper);
        }

        // Render Chart.js Chart
        renderProbabilityChart(probabilities, predictedClass);
    }

    // Render Chart.js Dynamic Visualizer
    function renderProbabilityChart(probabilities, predictedClass) {
        const ctx = document.getElementById('predictionChart');
        if (!ctx || typeof Chart === 'undefined') return;

        const labels = Object.keys(probabilities);
        const dataValues = Object.values(probabilities);

        const colors = labels.map(l => (l === predictedClass ? '#22d3ee' : '#1e3a5f'));
        const borderColors = labels.map(l => (l === predictedClass ? '#06b6d4' : '#2d4e8a'));

        if (chartInstance) {
            chartInstance.destroy();
        }

        chartInstance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Probability (%)',
                    data: dataValues,
                    backgroundColor: colors,
                    borderColor: borderColors,
                    borderWidth: 1.5,
                    borderRadius: 6
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function (context) {
                                return ` ${context.raw}% probability`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        min: 0,
                        max: 100,
                        ticks: { color: '#94a3b8', font: { family: 'JetBrains Mono', size: 10 } },
                        grid: { color: 'rgba(45, 78, 138, 0.25)' }
                    },
                    y: {
                        ticks: { color: '#e2e8f0', font: { family: 'Plus Jakarta Sans', weight: '600', size: 11 } },
                        grid: { display: false }
                    }
                }
            }
        });
    }

    // Handle Neural Classification Trigger
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', async function () {
            if (!selectedFile && !selectedSampleFilename) return;

            // Enter loading state
            analyzeBtn.disabled = true;
            analyzeBtnText.textContent = 'Analyzing Deep Convolutional Features...';
            analyzeSpinner.classList.remove('d-none');
            analyzeIcon.classList.add('d-none');
            if (scanningLine) scanningLine.classList.remove('d-none');
            
            if (inferenceStatusBadge) {
                inferenceStatusBadge.textContent = 'Inference in Progress...';
                inferenceStatusBadge.className = 'badge bg-warning text-dark';
            }

            // Hide Grad-CAM section during analysis
            if (gradcamSection) gradcamSection.classList.add('d-none');

            const formData = new FormData();
            const patientName = patientNameInput ? patientNameInput.value.trim() : '';
            formData.append('patient_name', patientName || 'Anonymous Case');

            if (selectedFile) {
                formData.append('mri_image', selectedFile);
            } else if (selectedSampleFilename) {
                formData.append('sample_file', selectedSampleFilename);
            }

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (data.success) {
                    // Switch output views
                    if (emptyResultsState) emptyResultsState.classList.add('d-none');
                    if (activeResultsState) {
                        activeResultsState.classList.remove('d-none');
                        activeResultsState.classList.add('d-flex');
                    }

                    if (inferenceStatusBadge) {
                        inferenceStatusBadge.textContent = 'Inference Verified';
                        inferenceStatusBadge.className = 'badge bg-teal-subtle text-teal border border-teal-subtle';
                    }

                    // Populate diagnostic UI elements
                    resultScanImg.src = data.image_url;
                    resultPredictedTitle.textContent = data.predicted_class;
                    resultConfidence.textContent = `${data.confidence}%`;
                    resultCaseId.textContent = `Scan #${data.scan_id} • ${data.patient_name}`;

                    // Badge theme
                    const badgeMap = {
                        'Glioma': 'bg-danger text-light',
                        'Meningioma': 'bg-warning text-dark',
                        'Pituitary': 'bg-info text-dark',
                        'No Tumor': 'bg-success text-light'
                    };
                    resultSeverityBadge.className = `badge ${badgeMap[data.predicted_class] || 'bg-teal text-dark'}`;
                    resultSeverityBadge.textContent = data.details.severity || 'Evaluated';

                    // Clinical details
                    resultDescription.textContent = data.details.description || '';
                    resultMriFeatures.innerHTML = `<strong>MRI Hallmarks:</strong> ${data.details.mri_features || ''}`;
                    resultPrecautions.textContent = data.details.precautions || 'Maintain radiological correlation.';

                    // Render multiclass probability distribution
                    renderProbabilities(data.probabilities, data.predicted_class);

                    // Render Grad-CAM tumor localization
                    renderGradCAM(data.gradcam || null, data.predicted_class);

                    // Update total scans counter
                    if (totalScansCounter) {
                        const currentVal = parseInt(totalScansCounter.textContent, 10) || 0;
                        totalScansCounter.textContent = (currentVal + 1).toString();
                    }

                    // Add new row to recent table if present
                    if (recentScansTableBody) {
                        const emptyRow = recentScansTableBody.querySelector('td[colspan]');
                        if (emptyRow) emptyRow.parentElement.remove();

                        const badgeClassMap = {
                            'Glioma': 'bg-danger-subtle text-danger border border-danger-subtle',
                            'Meningioma': 'bg-warning-subtle text-warning border border-warning-subtle',
                            'Pituitary': 'bg-info-subtle text-info border border-info-subtle',
                            'No Tumor': 'bg-success-subtle text-success border border-success-subtle'
                        };

                        const tr = document.createElement('tr');
                        tr.innerHTML = `
                            <td class="ps-4">
                                <a href="${data.image_url}" target="_blank" title="Inspect Full Image">
                                    <img src="${data.image_url}" alt="Thumbnail" class="rounded border border-subtle zoom-thumbnail" style="width: 44px; height: 44px; object-fit: cover;">
                                </a>
                            </td>
                            <td>
                                <div class="fw-semibold text-light">${data.patient_name}</div>
                                <small class="text-muted font-mono">Scan #${data.scan_id}</small>
                            </td>
                            <td>
                                <span class="badge ${badgeClassMap[data.predicted_class]}">${data.predicted_class}</span>
                            </td>
                            <td>
                                <div class="fw-bold font-mono text-teal">${data.confidence}%</div>
                            </td>
                            <td class="text-muted small">Just now</td>
                            <td class="text-end pe-4">
                                <form method="POST" action="/history/delete/${data.scan_id}" class="d-inline" onsubmit="return confirm('Delete this scan entry?');">
                                    <button type="submit" class="btn btn-outline-danger btn-sm border-0" title="Delete record">
                                        <i class="fa-regular fa-trash-can"></i>
                                    </button>
                                </form>
                            </td>
                        `;
                        recentScansTableBody.prepend(tr);
                    }

                    // Smooth scroll to results on mobile
                    if (window.innerWidth < 992) {
                        activeResultsState.scrollIntoView({ behavior: 'smooth' });
                    }

                } else {
                    alert('Prediction Notice: ' + (data.error || 'Unknown error occurred'));
                    if (inferenceStatusBadge) {
                        inferenceStatusBadge.textContent = 'Error';
                        inferenceStatusBadge.className = 'badge bg-danger';
                    }
                }
            } catch (err) {
                console.error(err);
                alert('Connection error occurred while communicating with the neural server.');
                if (inferenceStatusBadge) {
                    inferenceStatusBadge.textContent = 'Connection Error';
                    inferenceStatusBadge.className = 'badge bg-danger';
                }
            } finally {
                analyzeBtn.disabled = false;
                analyzeBtnText.textContent = 'Run Neural Classification';
                analyzeSpinner.classList.add('d-none');
                analyzeIcon.classList.remove('d-none');
                if (scanningLine) scanningLine.classList.add('d-none');
            }
        });
    }

    // ==========================================
    // History Table Live Filter & Search
    // ==========================================
    const historySearchInput = document.getElementById('historySearchInput');
    const filterButtons = document.querySelectorAll('.filter-btn');
    const historyRows = document.querySelectorAll('.history-row');

    if (historySearchInput && historyRows.length > 0) {
        historySearchInput.addEventListener('input', function () {
            const query = this.value.toLowerCase().trim();
            const activeFilterBtn = document.querySelector('.filter-btn.active');
            const activeFilter = activeFilterBtn ? activeFilterBtn.getAttribute('data-filter') : 'all';

            historyRows.forEach(row => {
                const patient = row.getAttribute('data-patient') || '';
                const tumorClass = row.getAttribute('data-class') || '';

                const matchesQuery = patient.includes(query);
                const matchesClass = (activeFilter === 'all' || tumorClass === activeFilter);

                if (matchesQuery && matchesClass) {
                    row.style.display = '';
                } else {
                    row.style.display = 'none';
                }
            });
        });
    }

    if (filterButtons.length > 0 && historyRows.length > 0) {
        filterButtons.forEach(btn => {
            btn.addEventListener('click', function () {
                filterButtons.forEach(b => b.classList.remove('active'));
                this.classList.add('active');

                const filter = this.getAttribute('data-filter');
                const query = historySearchInput ? historySearchInput.value.toLowerCase().trim() : '';

                historyRows.forEach(row => {
                    const tumorClass = row.getAttribute('data-class') || '';
                    const patient = row.getAttribute('data-patient') || '';

                    const matchesClass = (filter === 'all' || tumorClass === filter);
                    const matchesQuery = (!query || patient.includes(query));

                    if (matchesClass && matchesQuery) {
                        row.style.display = '';
                    } else {
                        row.style.display = 'none';
                    }
                });
            });
        });
    }
});
