document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('predictionForm');
    const predictBtn = document.getElementById('predictBtn');
    const loader = document.getElementById('loader');
    const resultPlaceholder = document.getElementById('resultPlaceholder');
    const resultContainer = document.getElementById('resultContainer');
    
    let shapChart = null;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // UI State: Loading
        predictBtn.disabled = true;
        loader.style.display = 'block';
        
        const formData = new FormData(form);
        const ecgFileInput = document.getElementById('ecgFile');
        const hasFiles = ecgFileInput.files.length > 0;

        try {
            let response;
            if (hasFiles) {
                // Multimodal prediction (formData already contains ecg_files from multiple selection)
                response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
            } else {
                // Clinical only prediction
                const clinicalData = {};
                formData.forEach((value, key) => {
                    if (key !== 'ecg_file') {
                        clinicalData[key] = parseFloat(value);
                    }
                });
                
                response = await fetch('/predict-clinical-only', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(clinicalData)
                });
            }

            if (!response.ok) {
                const errData = await response.json();
                console.error('Validation Error:', errData);
                alert(`Analysis failed: ${errData.detail || 'Check console for details'}`);
                throw new Error('Prediction failed');
            }
            
            const data = await response.json();
            updateUI(data);

        } catch (error) {
            console.error(error);
            alert('Analysis failed. Please check the console for details.');
        } finally {
            predictBtn.disabled = false;
            loader.style.display = 'none';
        }
    });

    function updateUI(data) {
        resultPlaceholder.classList.add('hidden');
        resultContainer.classList.remove('hidden');

        // 1. Update Risk Badge & Labels
        const riskLabel = data.risk_label;
        const riskBadge = document.getElementById('riskBadge');
        riskBadge.textContent = `${riskLabel.toUpperCase()} RISK`;
        riskBadge.className = 'risk-badge ' + getRiskClass(riskLabel);

        // 2. Update Probability Circle
        const highProb = data.probabilities.High * 100;
        const circle = document.getElementById('probabilityPath');
        const probText = document.getElementById('probText');
        
        circle.setAttribute('stroke-dasharray', `${highProb}, 100`);
        circle.style.stroke = getRiskColor(riskLabel);
        probText.textContent = `${highProb.toFixed(1)}%`;

        // 3. Update SHAP Chart
        const shapData = data.shap_explanation;
        if (Object.keys(shapData).length > 0) {
            renderShapChart(shapData);
        }

        // 4. Update Recommendations
        updateRecommendations(riskLabel);
        
        // Scroll to results
        resultContainer.scrollIntoView({ behavior: 'smooth' });
    }

    function getRiskClass(label) {
        if (label === 'High') return 'risk-high';
        if (label === 'Moderate') return 'risk-moderate';
        return 'risk-low';
    }

    function getRiskColor(label) {
        if (label === 'High') return '#ff3e3e';
        if (label === 'Moderate') return '#ff9100';
        return '#00e676';
    }

    function renderShapChart(shapDict) {
        const ctx = document.getElementById('shapChart').getContext('2d');
        
        // Sort features by magnitude
        const sortedFeatures = Object.entries(shapDict)
            .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
            .slice(0, 10);
            
        const labels = sortedFeatures.map(f => f[0]);
        const values = sortedFeatures.map(f => f[1]);
        const colors = values.map(v => v > 0 ? '#ff3e3e' : '#2196f3');

        if (shapChart) shapChart.destroy();

        shapChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Impact on Risk Score',
                    data: values,
                    backgroundColor: colors,
                    borderRadius: 5
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: '#a0a0a0' }
                    },
                    y: {
                        grid: { display: false },
                        ticks: { color: '#fff' }
                    }
                }
            }
        });
    }

    function updateRecommendations(risk) {
        const list = document.getElementById('recList');
        const desc = document.getElementById('riskDescription');
        list.innerHTML = '';
        
        if (risk === 'High') {
            desc.textContent = "CRITICAL: Significant cardiac anomalies detected. Immediate clinical intervention is recommended.";
            ['Emergency cardiology consultation', 'Continuous ECG monitoring', 'Troponin I/T blood tests', 'Urgent Echocardiogram'].forEach(text => {
                const li = document.createElement('li');
                li.textContent = text;
                list.appendChild(li);
            });
        } else if (risk === 'Moderate') {
            desc.textContent = "Caution: Some indicators of cardiac stress observed. Follow-up is advised.";
            ['Schedule Stress Test', 'Review medication compliance', '24-hour Holter monitoring', 'Consult primary physician'].forEach(text => {
                const li = document.createElement('li');
                li.textContent = text;
                list.appendChild(li);
            });
        } else {
            desc.textContent = "Normal: No immediate cardiac risks detected by the model. Maintain healthy lifestyle.";
            ['Routine annual checkup', 'Maintain balanced diet', 'Regular aerobic exercise', 'Monitor blood pressure'].forEach(text => {
                const li = document.createElement('li');
                li.textContent = text;
                list.appendChild(li);
            });
        }
    }
});
