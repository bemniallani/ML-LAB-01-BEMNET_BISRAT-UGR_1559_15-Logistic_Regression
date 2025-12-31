document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('predictionForm');
    const resetBtn = document.getElementById('resetBtn');
    const resultContainer = document.getElementById('resultContainer');
    const loader = document.getElementById('loader');

    // API Configuration for Logistic Regression
    const API_URL = 'http://localhost:8001';

    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        loader.style.display = 'flex';
        
        try {
            const formData = {
                Pregnancies: parseFloat(document.getElementById('pregnancies').value),
                Glucose: parseFloat(document.getElementById('glucose').value),
                BloodPressure: parseFloat(document.getElementById('bloodPressure').value),
                SkinThickness: parseFloat(document.getElementById('skinThickness').value),
                Insulin: parseFloat(document.getElementById('insulin').value),
                BMI: parseFloat(document.getElementById('bmi').value),
                DiabetesPedigreeFunction: parseFloat(document.getElementById('dpf').value),
                Age: parseFloat(document.getElementById('age').value)
            };

            const response = await fetch(`${API_URL}/predict`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(formData)
            });

            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

            const result = await response.json();
            updateResultUI(result);
            resultContainer.style.display = 'block';
            
        } catch (error) {
            console.error('Error:', error);
            alert('Error making prediction. Ensure Logistic Regression backend is running on port 8001.');
        } finally {
            loader.style.display = 'none';
        }
    });

    // Reset Form and Hide Results
    resetBtn.addEventListener('click', function() {
        form.reset();
        resultContainer.style.display = 'none';
    });

    function updateResultUI(result) {
        const statusText = document.getElementById('predictionStatus');
        const statusIcon = document.getElementById('statusIcon');
        const descriptionText = document.getElementById('riskDescription');
        
        // Logistic Regression logic (Probability > 0.5)
        const isDiabetic = result.probability_diabetic > 0.5;

        if (isDiabetic) {
            statusText.textContent = "Diabetic";
            statusText.style.color = "#e74c3c"; // Red
            statusIcon.innerHTML = '<i class="fas fa-notes-medical" style="color: #e74c3c;"></i>';
            descriptionText.textContent = "Based on statistical analysis, the patient shows a high probability of diabetes.";
        } else {
            statusText.textContent = "Non-Diabetic";
            statusText.style.color = "#2ecc71"; // Green
            statusIcon.innerHTML = '<i class="fas fa-heart" style="color: #2ecc71;"></i>';
            descriptionText.textContent = "Based on statistical analysis, the patient shows a low probability of diabetes.";
        }
    }
});