document.getElementById('attritionForm').addEventListener('submit', function(event) {
    event.preventDefault();
    
    const formData = new FormData(this);
    const data = Object.fromEntries(formData.entries());

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
    })
    .then(async response => {
        const responseData = await response.json();

        if (!response.ok) {
            throw new Error(responseData.error || 'Server error');
        }
        return responseData;
    })
    .then(data => {
        
        if (data && data.message) {
            console.log('Success:', data);
            document.getElementById('result').classList.remove('hidden');
            document.getElementById('predictionResult').textContent = `Prediction: ${data.message}`;
        } else {
            throw new Error('Unexpected response format');
        }
    })
    .catch((error) => {
        console.error('Error:', error.message || error);
        alert(`An error occurred: ${error.message || error}`);
    });
});
