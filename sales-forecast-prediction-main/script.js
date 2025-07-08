function predictQuantity() {
    var formData = new FormData(document.getElementById("predictionForm"));
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("predictionResult").innerHTML = "Predicted Quantity: " + data.predicted_quantity;
    })
    .catch(error => console.error('Error:', error));
}