document.getElementById('dropzone').addEventListener('click', () => {
    document.getElementById('imageInput').click();
});
document.getElementById('imageInput').addEventListener('change', (e) => {
    if (e.target.files.length) {
        document.getElementById('dropzone').style.borderColor = '#28a745';
    }
});
document.getElementById('predictBtn').addEventListener('click', async () => {
    const text = document.getElementById('textInput').value.trim();
    const file = document.getElementById('imageInput').files[0];
    if (!text || !file) {
        alert('Veuillez fournir un texte et une image');
        return;
    }
    const reader = new FileReader();
    reader.onloadend = async function() {
        const base64Image = reader.result.split(',')[1];
        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text, image: base64Image })
            });
            const data = await response.json();
            const resultDiv = document.getElementById('result');
            const color = data.prediction === 'coherent' ? '#28a745' : '#dc3545';
            resultDiv.innerHTML = `<div style="background:${color}; color:white; padding:15px; border-radius:5px;">
                <strong>Prédiction : ${data.prediction}</strong><br>
                Confiance : ${(data.confidence * 100).toFixed(2)}%
            </div>`;
        } catch (err) {
            alert('Erreur lors de la prédiction : ' + err.message);
        }
    };
    reader.readAsDataURL(file);
});
