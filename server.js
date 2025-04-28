const express = require('express');
const { spawn } = require('child_process');
const bodyParser = require('body-parser');
const app = express();
const port = 3000;

app.use(bodyParser.json());
app.use(express.static('.')); // Serve static files from current directory

// Endpoint to analyze MBTI
app.post('/analyze-mbti', (req, res) => {
    const text = req.body.text;
    
    // Create a temporary file with the text
    const fs = require('fs');
    const tempFilePath = './temp_chat_history.txt';
    
    fs.writeFileSync(tempFilePath, text);
    
    // Call the Python script for MBTI analysis
    const pythonProcess = spawn('python', ['inference_query.py', '--input', tempFilePath, '--api-only']);
    
    let resultData = '';
    
    pythonProcess.stdout.on('data', (data) => {
        resultData += data.toString();
    });
    
    pythonProcess.stderr.on('data', (data) => {
        console.error(`Python Error: ${data}`);
    });
    
    pythonProcess.on('close', (code) => {
        // Clean up the temporary file
        fs.unlinkSync(tempFilePath);
        
        if (code !== 0) {
            return res.status(500).json({ error: 'Failed to analyze MBTI' });
        }
        
        try {
            const result = JSON.parse(resultData);
            res.json(result);
        } catch (error) {
            console.error('Error parsing Python output:', error);
            res.status(500).json({ error: 'Failed to parse MBTI analysis result' });
        }
    });
});

app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});
