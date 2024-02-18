const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');

const OpenAI = require('openai');

const openai  = new OpenAI({
    apiKey: "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",
    });


// Setup the server
const app = express();
app.use(bodyParser.json());
app.use(cors());

// Endpoint for chatGPT
app.post('/chat', async (req, res) => {
    const {prompt} = req.body;
    const completion = await openai.createChatCompletion({
        model: 'gpt-3.5-turbo',
        max_tokens: 512,
        temperature: 0,
        prompt: prompt,
    });
    res.send(completion.data.choices[0].text);
});

const port = 8081;
app.listen(port, () => {
    console.log(`Server is listening on port ${port}`);
});