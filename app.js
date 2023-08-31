const API_URL = "https://api.openai.com/v1/chat/completions";
var API_KEY = localStorage.getItem("API_KEY");

function set_api_key() {
    if (API_KEY == null) {
        API_KEY = prompt("Enter your OpenAI API key to use 🤖.")
        localStorage.setItem("API_KEY", API_KEY);
    }
}

const instruction1 = `You are a writing assistant. You need to complete the following text and only return the part you add. Follow the rules below.
- Keep any markdown formatting syntax in the text.
- Do not change the meaning of the text.
- Do not plagiarize the text.
- Be fluent and natural sounding.
- Only return the part you add`;

const instruction2 = `You are an AI writing assistant.  You are expected to improve the writing of the given text and follow these rules.
- Keep the markdown formatting of the input text in the output text.
- Do not change the meaning of the text.
- Do not plagiarize the text.
- Keep the parahrase as the same length as the input text as much as possible.
- Be fluent and natural sounding.`;

const instruction3 = `You are an AI writing assistant. You are expected to paraphrase the given text and follow these rules.
- Keep the markdown formatting of the input text in the output text.
- Do not change the meaning of the text.
- Do not plagiarize the text.
- Keep the parahrase as the same length as the input text as much as possible.
- Be fluent and natural sounding.`;

const instruction4 = `You are an AI writing assistant. You are expected to correct the spelling of the given text and follow these rules.
- Keep the markdown formatting of the input text in the output text.
- Do not change the meaning of the text.
- Do not plagiarize the text.
- Keep the parahrase as the same length as the input text as much as possible.
- Be fluent and natural sounding.`;


async function run_openai(instruction, input) {
    try {
        set_api_key();
        // Fetch the response from the OpenAI API with the signal from AbortController
        const response = await fetch(API_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                Authorization: `Bearer ${API_KEY}`,
            },
            body: JSON.stringify({
                model: "gpt-3.5-turbo",
                messages: [{ "role": "system", "content": instruction }, { role: "user", content: input }],
            }),
        });
        const data = await response.json();
        console.log(data.choices[0].message.content);
        return data.choices[0].message.content;
    } catch (error) {
        console.error("Error:", error);
    }
}

async function auto_complete(input) {
    return await run_openai(instruction1, input);
}

async function make_fluent(input) {
    return await run_openai(instruction2, input);
}

function paraphrase(input) {
    return run_openai(instruction3, input);
}

function correctSpelling(input) {
    return run_openai(instruction4, input);
}

set_api_key();
