const API_URL = "https://api.openai.com/v1/chat/completions";
var API_KEY = localStorage.getItem("API_KEY");
var MAX_TOKENS = localStorage.getItem("MAX_TOKENS");
var API_TIMEOUT = 20000; // 20 seconds
var SHOW_HELP_ON_START = localStorage.getItem("SHOW_HELP_ON_START");

function set_api_settings() {
    if (API_KEY === "null" || API_KEY === null || API_KEY === "") {
        API_KEY = prompt("Enter your OpenAI API key to use 🤖.")
        localStorage.setItem("API_KEY", API_KEY);
    }

    if (MAX_TOKENS === "null" || MAX_TOKENS === null || MAX_TOKENS === "") {
        MAX_TOKENS = 200
        localStorage.setItem("MAX_TOKENS", MAX_TOKENS);
    }

    if (API_TIMEOUT === "null" || API_TIMEOUT === null || API_TIMEOUT === "") {
        API_TIMEOUT = 20000
        localStorage.setItem("API_TIMEOUT", API_TIMEOUT);
    }
}

function get_api_settings() {
    API_KEY = localStorage.getItem("API_KEY");
    MAX_TOKENS = localStorage.getItem("MAX_TOKENS");
    API_TIMEOUT = localStorage.getItem("API_TIMEOUT");
    SHOW_HELP_ON_START = localStorage.getItem("SHOW_HELP_ON_START");
    console.log("API_KEY: " + API_KEY);
    console.log("MAX_TOKENS: " + MAX_TOKENS);
    console.log("API_TIMEOUT: " + API_TIMEOUT);
    return { "API_KEY": API_KEY, "MAX_TOKENS": MAX_TOKENS, "API_TIMEOUT": API_TIMEOUT };
}

const instruction1 = `You are a writing assistant. You need to complete the following text and only return the part you add. Follow the rules below.
- Keep any markdown formatting syntax in the text.
- Do not change the meaning of the text.
- Do not plagiarize the text.
- Be fluent and natural sounding.
- Be brief and concise.
- Do not go over one paragraph with ${MAX_TOKENS} words.
- Only return the part you add`;

const instruction2 = `You are an AI writing assistant.  You are expected to improve the writing of the given text and follow these rules.
- Keep the markdown formatting of the input text in the output text.
- Do not change the meaning of the text.
- Do not plagiarize the text.
- Keep the parahrase as the same length as the input text as much as possible.
- Be fluent and natural sounding.
- Be brief and concise.
`;

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

const custom_prompt_instruction = `You are an AI writing assistant, and you are instructed to follow the user's instructions provided in the text, which are separated by a new line and a dash (--). Additionally, you must adhere to the following rules:
- Retain any markdown formatting syntax in the text.
- Preserve the original meaning of the text.
- Avoid plagiarism by generating original content.
- Maintain a fluent and natural-sounding writing style.
- Be brief and concise, not exceeding one paragraph or ${MAX_TOKENS} words.
- Only provide the portion of text that you add or modify.
`;

const custom_prompt_with_input_instruction = `You are an AI writing assistant. You follow the instruction and use the input text to perform the task. The instruction and the prompt are seperated by a new line and dash (--). Also follow the rules below.
- Keep any markdown formatting syntax in the text.
- Do not change the meaning of the text.
- Do not plagiarize the text.
- Be fluent and natural sounding.
- Be brief and concise.
- Do not go over one paragraph with ${MAX_TOKENS} words.
- Only return the part you add
`;


async function run_openai(instruction, input) {
    try {
        set_api_settings();
        let api_settings = get_api_settings();
        console.log("Run OpenAI...")
        console.log("Instruction: " + instruction);
        console.log("Input: " + input);
        // Create a new AbortController instance
        const controller = new AbortController();
        // Set a timeout of 10 seconds
        const timeout = setTimeout(() => {
            controller.abort();
            console.log("Request timed out.");
        }, api_settings["API_TIMEOUT"]);
        // Fetch the response from the OpenAI API with the signal from AbortController
        const response = await fetch(API_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                Authorization: `Bearer ${api_settings["API_KEY"]}`,
            },
            body: JSON.stringify({
                model: "gpt-4",
                messages: [{ "role": "system", "content": instruction }, { role: "user", content: input }],
                max_tokens: parseInt(api_settings["MAX_TOKENS"]),
            }),
            signal: controller.signal,
        });
        // Clear the timeout
        clearTimeout(timeout);
        console.log(response);
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

function userInstruction(instruction, input) {
    if (input == "" || input == null) {
        instruction = "\n--\n" + instruction;
        return run_openai(custom_prompt_instruction, instruction);
    }
    instruction = "\n--\n" + instruction + "\n--\n" + input;
    return run_openai(custom_prompt_with_input_instruction, instruction);
}

document.addEventListener("DOMContentLoaded", function () {
    set_api_settings();
    get_api_settings();
    console.log("API_KEY: " + API_KEY);
});
