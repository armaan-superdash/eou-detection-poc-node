import * as ort from "onnxruntime-node";
import fs from "node:fs";
import { AutoTokenizer, PreTrainedTokenizer } from "@huggingface/transformers";
import { type TMessage } from "./tokenizerTypes";

function normalizeMessage(message: TMessage): TMessage {
    message.content = message.content.replace("'", "").toLowerCase();
    return message;
}

function softmaxOptimized(logits: Float32Array) {
    const len = logits.length;
    let max = logits[0];
    for (let i = 1; i < len; i++) {
        if (logits[i] > max) max = logits[i];
    }

    const result = new Float32Array(len);
    let sum = 0;

    for (let i = 0; i < len; i++) {
        const val = Math.exp(logits[i] - max);
        result[i] = val;
        sum += val;
    }

    for (let i = 0; i < len; i++) {
        result[i] /= sum;
    }

    return result;
}


function formatChatMessage(chatMessages: TMessage[], tokenizer: PreTrainedTokenizer): string {
    const normalizedMessages = chatMessages.map(normalizeMessage);
    const formattedConversationText = tokenizer.apply_chat_template(normalizedMessages, {
        add_generation_prompt: false,
        tokenize: false
    });
    console.info("Conversation Text Formatted: ", formattedConversationText);
    return formattedConversationText.toString().split("<|im_end|>")[0];
}

async function calculateMessageEOU(messages: TMessage[], session: ort.InferenceSession, tokenizer: PreTrainedTokenizer): Promise<number> {
    const MAX_HISTORY_TOKENS = 512;
    const contextualText = formatChatMessage(messages, tokenizer);
    
    
    const inputs = await tokenizer(contextualText, {
        return_tensors: "pt", 
        truncation: true,
        max_length: MAX_HISTORY_TOKENS,
        padding: false
    });

    
    const inputIds = inputs.input_ids.data as number[];
    const inputTensor = new ort.Tensor('int64', 
        new BigInt64Array(inputIds.map(id => BigInt(id))), 
        [1, inputIds.length]
    );

    const feeds = {
        input_ids: inputTensor,
    };

    const results = await session.run(feeds);
    const logitsTensor = (results as unknown as {
        prob: {
            cpuData: Float32Array
        }
    }).prob;
    console.log(results);
    
    const logitsData = Array.from(logitsTensor.cpuData as Float32Array).flat();
    console.log("logits data: ", logitsData);
    // const lastTokenLogits = logitsData.slice(startIndex, startIndex + vocabSize);
    
    
    // let probabilities = softmaxOptimized(lastTokenLogits);

    // const endTokenIds = await tokenizer.encode("<|im_end|>", {
    //     add_special_tokens: false
    // });
    // const endTokenId = endTokenIds[endTokenIds.length - 1];
    
    return logitsData[0];
}

async function simulateCall(session: ort.InferenceSession, tokenizer: PreTrainedTokenizer, message: string): Promise<number[]> {
    
    const contextMessages: TMessage[] = [{
        role: "user",
        content: message
    }];
    
    try {
        const startTime = Date.now();
        const eouProbability = await calculateMessageEOU(contextMessages, session, tokenizer);
        const endTime = Date.now();
        return [eouProbability * 100, endTime - startTime];
    } catch (error) {
        console.error("Error calculating EOU:", error);
        throw error;
    }
}

async function initialize(): Promise<void> {
    try {
        console.log("Initializing ONNX Runtime...");
        const session = await ort.InferenceSession.create("./model.onnx");
        console.log("Loading tokenizer...");
        const tokenizer = await AutoTokenizer.from_pretrained("livekit/turn-detector", {
            local_files_only: false,
        });
        const testContent = fs.readFileSync("./testFile").toString().split("\n");
        const startTime = Date.now();
        const result = [];
        for(const test of testContent) {
            const [prob, timeTaken] = await simulateCall(session, tokenizer, test);
            result.push({
                sentence: test,
                prob,
                timeTaken
            });
        }
        const jsonResult = JSON.stringify(result);
        fs.writeFileSync("./result.json", jsonResult);
        const endTime = Date.now();
        console.log("Time Took: ", endTime - startTime, "ms");
    } catch (error) {
        console.error("Initialization error:", error);
    }
}

initialize();