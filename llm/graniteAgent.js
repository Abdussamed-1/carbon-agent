// llm/graniteAgent.js
/**
 * Fine-tuned Granite model için LangChain wrapper
 * RTX 4070 8GB optimize edilmiş
 */

import { LLM } from "@langchain/core/language_models/llms";
import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export class GraniteLLM extends LLM {
    constructor(options = {}) {
        super(options);
        this.modelPath = options.modelPath || './models/granite-esg-rtx4070';
        this.maxTokens = options.maxTokens || 150;
        this.temperature = options.temperature || 0.7;
        this.pythonPath = options.pythonPath || 'python';
        this.scriptPath = path.join(__dirname, '../scripts/granite-langchain-adapter.py');
        
        console.log('🤖 Granite LLM başlatılıyor...');
        console.log(`📁 Model path: ${this.modelPath}`);
    }
    
    _llmType() {
        return "granite-esg";
    }
    
    async _call(prompt, options = {}) {
        try {
            console.log('🔥 Granite model çağrılıyor...');
            
            const response = await this._callPythonScript(prompt, options);
            
            console.log('✅ Granite yanıt alındı');
            return response;
            
        } catch (error) {
            console.error('❌ Granite LLM hatası:', error);
            throw new Error(`Granite model error: ${error.message}`);
        }
    }
    
    async _callPythonScript(prompt, options = {}) {
        return new Promise((resolve, reject) => {
            const args = [
                this.scriptPath,
                '--model-path', this.modelPath,
                '--prompt', prompt,
                '--max-tokens', (options.maxTokens || this.maxTokens).toString(),
                '--temperature', (options.temperature || this.temperature).toString(),
                '--smart-mode'  // ESG-optimized mode
            ];
            
            console.log('🐍 Python script çalıştırılıyor...');
            
            const pythonProcess = spawn(this.pythonPath, args, {
                stdio: ['pipe', 'pipe', 'pipe']
            });
            
            let output = '';
            let errorOutput = '';
            
            pythonProcess.stdout.on('data', (data) => {
                output += data.toString();
            });
            
            pythonProcess.stderr.on('data', (data) => {
                errorOutput += data.toString();
            });
            
            pythonProcess.on('close', (code) => {
                if (code === 0) {
                    try {
                        const result = JSON.parse(output);
                        resolve(result.response || result.text || output);
                    } catch (parseError) {
                        // JSON parse edilemezse, raw output döndür
                        resolve(output.trim());
                    }
                } else {
                    reject(new Error(`Python script failed with code ${code}: ${errorOutput}`));
                }
            });
            
            pythonProcess.on('error', (error) => {
                reject(new Error(`Failed to start Python script: ${error.message}`));
            });
        });
    }
}

// Factory function
export function createGraniteLLM(config = {}) {
    return new GraniteLLM({
        modelPath: config.GRANITE_MODEL_PATH || './models/granite-esg-rtx4070',
        maxTokens: config.MAX_TOKENS || 150,
        temperature: config.TEMPERATURE || 0.7,
        pythonPath: config.PYTHON_PATH || 'python'
    });
}