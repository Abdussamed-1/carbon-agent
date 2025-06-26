import { AgentExecutor, createReactAgent } from "langchain/agents";
import { RAGRetrieverTool } from "./tools/ragRetriever.js";
import { StrategyGeneratorTool } from "./tools/strategyGenerator.js";
import { pull } from "langchain/hub";

export async function createDynamicAgent({ llm, chromaManager, embeddings }) {
  const tools = [
    new RAGRetrieverTool({ chromaManager, embeddings }),
    new StrategyGeneratorTool({ llm }),
    // Yeni tool eklemek için buraya ekle
  ];

  const prompt = await pull("hwchase17/react");
  const agent = await createReactAgent({ llm, tools, prompt });
  const agentExecutor = new AgentExecutor({ agent, tools, verbose: true });
  return agentExecutor;
} 