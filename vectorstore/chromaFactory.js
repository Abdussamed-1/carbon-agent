import { ChromaClient } from "chromadb";

export function getChromaClient(config) {
  return new ChromaClient({ path: config.CHROMA_DB_PATH });
} 