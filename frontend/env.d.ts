declare namespace NodeJS {
  interface ProcessEnv {
    BACKEND_URL?: string;
    LLM_BASE_URL?: string;
    LLM_API_KEY?: string;
    LLM_MODEL?: string;
  }
}
