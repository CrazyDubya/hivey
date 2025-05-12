# Hive Mind LLM Integration: Implementation Plan

This document outlines the plan to integrate Ollama (with a custom 'long-gemma' model) and X.AI (Groq-3-latest) into the Swarm-based Hive Mind system.

**Confirmed API Details:**

*   **Ollama ('long-gemma'):**
    *   List local models: `GET http://localhost:11434/api/tags`
    *   Chat completion: `POST http://localhost:11434/api/chat`
        *   Request body: `{"model": "long-gemma", "messages": [{"role": ..., "content": ...}], "stream": false}`
*   **X.AI (Groq-3-latest):**
    *   Uses OpenAI Python library.
    *   `api_key = os.getenv("XAI_API_KEY")`
    *   `base_url = "https://api.x.ai/v1"`
    *   Model: `"grok-3-latest"`

---

**Phase 1: Environment Setup & Core Abstractions**

1.  **Ollama Setup (Custom Model Focus):**
    *   **User Action:** Ensure custom "long-gemma" model is correctly served by local Ollama at `http://localhost:11434`.
    *   **Cascade Action:** Implement logic in `llm_clients.py` to:
        *   (Optional) Query Ollama's `GET /api/tags` to list available models and verify "long-gemma".
        *   Use "long-gemma" as the model identifier in `POST /api/chat` calls. Set `stream: false`.

2.  **X.AI API Access:**
    *   **User Action:** Ensure `XAI_API_KEY` environment variable is set.
    *   **Cascade Action:** Implement logic in `llm_clients.py` to call X.AI API using the `openai` library, `base_url="https://api.x.ai/v1"`, and model `"grok-3-latest"`.

3.  **Python Libraries:**
    *   **User Action:** Install/confirm `requests`, `openai`, `python-dotenv`.

4.  **`Swarm` Library/Abstraction Layer:**
    *   **User Action:** Verify the existing `swarm` library (`Swarm`, `Agent`, `Result`) is accessible.
    *   **Cascade Action (Guidance):** The `Swarm.run()` method (in the user's `swarm.py`) will need to be modified to accept a parameter indicating which LLM backend to use. It should then call the appropriate functions from `llm_clients.py`.

---

**Phase 2: Model Integration**

1.  **Ollama ("long-gemma") Integration in `Swarm`:**
    *   The `Swarm.run()` method, when directed to use Ollama, will call the Ollama interaction function from `llm_clients.py`.

2.  **X.AI (Groq-3-latest) Integration in `Swarm`:**
    *   The `Swarm.run()` method, when directed to use X.AI, will call the X.AI interaction function from `llm_clients.py`.

---

**Phase 3: Agent Configuration & Logic for Model Selection**

1.  **Model Selection Strategy:**
    *   Decide how agents (or the `OrganizerAgent`) will choose between "long-gemma" (low-edge) and "Groq-3-latest" (high-thinking). Options:
        *   Agent-Specific configuration.
        *   Task-Based complexity assessment by `OrganizerAgent`.
        *   Dynamic escalation based on `JudgePanelAgent` evaluation.
    *   Implement the chosen strategy within `OrganizerAgent.coordinate_world_building` or individual agent logic to pass the selected model backend to `client.run()`.

2.  **`NeuralNetAgent` Adaptation:**
    *   Clarify its role: Will it use one of the LLMs (e.g., "long-gemma" for simpler tasks) or remain a placeholder for other true NN models?
    *   Adapt its `perform_task` method accordingly.

---

**Phase 4: Testing & Evaluation**

1.  **Unit Tests (Model Adapters):**
    *   Write tests for functions in `llm_clients.py`, mocking API responses.

2.  **Agent-Level Tests:**
    *   For key agents (`GeographyAgent`, `CultureAgent`, `MetaAgent`, etc.):
        *   Define simple test prompts.
        *   Run agent with "long-gemma" and evaluate output.
        *   Run agent with "Groq-3-latest" and evaluate output.
        *   Compare results based on relevance, detail, instruction adherence.
    *   **`JudgePanelAgent` Enhancement:** Implement more concrete evaluation logic (e.g., keyword checking, length metrics, LLM-as-judge).

3.  **Integration Tests (Full Swarm):**
    *   Use `world_builder.py:main()` as the basis.
    *   **Scenario 1 (All Low-Edge):** Configure swarm for "long-gemma".
    *   **Scenario 2 (All High-Thinking):** Configure swarm for "Groq-3-latest".
    *   **Scenario 3 (Mixed):** Implement and test the chosen model selection strategy.
    *   Monitor: Agent communication, `context_variables`, `JudgePanelAgent` evaluations, reruns, experience capture, `InspirationAgent` and `MetaAgent` outputs.

4.  **Specific Feature Tests:**
    *   Test dynamic agent creation (`InspirationAgent`, `OrganizerAgent`).
    *   Test self-learning mechanism (experience capture, `MetaAgent` processing).
    *   Test error handling/fallback in `OrganizerAgent`.

---

**Phase 5: Iteration and Refinement**

1.  **Performance Analysis:** Qualitatively assess output quality and speed differences.
2.  **Prompt Engineering:** Refine `OVERARCHING_PROMPT` and agent-specific instructions for optimal performance from both models.
3.  **Knowledge Base & Self-Improvement:** Evaluate the effectiveness of the self-learning loop.
