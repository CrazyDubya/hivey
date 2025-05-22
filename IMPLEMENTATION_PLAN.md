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

## Phase X: Swarm-Generated System Improvements (Post Initial MVP)

This phase focuses on addressing the suggestions generated by the Swarm itself (Task ID: `09cecd52-7c8c-4df6-b35a-e696ad3e60c2`) to enhance the system's robustness, scalability, maintainability, and feature set. The suggestions are broad and will require careful prioritization.

**Overarching Goals for this Phase:**
*   Mature the system architecture for long-term viability.
*   Improve operational stability and performance.
*   Enhance developer experience and maintainability.
*   Strategically expand features and capabilities.

**Methodology:**
For each major suggestion category, the approach will generally be:
1.  **Deep Dive & Research:** Thoroughly investigate the implications, benefits, and effort required for each suggestion within the category.
2.  **Prioritization:** Based on impact, effort, and current system needs, prioritize which specific items to tackle first.
3.  **Proof of Concept (PoC):** For significant architectural changes (e.g., message queues, microservices, new databases), develop PoCs to validate the approach.
4.  **Iterative Implementation:** Implement changes incrementally, with thorough testing at each stage.

--- 

### Sub-Phase X.1: Foundational Robustness & Maintainability

*   **Target Areas:** Reliability/Robustness, Maintainability/Code Organization, Initial Security Hardening.
*   **Priority Items:**
    1.  **Formalize Dependency Management (Poetry/PDM):**
        *   **Action:** Choose between Poetry or PDM. Migrate `requirements.txt` (if any) and manage dependencies strictly.
        *   **Rationale:** Reproducible builds, better conflict resolution, cleaner project structure.
    2.  **Code Linting, Formatting, Static Typing (Black, Flake8, MyPy):**
        *   **Action:** Configure and integrate these tools into the development workflow. Set up pre-commit hooks.
        *   **Rationale:** Improves code quality, readability, and catches errors early.
    3.  **Enhanced Error Handling & Logging Review:**
        *   **Action:** Review current error handling. Implement more specific error codes and structured logging for easier debugging. Ensure comprehensive coverage.
        *   **Rationale:** Improves diagnosability and system stability.
    4.  **Input Validation & Sanitization (Initial Pass):**
        *   **Action:** Review API endpoints and critical data paths for proper input validation (e.g., using Pydantic models effectively) and sanitization where necessary.
        *   **Rationale:** Basic security hygiene.

### Sub-Phase X.2: Scalability & Performance Enhancements

*   **Target Areas:** Scalability, Asynchronous Task Processing, Database Optimization.
*   **Priority Items (May require PoCs first):**
    1.  **Message Queue for Background Tasks (RabbitMQ/Redis Streams/Kafka):**
        *   **Action:** Research and select a suitable message queue. Implement a PoC to offload task processing from FastAPI `BackgroundTasks`.
        *   **Rationale:** Decouples task submission from execution, improves fault tolerance, enables independent scaling of workers.
    2.  **Database Scalability Assessment:**
        *   **Action:** Monitor SQLite performance under simulated load. If bottlenecks appear, research and PoC migration to PostgreSQL or a suitable NoSQL alternative (e.g., for task results or knowledge base).
        *   **Rationale:** Ensures the database doesn't become a limiting factor.
    3.  **Centralized Logging & Basic Monitoring (ELK/Prometheus+Grafana - Initial Setup):**
        *   **Action:** Set up a basic centralized logging solution. Explore initial metrics collection with Prometheus.
        *   **Rationale:** Foundation for better observability and operational insight.
    4.  **Consider Containerization (Docker) for Future Needs:**
        *   **Action:** If deployment complexity increases, or if scaling requires multiple instances, revisit containerizing the service with Docker. Create Dockerfiles and `docker-compose.yml` as needed.
        *   **Rationale:** Can simplify deployment, ensure consistency across environments, and facilitate advanced scaling if required later.

### Sub-Phase X.3: Advanced Features & Architectural Evolution

*   **Target Areas:** Feature Enhancements, Data Management, Advanced Security, Microservices.
*   **Priority Items (Longer-term, iterative):**
    1.  **Knowledge Base Evolution (Graph Database - Neo4j PoC):**
        *   **Action:** If complex relationships in data become prominent, research and PoC using a graph database for parts of the knowledge base or agent interaction tracking.
        *   **Rationale:** Better suited for managing and querying highly interconnected data.
    2.  **Enhanced Authentication (OAuth 2.0/JWT):**
        *   **Action:** If requirements evolve (e.g., user accounts, third-party app integration), plan and implement OAuth 2.0 or JWT.
        *   **Rationale:** Stronger, more flexible security model.
    3.  **Microservices Architecture Exploration:**
        *   **Action:** Based on experience from X.1 and X.2, identify suitable candidates for microservices. PoC the extraction of one or two components.
        *   **Rationale:** Incremental move towards a more scalable and maintainable architecture if justified by complexity and team size.
    4.  **Investigate New Agent Types & Improved Communication:**
        *   **Action:** Based on system usage and needs, design and implement new specialized agents. Explore more sophisticated inter-agent communication protocols (building on message queues).
        *   **Rationale:** Expands system capabilities.
    5.  **Cost Optimization Strategies (LLM Usage):**
        *   **Action:** Continuously monitor LLM costs. Experiment with prompt engineering, model tiering, and explore feasibility of fine-tuning for high-volume tasks.
        *   **Rationale:** Manages operational expenses.

### Sub-Phase X.4: Continuous Improvement & Iteration

*   **Target Areas:** Ongoing Testing, Documentation, UI/UX, Security Audits.
*   **Priority Items (Ongoing):**
    1.  **Automated Integration & End-to-End Testing:**
        *   **Action:** Continuously expand test coverage. Automate testing of critical workflows.
        *   **Rationale:** Ensures reliability as the system evolves.
    2.  **Comprehensive Documentation (Living Document):**
        *   **Action:** Maintain and update all forms of documentation (code, API, architectural) as changes are made.
        *   **Rationale:** Keeps knowledge accessible and current.
    3.  **User Interface (UI) Development (If Prioritized):**
        *   **Action:** If a UI becomes a priority, plan and develop it iteratively.
    4.  **Periodic Security Audits:**
        *   **Action:** Schedule and conduct security reviews/audits as the system matures and handles more sensitive operations.

--- 

This plan provides a high-level roadmap. Each item will require more detailed planning and breakdown as it's approached. The order and specific content of sub-phases may be adjusted based on evolving priorities and findings from earlier work.
