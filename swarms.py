import openai
import sqlite3
import logging
from typing import List, Dict, Any, Optional, Callable, Tuple, Union
import json
import os
import time
import numpy as np
from dataclasses import dataclass, field, asdict
from datetime import datetime
from dotenv import load_dotenv
from .utils import get_embedding, cosine_similarity, client as openai_client
from .llm_clients import call_ollama_chat, call_xai_chat
import re
# Load environment variables
load_dotenv()

# Configure OpenAI API (This global openai.api_key might be redundant if client from utils is used consistently)
# It's good practice to ensure it's set for any direct openai legacy calls if they exist elsewhere, but new calls should use the client.
if os.getenv("OPENAI_API_KEY"):
    openai.api_key = os.getenv("OPENAI_API_KEY")
else:
    logger.warning("OPENAI_API_KEY not found in environment variables.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SwarmMind")

# Constants
EMBEDDING_MODEL = "text-embedding-ada-002"
CHAT_MODEL = "xai/grok-3-latest"
DB_NAME = "swarmmind.db"
DEFAULT_LLM_MODEL = "xai/grok-3-latest"

@dataclass
class Message:
    role: str
    content: str

@dataclass
class Result:
    value: str
    context_variables: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Function:
    name: str
    description: str
    handler: Callable

class Agent:
    def __init__(
        self,
        name: str,
        instructions: str,
        agent_type: str = "WorkerAgent",
        llm_model_identifier: str = DEFAULT_LLM_MODEL,
        max_memory_tokens: int = 2000,
        supervisor_agent: Optional['Agent'] = None,
        db_path: str = DB_NAME
    ):
        self.name = name
        self.instructions = instructions
        self.agent_type = agent_type
        self.llm_model_identifier = llm_model_identifier
        self.short_term_memory: List[Dict[str, Any]] = []
        self.long_term_memory: List[Dict[str, Any]] = []
        self.creation_time = datetime.now().isoformat()
        self.task_count = 0
        self.success_rate = 0.0
        self.last_active = None
        self.embeddings_cache = {}
        self.supervisor_agent = supervisor_agent
        self.db_path = db_path

    def remember(self, info: Dict[str, Any], long_term: bool = False):
        """Store information in agent's memory"""
        info["timestamp"] = datetime.now().isoformat()
        
        if long_term:
            self.long_term_memory.append(info)
        else:
            self.short_term_memory.append(info)
            # Limit short-term memory to last 10 items
            if len(self.short_term_memory) > 10:
                self.short_term_memory.pop(0)
    
    def recall(self, query: str = None, long_term: bool = False, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve information from agent's memory, optionally using semantic search"""
        memory = self.long_term_memory if long_term else self.short_term_memory
        
        if not query or not memory:
            return memory[:limit]
        
        # Get embedding for query
        query_embedding = get_embedding(query)
        
        # Calculate similarity scores
        scores = []
        for i, entry in enumerate(memory):
            content = entry.get("content", "")
            if not content:
                scores.append(0)
                continue
                
            # Cache embeddings to avoid redundant API calls
            if content not in self.embeddings_cache:
                self.embeddings_cache[content] = get_embedding(content)
            
            entry_embedding = self.embeddings_cache[content]
            similarity = cosine_similarity(query_embedding, entry_embedding)
            scores.append(similarity)
        
        # Sort by similarity
        sorted_pairs = sorted(zip(scores, memory), key=lambda x: x[0], reverse=True)
        return [item for _, item in sorted_pairs[:limit]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary for serialization"""
        return {
            "name": self.name,
            "instructions": self.instructions,
            "tier": self.agent_type,
            "functions": [],
            "creation_time": self.creation_time,
            "task_count": self.task_count,
            "success_rate": self.success_rate,
            "last_active": self.last_active
        }

    def _get_llm_response(self, task_description: str, model_params: Dict[str, Any] = None) -> str:
        """Get a response from the LLM"""
        model_params = model_params or {}
        system_prompt = (
            f"You are {self.name}, a specialized agent in the SwarmMind collective intelligence system.\n\n"
            f"Your instructions: {self.instructions}\n\n"
            f"Task: {task_description}\n\n"
            "Respond with clear, concise, and detailed information related to your specialization. "
            "Focus on producing high-quality output that will contribute to the collective task."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_description}
        ]
        
        response_content = None
        
        try:
            logger.info(f"Agent {self.name} using model: {self.llm_model_identifier} for task: {task_description[:100]}...")
            
            if self.llm_model_identifier.startswith("ollama/"):
                model_name = self.llm_model_identifier.split("/", 1)[1]
                ollama_response = call_ollama_chat(model_name=model_name, messages=messages, options=model_params)
                if ollama_response and ollama_response.get("message") and isinstance(ollama_response["message"], dict):
                    response_content = ollama_response["message"].get("content")
                elif ollama_response and ollama_response.get("error"):
                    logger.error(f"Ollama API error for {self.name} ({model_name}): {ollama_response.get('error')}")
                else:
                    logger.warning(f"Unexpected Ollama response structure for {self.name} ({model_name}): {ollama_response}")
            
            elif self.llm_model_identifier.startswith("xai/"):
                model_name = self.llm_model_identifier.split("/", 1)[1]
                # X.AI client uses OpenAI SDK, so model_params like temperature might be passed differently or at client instantiation.
                # For now, call_xai_chat doesn't take 'options' directly like ollama.
                # This might need adjustment in call_xai_chat if such params are crucial.
                xai_response = call_xai_chat(model_name=model_name, messages=messages)
                if xai_response and xai_response.get("choices") and xai_response["choices"]:
                    message = xai_response["choices"][0].get("message")
                    if message and isinstance(message, dict):
                        response_content = message.get("content")
                else:
                    logger.warning(f"Unexpected X.AI response structure for {self.name} ({model_name}): {xai_response}")
            
            else: # Default to OpenAI or if identifier has no prefix / starts with "openai/"
                openai_model_name = self.llm_model_identifier
                if openai_model_name.startswith("openai/"):
                    openai_model_name = openai_model_name.split("/", 1)[1]
                
                completion = openai_client.chat.completions.create(
                    model=openai_model_name,
                    messages=messages,
                    temperature=model_params.get("temperature", 0.7),
                    max_tokens=model_params.get("max_tokens", 1024) # Ensure max_tokens is passed if needed
                )
                response_content = completion.choices[0].message.content

            if response_content:
                self.remember({"role": "assistant", "content": response_content})
                self.last_active = datetime.now()
                logger.info(f"Agent {self.name} received response: {response_content[:100]}...")
                return response_content.strip()
            else:
                logger.error(f"Agent {self.name} received no content from LLM ({self.llm_model_identifier}).")
                return "Error: No content received from LLM."

        except openai.APIError as e:
            logger.error(f"OpenAI API Error for agent {self.name} ({self.llm_model_identifier}): {e}")
            return f"Error: OpenAI API Error - {e}"
        except Exception as e:
            logger.error(f"General error in _get_llm_response for agent {self.name} ({self.llm_model_identifier}): {e}")
            return f"Error: An unexpected error occurred - {e}"

    def run(self, task_description: str, context: Optional[List[Dict[str, Any]]] = None) -> str:
        """Run the agent on a given task"""
        # Update agent status
        self.task_count += 1
        self.last_active = datetime.now().isoformat()
        
        # Combine context
        full_context = {}
        if context:
            full_context.update(context)
            
        # Get relevant experiences
        relevant_experiences = self.recall(query=task_description, limit=3)
        experiences_text = "\n".join([f"Task: {exp.get('task', 'Unknown')}\nContent: {exp.get('content', '')[:100]}...\n" for exp in relevant_experiences])
        
        # Prepare the prompt with enhanced context
        system_prompt = (
            f"You are {self.name}, a specialized agent in the SwarmMind collective intelligence system.\n\n"
            f"Your instructions: {self.instructions}\n\n"
            f"Task: {task_description}\n\n"
        )
        
        if experiences_text:
            system_prompt += f"Relevant past experiences:\n{experiences_text}\n\n"
            
        if full_context:
            system_prompt += f"Context variables:\n{json.dumps(full_context, indent=2)}\n\n"
            
        system_prompt += (
            "Respond with clear, concise, and detailed information related to your specialization. "
            "Focus on producing high-quality output that will contribute to the collective task."
        )
        
        return self._get_llm_response(task_description, model_params={"temperature": 0.7, "max_tokens": 1024})

class Swarm:
    def __init__(self):
        """Initialize the swarm with database connection and load agents"""
        self.agents: Dict[str, Agent] = {}
        self.supervisors: Dict[str, Agent] = {}  
        self.meta_agents: Dict[str, Agent] = {}
        self.context_variables: Dict[str, Any] = {}
        self.global_memory: List[Dict[str, Any]] = []
        self.knowledge_base = KnowledgeBase()
        self.pending_agents: List[Dict[str, Any]] = []
        
        # Initialize database
        self._init_db()
        
        # Load experiences
        self.experiences = self.knowledge_base.get_experiences()
        
        # Create essential agents
        self._initialize_essential_agents()
        
        logger.info(f"SwarmMind initialized with {len(self.agents)} worker agents, "
                   f"{len(self.supervisors)} supervisors, and {len(self.meta_agents)} meta agents")

    def _init_db(self):
        """Initialize database tables if they don't exist"""
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        # Create experiences table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task TEXT,
                agent_name TEXT,
                content TEXT,
                confidence_score REAL,
                feedback TEXT,
                embedding TEXT,
                timestamp TEXT
            )
        ''')
        
        # Create agents table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE,
                instructions TEXT,
                tier TEXT,
                creation_time TEXT,
                task_count INTEGER,
                success_rate REAL,
                last_active TEXT
            )
        ''')
        
        conn.commit()
        conn.close()

    def _initialize_essential_agents(self):
        """Create the essential agents for the swarm to function"""
        # Meta-agents (highest tier)
        self.add_agent(
            name="OrganizerAgent",
            instructions=(
                "You are the Organizer Agent responsible for coordinating the swarm. "
                "You analyze tasks, delegate to appropriate agents, and ensure coherent outputs. "
                "You can propose new agents when needed based on task requirements."
            ),
            agent_type="meta"
        )
        
        self.add_agent(
            name="JudgeAgent",
            instructions=(
                "You evaluate the outputs of other agents, providing confidence scores and feedback. "
                "You ensure the quality, relevance, and coherence of content generated by the swarm. "
                "Score from 0.001 to 0.999 and provide specific improvement feedback."
            ),
            agent_type="meta"
        )
        
        self.add_agent(
            name="InspiratorAgent", 
            instructions=(
                "You are a highly creative and analytical AI. Your role is to identify gaps in the swarm's capabilities "
                "and propose new, specialized agents that would enhance the collective intelligence. When proposing, "
                "suggest a suitable name, detailed instructions, an agent_type (worker/supervisor). For the LLM model, "
                "you must choose between 'low' (for simpler tasks, maps to ollama/long-gemma) or 'high' (for complex tasks, maps to xai/grok-3-latest). "
                "Provide a clear rationale for your proposal, including your choice of 'low' or 'high' for the model."
            ),
            agent_type="meta",
            llm_model_identifier=DEFAULT_LLM_MODEL
        )
        
        # Supervisor agents (middle tier)
        self.add_agent(
            name="WorldSupervisor",
            instructions=(
                "You supervise agents involved in world-building tasks. "
                "You coordinate their efforts and ensure consistency across geographical, cultural, historical, "
                "and technological aspects of created worlds."
            ),
            agent_type="supervisor"
        )
        
        self.add_agent(
            name="NarrativeSupervisor",
            instructions=(
                "You supervise agents involved in narrative creation. "
                "You ensure coherent storylines, character development, and plot progression. "
                "You coordinate between character, plot, and dialogue agents."
            ),
            agent_type="supervisor"
        )
        
        # Worker agents (base tier)
        self.add_agent(
            name="GeographyAgent",
            instructions=(
                "Create detailed geographical aspects of worlds including continents, climates, "
                "terrain features, natural resources, and ecosystems. Consider how geography "
                "influences other aspects of the world such as culture and politics."
            ),
            llm_model_identifier="ollama/long-gemma",
            agent_type="worker"
        )
        
        self.add_agent(
            name="CultureAgent",
            instructions=(
                "Develop rich cultural elements including customs, traditions, languages, "
                "arts, religions, social structures, and values. Create distinct cultural groups "
                "and explain how they interact with each other and their environment."
            ),
            llm_model_identifier="xai/grok-3-latest",
            agent_type="worker"
        )
        
        self.add_agent(
            name="HistoryAgent",
            instructions=(
                "Craft detailed historical timelines including major events, wars, discoveries, "
                "technological advancements, political shifts, and cultural developments. "
                "Create a sense of how the past shapes the present world state."
            ),
            llm_model_identifier="xai/grok-3-latest",
            agent_type="worker"
        )
        
        self.add_agent(
            name="CharacterAgent",
            instructions=(
                "Create compelling characters with distinct personalities, motivations, backgrounds, "
                "relationships, strengths, and flaws. Ensure characters feel authentic and reflect "
                "their cultural and historical context."
            ),
            llm_model_identifier="xai/grok-3-latest",
            agent_type="worker"
        )

    def add_agent(self, name: str, instructions: str, agent_type: str = "worker", llm_model_identifier: str = DEFAULT_LLM_MODEL, functions: List[Dict[str, Any]] = None) -> Agent:
        """Add a new agent to the swarm"""
        # 'functions' param is kept for now if needed by older parts, but Agent itself doesn't use it in the new constructor
        agent = Agent(name=name, instructions=instructions, agent_type=agent_type, llm_model_identifier=llm_model_identifier)
        
        # Add to appropriate collection based on tier
        if agent_type == "meta":
            self.meta_agents[name] = agent
        elif agent_type == "supervisor":
            self.supervisors[name] = agent
        else:  # worker
            self.agents[name] = agent
            
        # Store in database
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO agents 
            (name, instructions, tier, creation_time, task_count, success_rate, last_active)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (name, instructions, agent_type, agent.creation_time, 0, 0.0, None))
        conn.commit()
        conn.close()
        
        logger.info(f"Added {agent_type} agent: {name}")
        return agent

    def run_agent(self, agent_name: str, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run a specific agent on a given task"""
        agent = self._get_agent(agent_name)
        if not agent:
            logger.error(f"Agent {agent_name} not found")
            return {"error": f"Agent {agent_name} not found"}
        
        logger.info(f"Running agent: {agent_name} on task: {task[:50]}...")
        
        # Update agent status
        agent.task_count += 1
        agent.last_active = datetime.now().isoformat()
        
        # Combine context
        full_context = self.context_variables.copy()
        if context:
            full_context.update(context)
            
        # Get relevant experiences
        relevant_experiences = self._get_relevant_experiences(task, agent_name, limit=3)
        experiences_text = self._format_experiences(relevant_experiences)
        
        # Get relevant memories
        relevant_memories = agent.recall(query=task, limit=3)
        memories_text = self._format_memories(relevant_memories)
        
        # Prepare the prompt with enhanced context
        system_prompt = (
            f"You are {agent_name}, a specialized agent in the SwarmMind collective intelligence system.\n\n"
            f"Your instructions: {agent.instructions}\n\n"
            f"Task: {task}\n\n"
        )
        
        if experiences_text:
            system_prompt += f"Relevant past experiences:\n{experiences_text}\n\n"
            
        if memories_text:
            system_prompt += f"Your relevant memories:\n{memories_text}\n\n"
            
        if full_context:
            system_prompt += f"Context variables:\n{json.dumps(full_context, indent=2)}\n\n"
            
        system_prompt += (
            "Respond with clear, concise, and detailed information related to your specialization. "
            "Focus on producing high-quality output that will contribute to the collective task."
        )
        
        try:
            # Call OpenAI API
            response = agent.run(task, context)
            
            output = response
            
            # Store in agent's memory
            agent.remember({"task": task, "content": output}, long_term=True)
            
            # Evaluate the output using Judge Agent
            evaluation = self._evaluate_output(agent_name, output, task)
            
            # Update agent success rate
            agent.success_rate = (agent.success_rate * (agent.task_count - 1) + 
                                evaluation["confidence_score"]) / agent.task_count
            
            # Save the experience to knowledge base
            self.knowledge_base.save_experience(
                task=task,
                agent_name=agent_name,
                content=output,
                confidence_score=evaluation["confidence_score"],
                feedback=evaluation["feedback"]
            )
            
            # Update global memory
            self.global_memory.append({
                "timestamp": datetime.now().isoformat(),
                "agent": agent_name,
                "task": task,
                "output": output,
                "evaluation": evaluation
            })
            
            # Limit global memory size
            if len(self.global_memory) > 20:
                self.global_memory.pop(0)
                
            return {
                "agent_name": agent_name,
                "output": output,
                "evaluation": evaluation
            }
            
        except Exception as e:
            logger.error(f"Error running agent {agent_name}: {e}")
            return {"error": str(e), "agent_name": agent_name}

    def _get_agent(self, name: str) -> Optional[Agent]:
        """Get an agent by name from any tier"""
        if name in self.agents:
            return self.agents[name]
        elif name in self.supervisors:
            return self.supervisors[name]
        elif name in self.meta_agents:
            return self.meta_agents[name]
        return None

    def _evaluate_output(self, agent_name: str, output: str, task: str) -> Dict[str, Any]:
        """Evaluate an agent's output using the Judge Agent"""
        judge = self.meta_agents.get("JudgeAgent")
        if not judge:
            logger.warning("JudgeAgent not found, cannot evaluate output.")
            return {
                "confidence_score": 0.75, # Default if no judge
                "feedback": "Evaluation not available (JudgeAgent not found)"
            }
            
        prompt = (
            f"Task given to {agent_name}: {task}\n\n"
            f"Output from {agent_name}:\n\n{output}\n\n"
            "Evaluate this output's quality, relevance, and coherence. "
            "Provide a confidence score between 0.001 (poor) and 0.999 (excellent) "
            "and specific feedback including strengths and areas for improvement. "
            "Format your response clearly, ensuring the confidence score is easily parsable (e.g., 'Confidence Score: 0.85')."
        )
        
        import re # MODIFIED: Ensure re is imported in this scope
        evaluation_text = None
        judge_model_identifier = judge.llm_model_identifier
        messages = [
            {"role": "system", "content": judge.instructions},
            {"role": "user", "content": prompt}
        ]

        try:
            logger.info(f"JudgeAgent ({judge_model_identifier}) evaluating output from {agent_name} for task: {task[:50]}...")

            if judge_model_identifier.startswith("ollama/"):
                model_name = judge_model_identifier.split("/", 1)[1]
                ollama_response = call_ollama_chat(model_name=model_name, messages=messages, options={"temperature": 0.3})
                if ollama_response and ollama_response.get("message") and isinstance(ollama_response["message"], dict):
                    evaluation_text = ollama_response["message"].get("content")
                elif ollama_response and ollama_response.get("error"):
                    logger.error(f"Ollama API error for JudgeAgent ({model_name}): {ollama_response.get('error')}")
            
            elif judge_model_identifier.startswith("xai/"):
                model_name = judge_model_identifier.split("/", 1)[1]
                xai_response = call_xai_chat(model_name=model_name, messages=messages, temperature=0.3)
                if xai_response and xai_response.get("choices") and xai_response["choices"]:
                    message_obj = xai_response["choices"][0].get("message")
                    if message_obj and isinstance(message_obj, dict):
                        evaluation_text = message_obj.get("content")
            
            else: # Default to OpenAI or if identifier has no prefix / starts with "openai/"
                openai_model_name = judge_model_identifier
                if openai_model_name.startswith("openai/"):
                    openai_model_name = openai_model_name.split("/", 1)[1]
                elif "ollama/" in openai_model_name or "xai/" in openai_model_name:
                    # This case should ideally not be hit if JudgeAgent is configured correctly for OpenAI path
                    logger.warning(f"JudgeAgent model '{openai_model_name}' looks like non-OpenAI, but processing via OpenAI path. Defaulting to {CHAT_MODEL} (actual OpenAI model).")
                    # CHAT_MODEL should be an OpenAI compatible model name if we reach here as a fallback
                    if CHAT_MODEL.startswith("xai/") or CHAT_MODEL.startswith("ollama/"):
                        # If CHAT_MODEL itself is not OpenAI, this is a deeper config issue. Use a hardcoded OpenAI model.
                        openai_model_name = "gpt-3.5-turbo" # Safest OpenAI default
                        logger.error(f"CHAT_MODEL '{CHAT_MODEL}' is not OpenAI. JudgeAgent forced to use {openai_model_name}.")
                    else:
                        openai_model_name = CHAT_MODEL.split("/",1)[1] if "/" in CHAT_MODEL else CHAT_MODEL
                
                response = openai_client.chat.completions.create(
                    model=openai_model_name,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=500
                )
                evaluation_text = response.choices[0].message.content

            if not evaluation_text:
                logger.error(f"JudgeAgent ({judge_model_identifier}) received no content.")
                raise ValueError("No content received from LLM for evaluation.")

            # Parse the evaluation to extract score and feedback
            confidence_score = 0.75  # Default fallback
            feedback = evaluation_text
            
            # Try to extract confidence score (improved regex)
            score_match = re.search(r"(?:confidence\s*score|score):?\s*(\d(?:\.\d+)?)", evaluation_text, re.IGNORECASE)
            if score_match:
                try:
                    confidence_score = float(score_match.group(1))
                    confidence_score = max(0.001, min(0.999, confidence_score))
                except ValueError:
                    logger.warning(f"Could not parse confidence score from: {score_match.group(1)}")
                    pass # Keep default score
            else:
                logger.warning(f"No confidence score found in JudgeAgent output: {evaluation_text[:100]}...")
                    
            return {
                "confidence_score": confidence_score,
                "feedback": feedback
            }
            
        except Exception as e:
            logger.error(f"Error evaluating output with JudgeAgent ({judge_model_identifier}): {e}")
            return {
                "confidence_score": 0.5, # Default on error
                "feedback": f"Error in evaluation: {str(e)}"
            }

    def _get_relevant_experiences(self, task: str, agent_name: str = None, limit: int = 5) -> List[Dict[str, Any]]:
        """Get relevant experiences for a given task and optional agent"""
        return self.knowledge_base.semantic_search(task, agent_name, limit)

    def _format_experiences(self, experiences: List[Dict[str, Any]]) -> str:
        """Format experiences for inclusion in prompts"""
        if not experiences:
            return ""
            
        formatted = []
        for exp in experiences:
            formatted.append(
                f"Agent: {exp['agent_name']}\n"
                f"Task: {exp['task']}\n"
                f"Output: {exp['content'][:100]}...\n"
                f"Score: {exp['confidence_score']}\n"
            )
            
        return "\n".join(formatted)

    def _format_memories(self, memories: List[Dict[str, Any]]) -> str:
        """Format memories for inclusion in prompts"""
        if not memories:
            return ""
            
        formatted = []
        for mem in memories:
            formatted.append(
                f"Task: {mem.get('task', 'Unknown')}\n"
                f"Content: {mem.get('content', '')[:100]}...\n"
                f"Timestamp: {mem.get('timestamp', 'Unknown')}\n"
            )
            
        return "\n".join(formatted)

    def propose_new_agent(self, task: str) -> Dict[str, Any]:
        """Use the InspirationAgent to propose a new specialized agent"""
        inspirator = self.meta_agents.get("InspiratorAgent")
        if not inspirator:
            logger.error("InspirationAgent not found")
            return {"error": "InspirationAgent not found"}
            
        prompt = (
            f"Analyze this task and propose a new specialized agent that would enhance the swarm's capabilities:\n\n"
            f"Task: {task}\n\n"
            f"Current agents: {', '.join(list(self.agents.keys()) + list(self.supervisors.keys()))}\n\n"
            "Provide your response in this format:\n"
            "Agent Name: [name]\n"
            "Instructions: [detailed instructions]\n"
            "Agent Type: [worker/supervisor]\n" 
            "LLM Model Identifier: [low/high]\n"  # MODIFIED: Prompt for low/high
            "Rationale: [why this agent would be valuable]"
        )
        
        # Use the InspiratorAgent's own LLM to generate the proposal
        # The _get_llm_response method is part of the Agent class
        # We need to ensure the InspiratorAgent can call it or we call it in a similar way
        # For simplicity, we'll replicate the LLM call logic here using inspirator's model ID
        
        response_text = inspirator._get_llm_response(prompt) # Using agent's own capability

        if response_text.startswith("Error:"):
             logger.error(f"InspiratorAgent failed to generate proposal: {response_text}")
             return {"status": "error", "message": f"InspiratorAgent error: {response_text}"}
        
        proposal = response_text
        
        # Parse the proposal
        import re
        name_match = re.search(r"Agent Name:\s*(.+)", proposal, re.IGNORECASE)
        instructions_match = re.search(r"Instructions:\s*(.+?)(?=Agent Type:|LLM Model Identifier:|Rationale:|$)", proposal, re.DOTALL | re.IGNORECASE)
        agent_type_match = re.search(r"Agent Type:\s*(worker|supervisor)", proposal, re.IGNORECASE) 
        llm_identifier_match = re.search(r"LLM Model Identifier:\s*([\w\-\/\.]+)", proposal, re.IGNORECASE) 
        rationale_match = re.search(r"Rationale:\s*(.+)", proposal, re.DOTALL | re.IGNORECASE)
        
        if name_match and instructions_match and agent_type_match:
            # MODIFIED: Map low/high to actual model identifiers
            suggested_model_tier = llm_identifier_match.group(1).strip().lower() if llm_identifier_match else "high"
            
            if suggested_model_tier == "low":
                final_llm_model_identifier = "ollama/long-gemma"
            else: # Default to 'high' if not 'low' or if not specified
                final_llm_model_identifier = "xai/grok-3-latest"
                if suggested_model_tier != "high" and llm_identifier_match: # Log if it wasn't 'high' but defaulted
                     logger.warning(f"InspiratorAgent suggested LLM tier '{suggested_model_tier}', defaulting to 'high' (xai/grok-3-latest).")

            agent_proposal = {
                "name": name_match.group(1).strip(),
                "instructions": instructions_match.group(1).strip(),
                "agent_type": agent_type_match.group(1).strip().lower(), 
                "llm_model_identifier": final_llm_model_identifier, # MODIFIED: Use mapped identifier
                "rationale": rationale_match.group(1).strip() if rationale_match else "No rationale provided"
            }
            
            self.pending_agents.append(agent_proposal)
            
            return {
                "status": "success",
                "proposal": agent_proposal
            }
        else:
            logger.error(f"Failed to parse agent proposal. Raw: {proposal}")
            return {
                "status": "error",
                "message": "Failed to parse agent proposal",
                "raw_proposal": proposal
            }
            
    def approve_agent(self, agent_index: int) -> Dict[str, Any]:
        """Approve and create a pending agent"""
        if agent_index < 0 or agent_index >= len(self.pending_agents):
            return {"error": "Invalid agent index"}
            
        agent_spec = self.pending_agents[agent_index]
        new_agent = self.add_agent(
            name=agent_spec["name"],
            instructions=agent_spec["instructions"],
            agent_type=agent_spec["agent_type"], 
            llm_model_identifier=agent_spec.get("llm_model_identifier", DEFAULT_LLM_MODEL) 
        )
        
        # Remove from pending list
        self.pending_agents.pop(agent_index)
        
        return {
            "status": "success",
            "message": f"Agent {new_agent.name} created successfully",
            "agent": new_agent.to_dict()
        }

    def organize_task(self, task: str) -> Dict[str, Any]:
        """Use the OrganizerAgent to break down a task and delegate to appropriate agents"""
        organizer = self.meta_agents.get("OrganizerAgent")
        if not organizer:
            logger.error("OrganizerAgent not found")
            return {"error": "OrganizerAgent not found"}
            
        # Get list of available agents
        available_agents = {
            "workers": [a.name for a in self.agents.values()],
            "supervisors": [a.name for a in self.supervisors.values()]
        }
        
        prompt = (
            f"Analyze this task and organize a workflow to accomplish it effectively:\n\n"
            f"Task: {task}\n\n"
            f"Available agents: {json.dumps(available_agents, indent=2)}\n\n"
            "Break down the task into subtasks and assign them to appropriate agents. "
            "For each subtask, specify:\n"
            "1. The agent to handle it\n"
            "2. The specific subtask description\n"
            "3. The order of execution\n\n"
            "Provide your workflow plan in JSON format."
        )
        
        try:
            response = openai_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": organizer.instructions},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            workflow_plan = response.choices[0].message['content']
            
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'```json\n(.*?)\n```', workflow_plan, re.DOTALL)
            if json_match:
                workflow_json = json_match.group(1)
            else:
                # Try to find anything that looks like JSON
                json_match = re.search(r'(\{.*\})', workflow_plan, re.DOTALL)
                if json_match:
                    workflow_json = json_match.group(1)
                else:
                    workflow_json = workflow_plan
            
            try:
                workflow = json.loads(workflow_json)
            except json.JSONDecodeError:
                logger.error(f"Could not parse workflow JSON: {workflow_json}")
                # Fallback to a basic workflow
                workflow = {
                    "workflow": [
                        {
                            "step": 1,
                            "agent": next(iter(self.agents), "GeographyAgent"),
                            "subtask": task
                        }
                    ]
                }
            
            # Execute the workflow
            results = self._execute_workflow(workflow, task)
            
            # Combine results
            combined_result = self._combine_results(results, task)
            
            return {
                "status": "success",
                "workflow": workflow,
                "results": results,
                "combined_result": combined_result
            }
                
        except Exception as e:
            logger.error(f"Error organizing task: {e}")
            return {"error": str(e)}

    def _execute_workflow(self, workflow: Dict[str, Any], original_task: str) -> List[Dict[str, Any]]:
        """Execute a workflow by running each agent in sequence"""
        results = []
        
        # Extract workflow steps
        steps = workflow.get("workflow", [])
        
        # Sort steps by order if available
        steps.sort(key=lambda x: x.get("step", 0))
        
        for step in steps:
            agent_name = step.get("agent", "")
            subtask = step.get("subtask", original_task)
            
            # Skip if agent doesn't exist
            if not self._get_agent(agent_name):
                logger.warning(f"Agent {agent_name} not found, skipping step")
                continue
                
            # Update context with previous results
            context = {
                "original_task": original_task,
                "previous_results": results,
                "current_step": step
            }
            
            # Run the agent
            result = self.run_agent(agent_name, subtask, context)
            results.append(result)
            
            # Add short delay to avoid rate limits
            time.sleep(0.5)
            
        return results

    def _combine_results(self, results: List[Dict[str, Any]], task: str) -> Dict[str, Any]:
        """Combine results from multiple agents into a coherent output"""
        # If only one result, return it directly
        if len(results) == 1:
            return results[0]
            
        # Prepare inputs for combination
        inputs = []
        for result in results:
            agent_name = result.get("agent_name", "Unknown")
            output = result.get("output", "")
            inputs.append(f"Agent: {agent_name}\nOutput: {output}\n")
            
        combined_input = "\n".join(inputs)
        
        # Use a supervisor if available, otherwise OrganizerAgent
        supervisor = None
        if "WorldSupervisor" in self.supervisors:
            supervisor = self.supervisors["WorldSupervisor"]
        elif "NarrativeSupervisor" in self.supervisors:
            supervisor = self.supervisors["NarrativeSupervisor"]
        else:
            supervisor = self.meta_agents.get("OrganizerAgent")
            
        if not supervisor:
            # Simple concatenation as fallback
            return {
                "combined_output": combined_input,
                "method": "simple_concatenation"
            }
            
        prompt = (
            f"Task: {task}\n\n"
            f"Combine the following outputs from different agents into a coherent, unified response:\n\n"
            f"{combined_input}\n\n"
            "Ensure the combined output is cohesive, eliminates redundancy, and resolves any contradictions. "
            "The final output should read as a unified response that accomplishes the original task."
        )
        
        try:
            response = openai_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": supervisor.instructions},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            
            combined_output = response.choices[0].message['content']
            
            return {
                "combined_output": combined_output,
                "method": f"combined_by_{supervisor.name}"
            }
            
        except Exception as e:
            logger.error(f"Error combining results: {e}")
            return {
                "error": str(e),
                "method": "error_in_combination"
            }

class KnowledgeBase:
    def __init__(self):
        self.conn = sqlite3.connect(DB_NAME)
        self.cursor = self.conn.cursor()

    def get_experiences(self) -> List[Dict[str, Any]]:
        """Retrieve all experiences from the knowledge base"""
        self.cursor.execute("SELECT id, task, agent_name, content, confidence_score, feedback, embedding, timestamp FROM experiences")
        rows = self.cursor.fetchall()
        experiences = []
        for row in rows:
            experience = {
                "id": row[0],
                "task": row[1],
                "agent_name": row[2],
                "content": row[3],
                "confidence_score": row[4],
                "feedback": row[5],
                "embedding": row[6], # Stored as JSON string
                "timestamp": row[7]
            }
            experiences.append(experience)
        return experiences

    def save_experience(self, task: str, agent_name: str, content: str, confidence_score: float, feedback: str):
        """Save an experience to the knowledge base"""
        content_embedding = get_embedding(content) # MODIFIED: Use utility
        embedding_json = json.dumps(content_embedding) # MODIFIED: Serialize to JSON string

        self.cursor.execute('''
            INSERT INTO experiences (task, agent_name, content, confidence_score, feedback, timestamp, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (task, agent_name, content, confidence_score, feedback, datetime.now().isoformat(), embedding_json))
        self.conn.commit()

    def semantic_search(self, query: str, agent_name: str = None, limit: int = 5) -> List[Dict[str, Any]]:
        """Perform a semantic search for experiences"""
        query_embedding = get_embedding(query) # MODIFIED: Use utility
        if not any(query_embedding):
            logger.warning("Query embedding failed, returning no search results.")
            return []

        experiences_data = self.get_experiences()
        if not experiences_data:
            return []

        if agent_name:
            experiences_data = [exp for exp in experiences_data if exp['agent_name'] == agent_name]

        scores = []
        valid_experiences_for_scoring = []

        for exp in experiences_data:
            exp_content = exp.get("content", "")
            if not exp_content: 
                continue 

            exp_embedding_json = exp.get("embedding") # This is a JSON string
            exp_embedding = None

            if exp_embedding_json:
                try:
                    exp_embedding = json.loads(exp_embedding_json) # MODIFIED: Deserialize from JSON string
                except (TypeError, json.JSONDecodeError):
                    logger.warning(f"Failed to deserialize embedding for experience ID {exp.get('id')}. Generating new one.")
                    # Fallback: generate embedding if stored one is corrupt/missing
                    exp_embedding = get_embedding(exp_content) # MODIFIED: Use utility
            else:
                exp_embedding = get_embedding(exp_content) # MODIFIED: Use utility
            
            if not exp_embedding or not any(exp_embedding):
                continue
            
            similarity = cosine_similarity(query_embedding, exp_embedding) # MODIFIED: Use utility
            scores.append(similarity)
            valid_experiences_for_scoring.append(exp)
        
        if not scores:
            return []
            
        sorted_pairs = sorted(zip(scores, valid_experiences_for_scoring), key=lambda x: x[0], reverse=True)
        return [item for _, item in sorted_pairs[:limit]]

    def close(self):
        self.conn.close()


# Example Usage (Optional - for testing)
if __name__ == '__main__':
    # Ensure environment variables OPENAI_API_KEY and XAI_API_KEY are set
    # Ensure Ollama server is running and 'long-gemma' model is available (`ollama pull long-gemma`)

    logger.info("Initializing SwarmMind System for testing...")
    swarm = Swarm()
    kb = KnowledgeBase() # Swarm already initializes its own kb, this one is for direct testing if needed

    logger.info("SwarmMind System Initialized.")
    logger.info(f"Found {len(swarm.agents)} worker agents, {len(swarm.supervisors)} supervisor agents, {len(swarm.meta_agents)} meta agents.")

    # --- Test Agent Runs --- #
    logger.info("\n--- Testing GeographyAgent (Ollama long-gemma) ---")
    geo_task = "Describe the geography of a newly discovered volcanic island chain named 'Aethelgard'."
    geo_result = swarm.run_agent("GeographyAgent", geo_task)
    logger.info(f"GeographyAgent Result: {json.dumps(geo_result, indent=2)}")

    logger.info("\n--- Testing CultureAgent (X.AI grok-3-latest) ---")
    culture_task = "Develop the basic cultural tenets of a seafaring people inhabiting the Aethelgard islands."
    culture_result = swarm.run_agent("CultureAgent", culture_task)
    logger.info(f"CultureAgent Result: {json.dumps(culture_result, indent=2)}")

    logger.info("\n--- Testing HistoryAgent (X.AI grok-3-latest) ---")
    history_task = "Outline a brief mythical history for the Aethelgard islanders, explaining their origins."
    history_result = swarm.run_agent("HistoryAgent", history_task)
    logger.info(f"HistoryAgent Result: {json.dumps(history_result, indent=2)}")

    # --- Test KnowledgeBase --- #
    logger.info("\n--- Testing KnowledgeBase Operations ---")
    # Experiences should have been saved via swarm.run_agent -> _evaluate_output -> kb.save_experience
    # For a direct test, let's save one more.
    logger.info("Saving a direct test experience to KB...")
    kb.save_experience(
        task="Manual Test Entry", 
        agent_name="TestLogger", 
        content="This is a manually logged experience about ancient Aethelgardian fishing techniques.",
        confidence_score=0.95,
        feedback="Positive manual entry"
    )
    logger.info("Performing semantic search for 'Aethelgard geography or culture or history or fishing techniques'")
    # Use swarm's KB instance for searching experiences logged by agents
    search_results = swarm.knowledge_base.semantic_search("Aethelgard geography or culture or history or fishing techniques", limit=5)
    logger.info(f"Semantic Search Results: {json.dumps(search_results, indent=2)}")

    # --- Test Agent Proposal and Approval --- #
    logger.info("\n--- Testing Agent Proposal and Approval ---")
    proposal_task = "The Aethelgardian islanders need a specialized agent to design their unique magical system based on geothermal vents."
    logger.info(f"Proposing new agent for task: {proposal_task}")
    proposal_result = swarm.propose_new_agent(proposal_task)
    logger.info(f"Agent Proposal Result: {json.dumps(proposal_result, indent=2)}")

    if proposal_result.get("status") == "success" and swarm.pending_agents:
        logger.info("Approving first pending agent...")
        # Assuming the proposal included a valid llm_model_identifier or defaults correctly
        approval_result = swarm.approve_agent(0) 
        logger.info(f"Agent Approval Result: {json.dumps(approval_result, indent=2)}")
        logger.info(f"Agents after approval: {list(swarm.agents.keys())}")

        # --- Test the newly approved agent --- 
        if approval_result.get("status") == "success" and "agent" in approval_result and "name" in approval_result["agent"]:
            newly_approved_agent_name = approval_result["agent"]["name"]
            logger.info(f"\n--- Testing newly approved agent: {newly_approved_agent_name} (X.AI grok-3-latest expected) ---")
            magma_task = (
                "Develop the core principles of the geothermal magic system for Aethelgard. "
                "Describe how energy is harnessed from vents, list 2-3 example abilities linked to geothermal power, "
                "and mention one significant potential risk or limitation of using this magic."
            )
            magma_result = swarm.run_agent(newly_approved_agent_name, magma_task)
            logger.info(f"Agent {newly_approved_agent_name} Result: {json.dumps(magma_result, indent=2)}")
        else:
            logger.warning("Could not retrieve name of newly approved agent, or approval failed. Skipping its test.")
    else:
        logger.info("No pending agents to approve or proposal was not successful.")

    logger.info("SwarmMind System testing complete. All KBs closed.")

    kb.close() # Close the directly instantiated kb
    swarm.knowledge_base.close() # Close the swarm's kb instance
