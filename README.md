# MCP Agent

## Introduction

The MCP Agent (`mcp_agent.py`) is a Python-based application designed to interact with any Anthropic MCP server using JSON-RPC 2.0, enabling dynamic discovery and invocation of tools, resources, and prompts. Its initial implementation is tested against an OSDU-specific MCP server, providing access to OSDU data via tools (`add_numbers`, `get_casings_for_well`, `list_all_wells`) and resources (`osdu:wells`, `osdu:trajectories`, `osdu:casings`). It uses Langchain for agent orchestration, Grok 3 for semantic query matching, and `requests` for JSON-RPC communication. The agent is a generic implementation for interacting with any Anthropic MCP server via JSON-RPC 2.0, with its initial implementation tested against an OSDU-specific MCP server. It supports natural language queries (e.g., "Can you list Tools?", "list wells") for MCP tools and resources, with basic context awareness (storing conversation history). Advanced context-aware follow-ups (e.g., "How many are over 500 feet?" after "list wells") and multi-step query handling are upcoming. It operates with minimal dependencies.

## Setup

### Prerequisites

- **Python**: Version 3.13.5 or later (assuming dependency compatibility).
- **MCP Server**: Running at `https://inno-fedsrv-mcp-api.azurewebsites.net/mcp/` (OSDU-specific) or any MCP server URL (e.g., `http://0.0.0.0:8000/mcp/` for local testing).
- **xAI API Key**: Required for Grok 3 semantic matching.

### Installation

1. Clone the repository or copy the project files:
   ```bash
   git clone <repository_url>
   cd inno-fedsrv-mcp-testagent
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Required packages:
   - `langchain>=0.2.0`
   - `langchain-core>=0.2.0`
   - `langchain-community>=0.2.0`
   - `requests>=2.31.0`
   - `python-dotenv>=1.0.1`

4. Verify dependencies:
   ```bash
   pip show langchain langchain-core langchain-community requests python-dotenv
   ```

### Configuration

1. **Create `.env` file** (optional, if not using `config.json`):
   ```bash
   XAI_API_KEY=your_xai_api_key
   DEFAULT_MODEL=grok-3-latest
   DEFAULT_MCP_SERVER=https://inno-fedsrv-mcp-api.azurewebsites.net/mcp/
   ```

2. **Update `config.json`**:
   - Ensure `xai_api_key` is set (or use `.env`).
   - Set `default_mcp_server` to your MCP server URL (e.g., OSDU server or custom MCP server).
   - `orchestrator_prompts` and `executor_prompts` are pre-configured for semantic matching and expandability.

   Example `config.json`:
   ```json
   {
     "default_model": "grok-3-latest",
     "default_mcp_server": "https://inno-fedsrv-mcp-api.azurewebsites.net/mcp/",
     "default_mcp_serverOFF": "http://0.0.0.0:8000/mcp/",
     "xai_api_key": "your_xai_api_key",
     "orchestrator_prompts": [
       "You are an OSDU data analyzer agent for an MCP server. Map user queries to intents and select actions (tool call, resource read, or list primitives).",
       "Analyze the query and conversation history. Determine intent and map to:",
       "- Tool call: If query matches a tool description, return action 'tool', tool_name, tool_input (from schema).",
       "- Resource read: If query asks for data from a resource URI, return action 'resource', resource_uri.",
       "- List primitives: If query asks to list tools (e.g., 'What are the Tools available?', 'list tools', 'show tools', 'Can you list Tools?'), resources, or prompts, return action 'list', type 'tools', 'resources', or 'prompts'.",
       "Use semantic understanding to handle variations (e.g., 'show wells' = list_all_wells).",
       "Extract parameters from query (e.g., 'casings for well2' = tool_input {'well_id': 'well2'}).",
       "Return JSON: {action: 'tool'|'resource'|'list'|'error', tool_name: string, tool_input: dict, resource_uri: string, type: string, message: string}",
       "Ensure queries asking for available tools (e.g., 'What are the Tools available?') return action 'list' with type 'tools'.",
       "Do not map queries asking to list tools (e.g., 'list tools', 'What are the Tools?') to tools like 'list_all_wells' requiring well_id."
     ],
     "executor_prompts": [
       "Confirm tool or resource selection for the sub-task: {sub_task}.",
       "Return JSON: {action: 'tool'|'resource', tool_name: string, tool_input: dict, resource_uri: string}"
     ]
   }
   ```

## Usage

1. **Start the MCP Server**: Ensure the server is running (e.g., `app.py` for OSDU or your custom MCP server). Test connectivity:
   ```bash
   curl -X POST https://inno-fedsrv-mcp-api.azurewebsites.net/mcp/ -H "Content-Type: application/json" -d '{"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": 1}'
   ```

2. **Run the Agent**:
   ```bash
   python3 mcp_agent.py
   ```

3. **Interact with the Agent**:
   - Prompt: `mcp-agent >`
   - Example queries (OSDU-specific examples; adapt to your MCP server’s tools/resources):
     - `Can you list Tools?`: Lists tools (e.g., `add_numbers`, `get_casings_for_well`, `list_all_wells`).
     - `list resources`: Lists resources (e.g., `osdu:wells`, `osdu:trajectories`, `osdu:casings`).
     - `list wells` or `show wells`: Invokes `list_all_wells`.
     - `get casings for well1`: Invokes `get_casings_for_well`.
     - `list osdu:wells`: Reads `osdu:wells` resource.
   - Type `exit` to quit.

4. **Debugging**:
   - Check logs for:
     - Tool/resource discovery (e.g., `Debug: Discovered tools: [...]`).
     - Orchestrator actions (e.g., `Orchestrator: Received query`).
     - Grok 3 prompts/responses (e.g., `Debug: Grok 3 response: {...}`).
     - Memory state (e.g., `Orchestrator: Delegating to Executor with memory: [...]`).

## Inner Workings

### Agents

- **Orchestrator** (`main` function):
  - Manages the interactive loop, receiving user queries (line 229).
  - Maintains conversation memory using `ConversationBufferMemory` with `ChatMessageHistory` (line 205).
  - Delegates queries to the Executor, logs actions, and formats results (lines 236–246).
- **Executor** (`ExecutorAgent` class):
  - Uses Grok 3 via `call_grok_3` for semantic matching of single-step queries (lines 85–106).
  - Maps queries to MCP actions (tool call, resource read, list primitives) based on Grok 3’s JSON response (lines 108–134).
  - Executes tool calls via `send_mcp_request` (lines 67–83).
  - Dynamically discovers tools, resources, and prompts from any MCP server using JSON-RPC 2.0, making it adaptable to non-OSDU MCP servers by updating `default_mcp_server` in `config.json`.

### Data Flow

1. **Query Input**: User enters a query (e.g., `Can you list Tools?`) at `mcp-agent >`.
2. **Orchestrator**: Receives query, logs it (line 236), and delegates to `mcp_executor` with memory (line 238).
3. **Executor**:
   - Constructs a prompt from `orchestrator_prompts` in `config.json`, including tools, resources, prompts, history, query, tool_names, and agent_scratchpad for single-step query processing (line 115).
   - Sends prompt to Grok 3, parses response (e.g., `{action: "list", type: "tools"}`).
   - Executes action (e.g., return tool list, call `tools/call`, read resource).
4. **Orchestrator**: Formats result as JSON (line 243), updates memory (line 246), and returns to user.

## Upcoming Intended Features

- **Advanced context-aware follow-ups**: Process follow-up queries (e.g., "How many are over 500 feet?" after "list wells") by computing results from conversation history using Grok 3.
- **Orchestrator final step summarization**: Handle multi-step queries (e.g., "Get wells, match casings, sort by depth") with a final LLM call to aggregate/format results, using a `needs_formatting` flag in Grok 3 responses.
- **Unrelated query handling**: Provide meaningful responses for queries outside MCP capabilities (e.g., "What is the weather today?") by redirecting to relevant MCP actions or offering concise general answers.
- **Memory overflow handling**: Implement `ConversationSummaryBufferMemory` to summarize large conversation histories, preventing token limit issues.
- **Additional contingencies**: Support data privacy (e.g., sanitizing sensitive data), parallel execution (e.g., concurrent tool calls), and robust validation of tool inputs.
- **Support for other major LLMs like ChatGPT**: Extend the agent to use alternative LLMs (e.g., OpenAI’s ChatGPT) for semantic matching, configurable via `config.json`.
