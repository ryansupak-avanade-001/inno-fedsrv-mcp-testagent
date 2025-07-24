# osdu_agent.py
# OsduMCPAgent
import os
import json
import requests
import time
import traceback
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, BaseSingleActionAgent
from langchain_core.agents import AgentAction, AgentFinish
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.prompts import PromptTemplate
from langchain.tools import Tool

# Default fallback values
CONFIG_DEFAULTS = {
    "default_model": "grok-3-latest",
    "default_mcp_server": "http://0.0.0.0:8000/mcp/"  # Use local server for testing
}

# Detect Azure
IS_AZURE = bool(os.getenv("WEBSITE_HOSTNAME") or os.getenv("WEBSITE_SITE_NAME"))

# Load .env if not in Azure
if not IS_AZURE:
    dotenv_loaded = load_dotenv(dotenv_path=".env")
    print("Debug: .env file found and loaded:", dotenv_loaded)
    print("Debug: Current working directory:", os.getcwd())
    print("Debug: .env exists at path:", os.path.exists(".env"))

# Load from env vars first
config = {
    "default_model": os.getenv("DEFAULT_MODEL", CONFIG_DEFAULTS["default_model"]),
    "default_mcp_server": os.getenv("DEFAULT_MCP_SERVER", CONFIG_DEFAULTS["default_mcp_server"]),
    "xai_api_key": os.getenv("XAI_API_KEY")
}

# Load config.json if not in Azure
if not IS_AZURE:
    config_file = "config.json"
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            json_config = json.load(f)
            for key in config.keys():
                if key in json_config and json_config[key]:
                    print("Debug: Overwriting", key, "from config.json")
                    config[key] = json_config[key]
                else:
                    print("Debug: Skipping overwrite for", key, "(not present or empty in config.json)")

# Print masked API key for debugging
print("Debug: Final loaded XAI_API_KEY (masked):", "****" if config["xai_api_key"] else "None")

# Check for required API key
if not config.get("xai_api_key"):
    raise ValueError("XAI_API_KEY not set in env vars (including .env) or config.json. Get it from https://x.ai/api.")

# Set XAI_API_KEY in environment
os.environ["XAI_API_KEY"] = config["xai_api_key"]

# Send MCP request with retries
def send_mcp_request(method, params=None, server_url=None):
    if not server_url:
        server_url = config["default_mcp_server"]
    if not server_url.endswith('/'):
        server_url += '/'
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    payload = {"jsonrpc": "2.0", "method": method, "params": params or {}, "id": 1}
    print(f"Debug: Sending {method} request to {server_url} with payload:", json.dumps(payload, indent=2))
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            response = requests.post(server_url, headers=headers, json=payload, timeout=5)
            if response.status_code == 200:
                result = response.json().get("result", {})
                print(f"Debug: Successful {method} response:", json.dumps(result, indent=2))
                return result
            else:
                print(f"Debug: Failed {method} request on attempt {attempt + 1}: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Debug: Error on {method} request attempt {attempt + 1}: {str(e)}")
        time.sleep(1)  # Delay between retries
    print(f"Debug: Failed to retrieve {method} after {max_attempts} attempts")
    return {}

# Call Grok 3 API for semantic matching
def call_grok_3(prompt, max_retries=3):
    headers = {
        "Authorization": f"Bearer {config['xai_api_key']}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    payload = {
        "model": config["default_model"],
        "messages": [{"role": "system", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 512,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "response_format": {"type": "json_object"}
    }
    print("Debug: Using API key (masked):", "****" if config['xai_api_key'] else "None", "for model:", payload["model"])
    print("Debug: API request headers:", headers)
    print("Debug: API request URL:", "https://api.x.ai/v1/chat/completions")
    print("Debug: Sending Grok API request with payload:", json.dumps(payload, indent=2))
    for attempt in range(max_retries):
        try:
            response = requests.post("https://api.x.ai/v1/chat/completions", headers=headers, json=payload, timeout=10)
            if response.status_code == 200:
                result = response.json().get("choices", [{}])[0].get("message", {}).get("content", "{}")
                print("Debug: Grok 3 response:", result)
                try:
                    return json.loads(result)
                except json.JSONDecodeError as e:
                    print(f"Debug: Invalid JSON in Grok 3 response: {str(e)}")
                    return {"action": "error", "message": "Invalid JSON response from Grok 3"}
            else:
                print(f"Debug: Failed Grok 3 request on attempt {attempt + 1}: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Debug: Error on Grok 3 request attempt {attempt + 1}: {str(e)}")
        time.sleep(2 ** attempt)  # Exponential backoff
    print("Debug: Failed to retrieve Grok 3 response after retries")
    return {"action": "error", "message": "Failed to process query"}

# Custom agent for MCP tool invocation with Grok 3
class ExecutorAgent(BaseSingleActionAgent):
    tools: list
    instruction: str
    memory: ConversationBufferWindowMemory
    resources_list: list
    prompts_list: list

    @property
    def input_keys(self):
        return ["input"]

    def plan(self, intermediate_steps, **kwargs):
        query = kwargs.get("input", "")
        print("Debug: Available tools:", [t.name for t in self.tools])
        print("Debug: Tool descriptions:", {t.name: t.description for t in self.tools})
        print("Debug: Tool selection intent:", "list tools" if "list tools" in query.lower() or "what are the tools" in query.lower() else "other")
        print("Debug: Constructing prompt with instruction:", self.instruction)
        print("Debug: Conversation history:", self.memory.buffer_as_str)
        prompt = self.instruction.format(
            tools="\n".join(f"- Name: {t.name}, Description: {t.description}" for t in self.tools) if self.tools else "No tools available.",
            resources="\n".join(self.resources_list) if self.resources_list else "No resources available.",
            prompts="\n".join(self.prompts_list) if self.prompts_list else "No prompts available.",
            history=self.memory.buffer_as_str,
            query=query,
            tool_names=", ".join(t.name for t in self.tools) if self.tools else "None",
            agent_scratchpad=""
        )
        print("Debug: Full Grok 3 prompt sent:", prompt)
        grok_response = call_grok_3(prompt)
        print("Debug: Full Grok 3 response:", json.dumps(grok_response, indent=2))
        print("Debug: Processing action:", grok_response.get("action"))
        
        # Validate grok_response
        if not isinstance(grok_response, dict):
            print("Debug: Invalid grok_response type:", type(grok_response))
            return AgentFinish(return_values={"output": "Error: Invalid response from Grok 3"}, log="Executor: Invalid Grok 3 response")
        
        if grok_response.get("action") == "error":
            return AgentFinish(return_values={"output": grok_response.get("message", "Query processing failed")}, log="Executor: Error from Grok 3")
        
        if grok_response.get("action") == "list":
            if grok_response.get("type") == "tools":
                if not self.tools:
                    return AgentFinish(return_values={"output": "No tools available"}, log="Executor: No tools found")
                return AgentFinish(return_values={"output": [f"{t.name}: {t.description}" for t in self.tools]}, log="Executor: Returning tool list")
            elif grok_response.get("type") == "resources":
                return AgentFinish(return_values={"output": self.resources_list if self.resources_list else ["No resources available"]}, log="Executor: Returning resource list")
            elif grok_response.get("type") == "prompts":
                return AgentFinish(return_values={"output": self.prompts_list if self.prompts_list else ["No prompts available"]}, log="Executor: Returning prompt list")
        
        elif grok_response.get("action") == "tool":
            tool_name = grok_response.get("tool_name")
            tool_input = grok_response.get("tool_input", {})
            print("Debug: Tool selection details - tool_name:", tool_name, "tool_input:", json.dumps(tool_input, indent=2))
            for tool in self.tools:
                if tool.name == tool_name:
                    print("Debug: Matched tool:", tool.name)
                    if not isinstance(tool_input, dict):
                        print("Debug: Invalid tool_input type:", type(tool_input))
                        return AgentFinish(return_values={"output": f"Error: Invalid tool input for {tool_name}"}, log="Executor: Invalid tool input")
                    return AgentAction(tool=tool.name, tool_input=tool_input, log=f"Executor: Invoking {tool.name}")
            print(f"Debug: Tool {tool_name} not found")
            return AgentFinish(return_values={"output": f"Error: Tool {tool_name} not found"}, log="Executor: Tool not found")
        
        elif grok_response.get("action") == "resource":
            resource_uri = grok_response.get("resource_uri", "")
            if not resource_uri:
                print("Debug: Missing resource_uri in grok_response")
                return AgentFinish(return_values={"output": "Error: No resource URI provided"}, log="Executor: Missing resource URI")
            result = send_mcp_request("resources/read", {"uri": resource_uri})
            return AgentFinish(return_values={"output": result}, log=f"Executor: Reading resource {resource_uri}")
        
        print("Debug: No valid action in grok_response:", grok_response)
        return AgentFinish(return_values={"output": "No relevant tool or action found"}, log="Executor: No action taken")

    async def aplan(self, intermediate_steps, **kwargs):
        return self.plan(intermediate_steps, **kwargs)

# Discover MCP primitives and create LangChain tools
def discover_mcp_primitives(server_url=None):
    tools_list = send_mcp_request("tools/list", server_url=server_url).get("tools", [])
    resources_list = send_mcp_request("resources/list", server_url=server_url).get("resources", [])
    prompts_list = send_mcp_request("prompts/list", server_url=server_url).get("prompts", [])

    # Convert MCP tools to LangChain tools
    def create_tool_handler(tool_name):
        def tool_func(input_dict):
            print("Debug: MCP tool call - name:", tool_name, "params:", json.dumps(input_dict, indent=2))
            result = send_mcp_request("tools/call", {"name": tool_name, "params": input_dict})
            print("Debug: MCP tool call response:", json.dumps(result, indent=2))
            return result
        return tool_func

    tools = [
        Tool(
            name=tool["name"],
            func=create_tool_handler(tool["name"]),
            description=tool["description"]
        )
        for tool in tools_list
    ]
    print("Debug: Discovered tools:", [t.name for t in tools])
    print("Debug: Discovered resources:", [r["uri"] for r in resources_list])
    print("Debug: Discovered prompts:", [p["name"] for p in prompts_list])
    return tools, resources_list, prompts_list

# Create LangChain agent
def create_agents():
    tools, resources_list, prompts_list = discover_mcp_primitives()
    print("Debug: Agent initialized with model:", config["default_model"], "API key set:", bool(config["xai_api_key"]))
    formatted_tools = "\n".join(f"- Name: {tool.name}, Description: {tool.description}" for tool in tools) if tools else "None available."
    formatted_resources = "\n".join(f"- URI: {res['uri']}, Name: {res['name']}, Description: {res['description']}" for res in resources_list) if resources_list else "None available."
    formatted_prompts = "\n".join(f"- Name: {prompt['name']}, Description: {prompt['description']}" for prompt in prompts_list) if prompts_list else "None available."
    with open("config.json", 'r') as f:
        config_data = json.load(f)
    orchestrator_prompts = config_data.get("orchestrator_prompts", [])
    if not orchestrator_prompts:
        raise ValueError("No orchestrator_prompts found in config.json")
    instruction = "\n".join(orchestrator_prompts) + "\nTools: {tools}\nResources: {resources}\nPrompts: {prompts}\nPrevious conversation: {history}\nQuery: {query}\nTool names: {tool_names}\nAgent scratchpad: {agent_scratchpad}"
    prompt = PromptTemplate.from_template(instruction)
    resources_list = [f"{res['uri']}: {res['description']}" for res in resources_list]
    prompts_list = [f"{prompt['name']}: {prompt['description']}" for prompt in prompts_list]
    memory = ConversationBufferWindowMemory(
        k=5,
        chat_memory=ChatMessageHistory(),
        return_messages=True,
        memory_key="history",
        output_key="output"
    )
    mcp_executor = AgentExecutor(
        agent=ExecutorAgent(tools=tools, instruction=instruction, memory=memory, resources_list=resources_list, prompts_list=prompts_list),
        tools=tools,
        memory=memory,
        verbose=True,
        return_intermediate_steps=True,
        max_iterations=5
    )
    return mcp_executor, instruction

# Orchestrator function to run interactive agent loop
def main():
    mcp_executor, instruction = create_agents()
    print("Enter a query for the MCP-agent Executor. Type 'exit' to quit.")
    while True:
        query = input("mcp-agent > ")
        print("Orchestrator: Received query:", query)
        if query.lower() == "exit":
            break
        try:
            print("Orchestrator: Delegating to Executor with memory:", mcp_executor.memory.buffer_as_str)
            result = mcp_executor.invoke({"input": query})
            print("Orchestrator: Formatting and returning result")
            print("Result:", json.dumps(result["output"], indent=2))
            mcp_executor.memory.save_context({"input": query}, {"output": json.dumps(result["output"])})
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            print("Debug: Error details:", str(e), "Traceback:", traceback.format_exc())

if __name__ == "__main__":
    main()
