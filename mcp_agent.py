import os
import json
import requests
import time
import traceback
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, BaseSingleActionAgent
from langchain_core.agents import AgentAction, AgentFinish
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import ChatMessageHistory
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
    return {"action": "error", "message": "Failed to process prompt"}

# Custom agent for MCP tool invocation with Grok 3
class ExecutorAgent(BaseSingleActionAgent):
    tools: List[Tool]
    instruction: str
    formatter_prompt: str
    memory: ConversationBufferWindowMemory
    resources_list: List[str]
    prompts_list: List[str]

    def __init__(self, tools: List[Tool], instruction: str, formatter_prompt: str, memory: ConversationBufferWindowMemory, resources_list: List[str], prompts_list: List[str]):
        super().__init__(tools=tools, instruction=instruction, formatter_prompt=formatter_prompt, memory=memory, resources_list=resources_list, prompts_list=prompts_list)
        self.tools = tools
        self.instruction = instruction
        self.formatter_prompt = formatter_prompt
        self.memory = memory
        self.resources_list = resources_list
        self.prompts_list = prompts_list

    @property
    def input_keys(self):
        return ["input"]

    def plan(self, intermediate_steps, **kwargs):
        query = kwargs.get("input", "")
        print("Debug: Available tools:", [t.name for t in self.tools])
        print("Debug: Tool descriptions:", {t.name: t.description for t in self.tools})
        print("Debug: Tool selection intent:", "list tools" if "list tools" in query.lower() or "what are the tools" in query.lower() else "other")
        print("Debug: Raw instruction string:", repr(self.instruction))
        print("Debug: Raw formatter prompt:", repr(self.formatter_prompt))
        print("Debug: Conversation history:", self.memory.buffer_as_str)
        try:
            prompt = f"{self.instruction}\nTools: {', '.join(f'Name: {t.name}, Description: {t.description}' for t in self.tools) if self.tools else 'No tools available.'}\nResources: {', '.join(self.resources_list) if self.resources_list else 'No resources available.'}\nPrompts: {', '.join(self.prompts_list) if self.prompts_list else 'No prompts available.'}\nPrevious conversation: {self.memory.buffer_as_str}\nQuery: {query}\nTool names: {', '.join(t.name for t in self.tools) if self.tools else 'None'}\nAgent scratchpad: "
            print("Debug: Raw formatted prompt:", repr(prompt))
        except Exception as e:
            print(f"Debug: Template formatting error: {str(e)}")
            return AgentFinish(return_values={"output": f"Error: Invalid template processing {str(e)}"}, log=f"Executor: Template formatting error: {str(e)}")
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
        
        actions = grok_response.get("actions", []) if "actions" in grok_response else [{"action": grok_response.get("action"), "type": grok_response.get("type"), "tool_name": grok_response.get("tool_name"), "tool_input": grok_response.get("tool_input", {}), "resource_uri": grok_response.get("resource_uri", ""), "message": grok_response.get("message", "")}]
        print("Debug: Processing actions:", actions)
        combined_output = []
        
        for action_item in actions:
            action = action_item.get("action")
            if action == "list":
                action_type = action_item.get("type")
                print("Debug: Handling list action with type:", action_type)
                if action_type == "tools":
                    if not self.tools:
                        combined_output.append("No tools available")
                    else:
                        combined_output.extend([f"{t.name}: {t.description}" for t in self.tools])
                elif action_type == "resources":
                    combined_output.extend(self.resources_list if self.resources_list else ["No resources available"])
                elif action_type == "prompts":
                    combined_output.extend(self.prompts_list if self.prompts_list else ["No prompts available"])
                else:
                    combined_output.append(f"Error: Invalid list type {action_type}")
            elif action == "tool":
                tool_name = action_item.get("tool_name")
                tool_input = action_item.get("tool_input", {})
                print("Debug: Tool selection details - tool_name:", tool_name, "tool_input:", json.dumps(tool_input, indent=2))
                for tool in self.tools:
                    if tool.name == tool_name:
                        print("Debug: Matched tool:", tool.name)
                        if not isinstance(tool_input, dict):
                            print("Debug: Invalid tool_input type:", type(tool_input))
                            combined_output.append(f"Error: Invalid tool input for {tool_name}")
                        else:
                            return AgentAction(tool=tool.name, tool_input=tool_input, log=f"Executor: Invoking {tool.name}")
                print(f"Debug: Tool {tool_name} not found")
                combined_output.append(f"Error: Tool {tool_name} not found")
            elif action == "resource":
                resource_uri = action_item.get("resource_uri", "")
                if not resource_uri:
                    print("Debug: Missing resource_uri in action")
                    combined_output.append("Error: No resource URI provided")
                else:
                    result = send_mcp_request("resources/read", {"uri": resource_uri})
                    combined_output.append(result)
            else:
                print("Debug: Invalid action in action_item:", action_item)
                combined_output.append(f"Error: Invalid action {action}")
        
        if combined_output:
            print("Debug: Returning combined output:", combined_output)
            # Second LLM trip to format the output
            formatter_prompt = f"{self.formatter_prompt}\nOutput: {json.dumps(combined_output)}"
            print("Debug: Raw formatter prompt sent:", repr(formatter_prompt))
            formatted_response = call_grok_3(formatter_prompt)
            print("Debug: Formatted Grok 3 response:", json.dumps(formatted_response, indent=2))
            if isinstance(formatted_response, dict) and "tools" in formatted_response and "resources" in formatted_response:
                # Convert formatted_response to JSON string to avoid ValidationError
                formatted_output = json.dumps(formatted_response)
                return AgentFinish(return_values={"output": formatted_output}, log="Executor: Returning formatted results")
            else:
                print("Debug: Invalid formatted response:", formatted_response)
                return AgentFinish(return_values={"output": json.dumps(combined_output)}, log="Executor: Returning unformatted results due to invalid format")
        
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
    formatter_prompts = config_data.get("formatter_prompts", [])
    if not orchestrator_prompts:
        raise ValueError("No orchestrator_prompts found in config.json")
    if not formatter_prompts:
        raise ValueError("No formatter_prompts found in config.json")
    print("Debug: Raw orchestrator_prompts:", repr(orchestrator_prompts))
    print("Debug: Raw formatter_prompts:", repr(formatter_prompts))
    instruction = "\n".join(orchestrator_prompts)
    formatter_prompt = "\n".join(formatter_prompts)
    print("Debug: Raw instruction template created:", repr(instruction))
    print("Debug: Raw formatter prompt created:", repr(formatter_prompt))
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
        agent=ExecutorAgent(
            tools=tools,
            instruction=instruction,
            formatter_prompt=formatter_prompt,
            memory=memory,
            resources_list=resources_list,
            prompts_list=prompts_list
        ),
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
            mcp_executor.memory.save_context({"input": query}, {"output": result["output"]})
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            print("Debug: Error details:", str(e), "Traceback:", traceback.format_exc())

if __name__ == "__main__":
    main()
