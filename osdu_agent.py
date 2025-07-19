#osdu_agent.py
#OsduMCPAgent
import os
import json
import requests
import time
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, BaseSingleActionAgent
from langchain_core.agents import AgentAction, AgentFinish
from langchain.tools import StructuredTool
from langchain.prompts import PromptTemplate

# Default fallback values
CONFIG_DEFAULTS = {
    "default_model": "xai.grok-4",
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

# Custom agent for MCP tool invocation without LLM
class SimpleMCPAgent(BaseSingleActionAgent):
    tools: list
    instruction: str

    @property
    def input_keys(self):
        return ["input"]

    def plan(self, intermediate_steps, **kwargs):
        query = kwargs.get("input", "")
        if "wells" in query.lower() or "well data" in query.lower():
            for tool in self.tools:
                if tool.name == "list_all_wells":
                    return AgentAction(tool=tool.name, tool_input={}, log="Invoking list_all_wells")
        if "casings" in query.lower():
            for tool in self.tools:
                if tool.name == "get_casings_for_well":
                    return AgentAction(tool=tool.name, tool_input={"well_id": "well1"}, log="Invoking get_casings_for_well")
        if "add" in query.lower():
            for tool in self.tools:
                if tool.name == "add_numbers":
                    return AgentAction(tool=tool.name, tool_input={"a": 1, "b": 2}, log="Invoking add_numbers")
        if "list resources" in query.lower():
            return AgentFinish(return_values={"output": self.instruction}, log="Returning instruction")
        return AgentFinish(return_values={"output": "No relevant tool or action found"}, log="No action taken")

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
            return send_mcp_request("tools/call", {"name": tool_name, "params": input_dict})
        return tool_func

    tools = [
        StructuredTool.from_function(
            func=create_tool_handler(tool["name"]),
            name=tool["name"],
            description=tool["description"]
        )
        for tool in tools_list
    ]
    print("Debug: Discovered tools:", [t.name for t in tools])
    print("Debug: Discovered resources:", [r["uri"] for r in resources_list])
    print("Debug: Discovered prompts:", [p["name"] for p in prompts_list])
    return tools, resources_list, prompts_list

# Create LangChain agents
def create_agents():
    tools, resources_list, prompts_list = discover_mcp_primitives()
    formatted_tools = "\n".join(f"- Name: {tool.name}, Description: {tool.description}" for tool in tools) if tools else "None available."
    formatted_resources = "\n".join(f"- URI: {res['uri']}, Name: {res['name']}, Description: {res['description']}" for res in resources_list) if resources_list else "None available."
    formatted_prompts = "\n".join(f"- Name: {prompt['name']}, Description: {prompt['description']}" for prompt in prompts_list) if prompts_list else "None available."
    instruction = (
        "Analyze OSDU data using available tools. For queries about 'wells', 'Well Data', 'Well Resources', or 'Wells Resource', use the 'list_all_wells' tool via 'tools/call'. "
        "For queries requiring detailed analysis (e.g., casings), use 'get_casings_for_well'. For arithmetic, use 'add_numbers'. "
        "List resource URIs for 'list Resources' queries. Reply directly with results or formatted output: "
        "Tools: {formatted_tools}\nResources: {formatted_resources}\nPrompts: {formatted_prompts}"
    ).format(formatted_tools=formatted_tools, formatted_resources=formatted_resources, formatted_prompts=formatted_prompts)
    print("Debug: Agent instruction:", instruction)

    # Create agents using SimpleMCPAgent
    researcher = SimpleMCPAgent(tools=tools, instruction=instruction)
    analyzer = SimpleMCPAgent(tools=tools, instruction=instruction)
    researcher_executor = AgentExecutor(
        agent=researcher, tools=tools, verbose=True, return_intermediate_steps=True
    )
    analyzer_executor = AgentExecutor(
        agent=analyzer, tools=tools, verbose=True, return_intermediate_steps=True
    )
    return researcher_executor, analyzer_executor, instruction

# Main function to run interactive agent
def main():
    researcher, analyzer, instruction = create_agents()
    print("Type 'researcher' or 'analyzer' to switch agents, or enter a query. Type 'exit' to quit.")
    current_agent = researcher
    agent_name = "researcher"
    while True:
        query = input(f"{agent_name} > ")
        if query.lower() == "exit":
            break
        if query.lower() in ["researcher", "analyzer"]:
            current_agent = researcher if query.lower() == "researcher" else analyzer
            agent_name = query.lower()
            print(f"Switched to {agent_name} agent")
            continue
        try:
            result = current_agent.invoke({"input": query})
            print("Result:", json.dumps(result["output"], indent=2))
        except Exception as e:
            print(f"Error processing query: {str(e)}")

if __name__ == "__main__":
    main()
