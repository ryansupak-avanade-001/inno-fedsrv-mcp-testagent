#osdu_agent.py
#OsduMCPAgent
import asyncio
import os
import json
import yaml
import requests
from dotenv import load_dotenv
from mcp_agent.core.fastagent import FastAgent

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

# Set XAI_API_KEY in environment for FastAgent
os.environ["XAI_API_KEY"] = config["xai_api_key"]

# Write fastagent.config.yaml for non-sensitive settings (non-Azure)
if not IS_AZURE:
    config_path = "fastagent.config.yaml"
    config_data = {
        "mcp": {
            "servers": {
                "default": {
                    "transport": "http",
                    "url": config["default_mcp_server"],
                    "sampling": {
                        "model": config["default_model"]
                    }
                }
            }
        }
    }
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f)
    print("Debug: Wrote/updated fastagent.config.yaml")

# Self-discovery of MCP primitives
def discover_mcp_primitives(server_url):
    if not server_url.endswith('/'):
        server_url += '/'
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    methods = ["tools/list", "resources/list", "prompts/list"]
    discovered = {}
    for method in methods:
        endpoint = server_url
        payload = {"jsonrpc": "2.0", "method": method, "params": {}, "id": 1}
        print("Debug: Sending request to", endpoint, "with payload:", json.dumps(payload))
        try:
            response = requests.post(endpoint, headers=headers, json=payload)
            if response.status_code == 200:
                result = response.json().get("result", {})
                discovered[method] = result
                print("Discovered", method, ":", json.dumps(result, indent=2))
            else:
                print("Error discovering", method, ":", response.status_code, "-", response.text)
        except Exception as e:
            print("Error discovering", method, ":", str(e))
    return discovered

# Initialize FastAgent
fast = FastAgent("OsduMCPAgent")

# Define main function to run agent
async def main(fast):
    discovered = discover_mcp_primitives(config["default_mcp_server"])
    def create_agents(fast, tools_list, resources_list, prompts_list):
        formatted_tools = "\n".join(f"- Name: {tool['name']}, Description: {tool['description']}" for tool in tools_list) if tools_list else "None available."
        formatted_resources = "\n".join(f"- URI: {res['uri']}, Name: {res['name']}, Description: {res['description']}" for res in resources_list) if resources_list else "None available."
        formatted_prompts = "\n".join(f"- Name: {prompt['name']}, Description: {prompt['description']}" for prompt in prompts_list) if prompts_list else "None available."
        researcher_instruction = "Use 'tools/call' with 'list_all_wells' for queries with 'wells', 'Well Data', 'Well Resources', or 'Wells Resource'. List Resource URIs for 'list Resources'. Reply directly: Tools: {formatted_tools} Resources: {formatted_resources} Prompts: {formatted_prompts}"
        analyzer_instruction = "Analyze OSDU data with Tools like get_casings_for_well. Use 'tools/call' with 'list_all_wells' for queries with 'wells', 'Well Data', 'Well Resources', or 'Wells Resource'. List Resource URIs for 'list Resources'. Reply directly: Tools: {formatted_tools} Resources: {formatted_resources} Prompts: {formatted_prompts}"
        print("Debug: Researcher instruction:", researcher_instruction)
        print("Debug: Analyzer instruction:", analyzer_instruction)
        @fast.agent(name="researcher", instruction=researcher_instruction, servers=["default"])
        def researcher(): pass
        @fast.agent(name="analyzer", instruction=analyzer_instruction, servers=["default"])
        def analyzer(): pass
        @fast.chain(name="osdu_query", sequence=["researcher", "analyzer"])
        def osdu_query(): pass
    create_agents(fast, discovered.get("tools/list", {}).get("tools", []), discovered.get("resources/list", {}).get("resources", []), discovered.get("prompts/list", {}).get("prompts", []))
    async with fast.run() as agent:
        await agent.interactive()

# Run the agent
if __name__ == "__main__":
    asyncio.run(main(fast))
