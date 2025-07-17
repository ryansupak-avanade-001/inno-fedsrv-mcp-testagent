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
    "default_mcp_server": "https://your-app.azurewebsites.net/mcp/"  # Replace with your deployed server URL
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
                    print(f"Debug: Overwriting {key} from config.json")
                    config[key] = json_config[key]
                else:
                    print(f"Debug: Skipping overwrite for {key} (not present or empty in config.json)")

print("Debug: Final loaded XAI_API_KEY (masked):", "****" if config["xai_api_key"] else "None")

# Check for required API key
if not config.get("xai_api_key"):
    raise ValueError("XAI_API_KEY not set in env vars (including .env) or config.json. Get it from https://x.ai/api.")

# Set XAI_API_KEY in environment for FastAgent to read (avoids writing to secrets.yaml)
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
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream"
    }
    methods = ["tools/list", "resources/list", "prompts/list"]
    discovered = {}
    for method in methods:
        endpoint = server_url
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": {},
            "id": 1
        }
        response = requests.post(endpoint, headers=headers, json=payload, stream=True)
        if response.status_code == 200:
            result = None
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data:'):
                        data = line[5:].strip()
                        try:
                            result = json.loads(data).get("result", {})
                            discovered[method] = result
                            print(f"Discovered {method}: {json.dumps(result, indent=2)}")
                        except json.JSONDecodeError:
                            print(f"Error parsing data for {method}: {data}")
            if result is None:
                print(f"No data found in SSE for {method}")
        else:
            print(f"Error discovering {method}: {response.status_code} - {response.text}")
    return discovered

# Initialize FastAgent
fast = FastAgent("OsduMCPAgent")

# Define agents
@fast.agent(
    name="researcher",
    instruction="Gather OSDU data using available Resources and Prompts. Summarize findings.",
    servers=["default"]
)
@fast.agent(
    name="analyzer",
    instruction="Analyze gathered OSDU data using Tools like get_casings_for_well. Provide insights.",
    servers=["default"]
)
@fast.chain(
    name="osdu_query",
    sequence=["researcher", "analyzer"]
)
async def main():
    discover_mcp_primitives(config["default_mcp_server"])
    async with fast.run() as agent:
        await agent.interactive()

if __name__ == "__main__":
    asyncio.run(main())
