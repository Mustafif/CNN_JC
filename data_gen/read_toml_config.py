#!/usr/bin/env python3
"""
Helper script to read TOML config files and output as JSON for MATLAB.
Usage: python3 read_toml_config.py <config_file.toml>
"""
import sys
import json
import toml

def main():
    if len(sys.argv) != 2:
        print(json.dumps({"error": "Usage: python3 read_toml_config.py <config_file.toml>"}))
        sys.exit(1)
    
    config_file = sys.argv[1]
    
    try:
        # Read TOML file
        config = toml.load(config_file)
        
        # Output as JSON
        print(json.dumps(config, indent=2))
        
    except FileNotFoundError:
        print(json.dumps({"error": f"File not found: {config_file}"}))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": f"Error reading TOML: {str(e)}"}))
        sys.exit(1)

if __name__ == "__main__":
    main()
