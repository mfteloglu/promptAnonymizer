import sys
import json
from promptAnonymizer.pipeline.langgraph_app import run_pipeline


def main():
    text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Error on server at IP 192.168.1.10. Email me at user@example.com"
    result = run_pipeline(text)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
