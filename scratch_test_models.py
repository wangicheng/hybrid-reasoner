import traceback
from google import genai
import sys
import os

# Add src to path so we can import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from core.api_utils import get_current_api_key

def test_models():
    api_key = get_current_api_key()
    client = genai.Client(api_key=api_key)
    models = client.models.list()
    print([m.name for m in models if "embed" in m.name])

if __name__ == '__main__':
    test_models()
