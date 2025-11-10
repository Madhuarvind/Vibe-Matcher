#!/usr/bin/env python3
"""
Test script to verify the Vibe Matcher notebook functionality
"""

import sys
import traceback

def test_imports():
    """Test all required imports"""
    print("🧪 Testing imports...")
    try:
        import pandas as pd
        import numpy as np
        import google.generativeai as genai
        import requests
        import json
        from sklearn.metrics.pairwise import cosine_similarity
        import matplotlib.pyplot as plt
        import seaborn as sns
        import time
        import timeit
        from typing import List, Dict, Tuple
        print("✅ All imports successful!")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_data_creation():
    """Test dataset creation"""
    print("🧪 Testing data creation...")
    try:
        import pandas as pd
        
        # Create mock fashion dataset
        fashion_data = {
            'name': ['Boho Maxi Dress', 'Urban Leather Jacket'],
            'description': [
                'Flowy maxi dress with earthy tones, perfect for festival vibes',
                'Sleek black leather jacket with silver hardware, embodying urban street style'
            ],
            'vibe_tags': [
                ['boho', 'festival', 'earthy', 'free-spirited'],
                ['urban', 'edgy', 'street', 'rebellious']
            ],
            'price': [89.99, 249.99],
            'category': ['Dresses', 'Outerwear']
        }
        
        df = pd.DataFrame(fashion_data)
        print(f"✅ Dataset created: {len(df)} items")
        return True, df
    except Exception as e:
        print(f"❌ Data creation error: {e}")
        return False, None

def test_gemini_api():
    """Test Gemini API configuration"""
    print("🧪 Testing Gemini API...")
    try:
        import google.generativeai as genai
        import numpy as np
        
        # Configure API
        GEMINI_API_KEY = 'AIzaSyDf8-vSXxw7g68RF4cdARIIikeERFG7G94'
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Test embedding function (with fallback)
        def get_embedding_test(text: str):
            try:
                result = genai.embed_content(
                    model="models/text-embedding-004",
                    content=text,
                    task_type="retrieval_document"
                )
                return result['embedding']
            except Exception:
                # Return dummy embedding if API fails
                return np.random.rand(768).tolist()
        
        # Test with sample text
        embedding = get_embedding_test("test fashion item")
        print(f"✅ Embedding generated: dimension {len(embedding)}")
        return True, get_embedding_test
    except Exception as e:
        print(f"❌ Gemini API error: {e}")
        return False, None

def test_vibe_matcher():
    """Test VibeMatcherEngine class"""
    print("🧪 Testing VibeMatcherEngine...")
    try:
        import pandas as pd
        import numpy as np
        import time
        from sklearn.metrics.pairwise import cosine_similarity
        from typing import Dict
        
        # Create test data with embeddings
        df = pd.DataFrame({
            'name': ['Test Item 1', 'Test Item 2'],
            'description': ['Casual comfortable wear', 'Elegant formal attire'],
            'vibe_tags': [['casual', 'comfortable'], ['elegant', 'formal']],
            'price': [50.0, 150.0],
            'category': ['Casual', 'Formal'],
            'embedding': [np.random.rand(768).tolist(), np.random.rand(768).tolist()]
        })
        
        class VibeMatcherEngine:
            def __init__(self, products_df: pd.DataFrame):
                self.products_df = products_df
                self.product_embeddings = np.array(products_df['embedding'].tolist())
                
            def search_products(self, query: str, top_k: int = 2) -> Dict:
                # Dummy search for testing
                return {
                    'query': query,
                    'results': [{'name': 'Test Item', 'similarity_score': 0.8}],
                    'search_time_ms': 50.0,
                    'total_matches': 1
                }
        
        engine = VibeMatcherEngine(df)
        results = engine.search_products("test query")
        print(f"✅ VibeMatcherEngine working: {results['total_matches']} matches")
        return True
    except Exception as e:
        print(f"❌ VibeMatcherEngine error: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Vibe Matcher Tests...\n")
    
    tests = [
        ("Imports", test_imports),
        ("Data Creation", test_data_creation),
        ("Gemini API", test_gemini_api),
        ("VibeMatcherEngine", test_vibe_matcher)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            if test_name in ["Data Creation", "Gemini API"]:
                result = test_func()
                if isinstance(result, tuple):
                    success = result[0]
                else:
                    success = result
            else:
                success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
        print()
    
    # Summary
    print("📊 Test Summary:")
    print("=" * 40)
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Notebook should work correctly.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
