"""Fetch more diverse data sources to improve detection"""
import json
from datasets import load_dataset
from tqdm import tqdm

def fetch_essays():
    """Academic essays - formal human writing"""
    print("Fetching academic essays...")
    try:
        ds = load_dataset("qwedsacf/ivypanda-essays", split="train", streaming=True)
        samples = []
        for i, item in enumerate(ds):
            if i >= 5000: break
            text = item.get('TEXT', item.get('text', ''))
            if len(text) > 200:
                samples.append({"text": text[:2000], "label": "human", "source": "Essays"})
        print(f"  Got {len(samples)} essays")
        return samples
    except Exception as e:
        print(f"  Essays failed: {e}")
        return []

def fetch_stories():
    """Creative fiction stories"""
    print("Fetching creative stories...")
    try:
        ds = load_dataset("lksy/ru_story_generation", split="train", streaming=True)
        samples = []
        for i, item in enumerate(ds):
            if i >= 3000: break
            text = item.get('story', item.get('text', ''))
            if len(text) > 200:
                samples.append({"text": text[:2000], "label": "human", "source": "Stories"})
        print(f"  Got {len(samples)} stories")
        return samples
    except:
        # Fallback to ROCStories
        try:
            ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
            samples = []
            for i, item in enumerate(ds):
                if i >= 5000: break
                text = item.get('text', '')
                if len(text) > 100:
                    samples.append({"text": text[:2000], "label": "human", "source": "TinyStories"})
            print(f"  Got {len(samples)} tiny stories")
            return samples
        except Exception as e:
            print(f"  Stories failed: {e}")
            return []

def fetch_blogs():
    """Personal blog posts - casual human writing"""
    print("Fetching blog posts...")
    try:
        ds = load_dataset("blog_authorship_corpus", split="train", streaming=True)
        samples = []
        for i, item in enumerate(ds):
            if i >= 5000: break
            text = item.get('text', '')
            if len(text) > 200:
                samples.append({"text": text[:2000], "label": "human", "source": "Blogs"})
        print(f"  Got {len(samples)} blog posts")
        return samples
    except Exception as e:
        print(f"  Blogs failed: {e}")
        return []

def fetch_emails():
    """Email communications - natural human writing"""
    print("Fetching emails (Enron)...")
    try:
        ds = load_dataset("Yale-LILY/ENRON", split="train", streaming=True)
        samples = []
        for i, item in enumerate(ds):
            if i >= 5000: break
            text = item.get('body', item.get('text', ''))
            if len(text) > 100:
                samples.append({"text": text[:2000], "label": "human", "source": "Emails"})
        print(f"  Got {len(samples)} emails")
        return samples
    except Exception as e:
        print(f"  Emails failed: {e}")
        return []

def fetch_chatgpt_samples():
    """ChatGPT generated content"""
    print("Fetching ChatGPT samples...")
    try:
        ds = load_dataset("Hello-SimpleAI/HC3", "all", split="train", streaming=True)
        samples = []
        for i, item in enumerate(ds):
            if i >= 10000: break
            # HC3 has chatgpt_answers
            answers = item.get('chatgpt_answers', [])
            for ans in answers[:1]:
                if len(ans) > 100:
                    samples.append({"text": ans[:2000], "label": "ai", "source": "ChatGPT-HC3"})
        print(f"  Got {len(samples)} ChatGPT samples")
        return samples
    except Exception as e:
        print(f"  ChatGPT failed: {e}")
        return []

def fetch_llama_samples():
    """LLaMA/Vicuna generated content"""
    print("Fetching Vicuna samples...")
    try:
        ds = load_dataset("lmsys/chatbot_arena_conversations", split="train", streaming=True)
        samples = []
        for i, item in enumerate(ds):
            if i >= 5000: break
            convs = item.get('conversation_a', []) + item.get('conversation_b', [])
            for msg in convs:
                if msg.get('role') == 'assistant':
                    text = msg.get('content', '')
                    if len(text) > 100:
                        samples.append({"text": text[:2000], "label": "ai", "source": "ChatbotArena"})
        print(f"  Got {len(samples)} chatbot arena samples")
        return samples
    except Exception as e:
        print(f"  Chatbot arena failed: {e}")
        return []

def fetch_gpt4_samples():
    """GPT-4 generated content"""
    print("Fetching GPT-4 samples...")
    try:
        ds = load_dataset("teknium/GPT4-LLM-Cleaned", split="train", streaming=True)
        samples = []
        for i, item in enumerate(ds):
            if i >= 5000: break
            text = item.get('output', item.get('response', ''))
            if len(text) > 100:
                samples.append({"text": text[:2000], "label": "ai", "source": "GPT4-LLM"})
        print(f"  Got {len(samples)} GPT-4 samples")
        return samples
    except Exception as e:
        print(f"  GPT-4 failed: {e}")
        return []

def fetch_claude_samples():
    """Claude generated content"""
    print("Fetching Claude-style samples...")
    try:
        ds = load_dataset("Dahoas/full-hh-rlhf", split="train", streaming=True)
        samples = []
        for i, item in enumerate(ds):
            if i >= 5000: break
            text = item.get('response', item.get('chosen', ''))
            if len(text) > 100:
                samples.append({"text": text[:2000], "label": "ai", "source": "HH-RLHF-Full"})
        print(f"  Got {len(samples)} RLHF samples")
        return samples
    except Exception as e:
        print(f"  Claude failed: {e}")
        return []

def main():
    all_samples = []
    
    # Human sources
    all_samples.extend(fetch_essays())
    all_samples.extend(fetch_stories())
    all_samples.extend(fetch_blogs())
    all_samples.extend(fetch_emails())
    
    # AI sources
    all_samples.extend(fetch_chatgpt_samples())
    all_samples.extend(fetch_llama_samples())
    all_samples.extend(fetch_gpt4_samples())
    all_samples.extend(fetch_claude_samples())
    
    print(f"\n=== TOTAL: {len(all_samples)} new samples ===")
    human = sum(1 for s in all_samples if s['label'] == 'human')
    ai = sum(1 for s in all_samples if s['label'] == 'ai')
    print(f"Human: {human}, AI: {ai}")
    
    # Save
    with open('diverse_samples.json', 'w') as f:
        json.dump(all_samples, f)
    print("Saved to diverse_samples.json")

if __name__ == "__main__":
    main()
