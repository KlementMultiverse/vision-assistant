"""Vision module for VLM inference."""

import requests


class VisionModel:
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.endpoint = f"{base_url}/v1/chat/completions"

    def describe(self, image_base64: str, prompt: str = None) -> str:
        """Send image to VLM and get description."""
        if prompt is None:
            prompt = "Briefly describe what you see. Focus on people and their actions. Be concise (1-2 sentences)."

        payload = {
            "model": "smolvlm",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                    {"type": "text", "text": prompt}
                ]
            }],
            "max_tokens": 100,
            "temperature": 0.3
        }

        try:
            response = requests.post(self.endpoint, json=payload, timeout=30)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            return f"Error: {response.status_code}"
        except Exception as e:
            return f"Error: {e}"

    def analyze(self, image_base64: str) -> dict:
        """Analyze image and return structured info."""
        prompt = "Describe what you see briefly. Is there a person? What are they doing?"

        response = self.describe(image_base64, prompt)
        response_lower = response.lower()

        # Parse response using keywords
        result = {
            "person": False,
            "action": "unknown",
            "looking": False,
            "raw": response
        }

        # Detect person
        person_words = ["person", "man", "woman", "individual", "someone", "human", "face", "people"]
        result["person"] = any(word in response_lower for word in person_words)

        # Detect looking at camera
        looking_words = ["looking at camera", "facing camera", "looking straight", "looking ahead", "facing forward"]
        result["looking"] = any(phrase in response_lower for phrase in looking_words)

        # Detect actions
        if "standing" in response_lower:
            result["action"] = "standing"
        elif "sitting" in response_lower:
            result["action"] = "sitting"
        elif "waving" in response_lower or "wave" in response_lower:
            result["action"] = "waving"
        elif "walking" in response_lower:
            result["action"] = "walking"

        return result
