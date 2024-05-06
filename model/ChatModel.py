import requests
import json
import requests
import dashscope
import openai

def send_message(msg):
    """
    请求LLM函数
    """
    # 请求通义千问API
    # return send_tongyiqwen_message(msg)

    # 请求ChatGPT API
    return str(send_chatgpt_message(msg))


def send_local_qwen_message(message, user_input):
    """
    请求Qwen函数
    """
    
    headers = {
        # "Authorization": f"Bearer {config.API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "prompt": message,
        "history": [
            [
                "You are a helpful assistant.", ""
            ]
        ]
    }

    try:
        response = requests.post('Qwen_URL', headers=headers, json=data, verify=False)
        if response.status_code == 200:
            answer = response.json()['data']
            print('LLM输出:', answer)
            print('--------------------------------------------------------------------')
            return answer
        else:
            print(f"Error: {response.status_code}")
            return None
    except requests.RequestException as e:
        print(f"Request error: {e}")
        return None


def send_chatgpt_message(msg):
    """
    请求chatGPT函数
    """
    GPT_URL = "https://key.wenwen-ai.com/v1"
    API_KEY = "sk-6V2exWFBSa2lmuZ7C0D773D1BaEd4fB7A1B6A0A265D550C6"
    
    client = openai.OpenAI(base_url=GPT_URL,api_key=API_KEY)

    completion = client.chat.completions.create(
        model="gpt-4",
        messages = msg
    )

    return str(completion.choices[0].message.content)


def send_tongyiqwen_message(msg):
    """
    请求通义千问api函数
    """
    
    dashscope.api_key = "sk-013111ab4d7d4fb2919d6046bcxxxxxx"
    try:
        response = dashscope.Generation.call(
            model=dashscope.Generation.Models.qwen_plus, messages=msg
        )
        if response.status_code == 200:
            answer = response.output["text"]
            print("LLM输出:", answer)
            print(
                "--------------------------------------------------------------------"
            )
            return answer
        else:
            print(f"Error: {response.status_code}")
            return None
    except requests.RequestException as e:
        print(f"Request error: {e}")
        return None
