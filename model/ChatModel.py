import requests
import json

def send_message(msg):
    """
    请求LLM函数
    """
    

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-ChtJNYJD1sm5FqwA7bE8EfFa3eE847Fa9758E5626d64Cc9a"
    }
    data = {
        "model": "Qwen1.5-72b-chat",
        "messages": msg
    }

    response = requests.post("http://70.182.56.16:11000/v1/chat/completions", data=json.dumps(data),
                             headers=headers).content
    print(response)

    response_1 = json.loads(response)
    message = response_1["choices"][0]["message"]["content"]
    
    return str(message)
