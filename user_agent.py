import json
from utils.message import Msg
from agents.agent import AgentBase
from typing import Optional

user_prompt = "You are a gamer and have your own decision-making power"

class UserAgent(AgentBase):
    def __init__(self,
                 name : str,
                 sys_prompt: str,
                 model_config_name: str = None,
                 ) -> None:
        
        super().__init__(
            name=name,
            sys_prompt=sys_prompt,
            model_config_name=model_config_name,
        )
        self.memory.add(Msg("system", sys_prompt, role="system"))
    
    def reply(self, x: Optional[dict] = None) -> dict:
        if self.memory:
            self.memory.add(x)

        input_str = input("Please enter the X and Y coordinates of the game you want to play : ")

        input_X, input_Y = map(int, input_str.split())

        response =  [input_Y,input_X]

        self.speak(
            Msg(
                self.name,
                json.dumps(response, indent=4, ensure_ascii=False),
                role="user",
            ),
        )

        if self.memory:
            self.memory.add(Msg(self.name, response, role="user"))

        # Hide thought from the response
        return Msg(self.name, response, role="user")
