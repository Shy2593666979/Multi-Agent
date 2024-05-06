"""A Gomoku agent that can play the game with another agent."""

from typing import List, Optional, Sequence, Union

import json

from model.ChatModel import send_message
from utils.help import _convert_to_str
from utils.message import Msg
from agents.agent import AgentBase
from utils.message import MessageBase,Msg

HINT_PROMPT = """
You should respond in the following format, which can be loaded by json.loads in Python:
{{
    "thought": "analyze the present situation, and what move you should make",
    "move": [row index, column index]
}}
"""  # noqa



class GomokuAgent(AgentBase):
    """A Gomoku agent that can play the game with another agent."""

    def __init__(
        self,
        name: str,
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

        msg_hint = Msg("system", HINT_PROMPT, role="system")

        prompt = self.format(
            self.memory.get_memory(),
            msg_hint,
        )

        response = send_message(prompt)

        # For better presentation, we print the response proceeded by
        # json.dumps, this msg won't be recorded in memory
        self.speak(
            Msg(
                self.name,
                json.dumps(response, indent=4, ensure_ascii=False),
                role="assistant",
            ),
        )

        if self.memory:
            self.memory.add(Msg(self.name, response, role="assistant"))

        # Hide thought from the response
        return Msg(self.name, response, role="assistant")
    
    def format(
        self,
        *args: Union[MessageBase, Sequence[MessageBase]],
    ) -> List:
        """Format the messages for DashScope Chat API.

        In this format function, the input messages are formatted into a
        single system messages with format "{name}: {content}" for each
        message. Note this strategy maybe not suitable for all scenarios,
        and developers are encouraged to implement their own prompt
        engineering strategies.

        The following is an example:

        .. code-block:: python

            prompt = model.format(
                Msg("system", "You're a helpful assistant", role="system"),
                Msg("Bob", "Hi, how can I help you?", role="assistant"),
                Msg("user", "What's the date today?", role="user")
            )

        The prompt will be as follows:

        .. code-block:: python

            [
                {
                    "role": "system",
                    "content": "You're a helpful assistant",
                }
                {
                    "role": "user",
                    "content": (
                        "## Dialogue History\\n"
                        "Bob: Hi, how can I help you?\\n"
                        "user: What's the date today?"
                    )
                }
            ]


        Args:
            args (`Union[MessageBase, Sequence[MessageBase]]`):
                The input arguments to be formatted, where each argument
                should be a `Msg` object, or a list of `Msg` objects.
                In distribution, placeholder is also allowed.

        Returns:
            `List[dict]`:
                The formatted messages.
        """

        # Parse all information into a list of messages
        input_msgs = []
        for _ in args:
            if _ is None:
                continue
            if isinstance(_, MessageBase):
                input_msgs.append(_)
            elif isinstance(_, list) and all(
                isinstance(__, MessageBase) for __ in _
            ):
                input_msgs.extend(_)
            else:
                raise TypeError(
                    f"The input should be a Msg object or a list "
                    f"of Msg objects, got {type(_)}.",
                )

        messages = []

        # record dialog history as a list of strings
        dialogue = []
        for i, unit in enumerate(input_msgs):
            if i == 0 and unit.role == "system":
                # system prompt
                messages.append(
                    {
                        "role": unit.role,
                        "content": _convert_to_str(unit.content),
                    },
                )
            else:
                # Merge all messages into a dialogue history prompt
                dialogue.append(
                    f"{unit.name}: {_convert_to_str(unit.content)}",
                )

        dialogue_history = "\n".join(dialogue)

        user_content_template = "## Dialogue History\n{dialogue_history}"

        messages.append(
            {
                "role": "user",
                "content": user_content_template.format(
                    dialogue_history=dialogue_history,
                ),
            },
        )

        return messages
