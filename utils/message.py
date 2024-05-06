
from typing import Any, Optional, Union, Sequence, Literal
from uuid import uuid4
import json

from loguru import logger

from utils.help import _get_timestamp


class MessageBase(dict):
    """Base Message class, which is used to maintain information for dialog,
    memory and used to construct prompt.
    """

    def __init__(
        self,
        name: str,
        content: Any,
        role: Literal["user", "system", "assistant"] = "assistant",
        url: Optional[Union[Sequence[str], str]] = None,
        timestamp: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the message object

        Args:
            name (`str`):
                The name of who send the message. It's often used in
                role-playing scenario to tell the name of the sender.
            content (`Any`):
                The content of the message.
            role (`Literal["system", "user", "assistant"]`, defaults to "assistant"):
                The role of who send the message. It can be one of the
                `"system"`, `"user"`, or `"assistant"`. Default to
                `"assistant"`.
            url (`Optional[Union[list[str], str]]`, defaults to None):
                A url to file, image, video, audio or website.
            timestamp (`Optional[str]`, defaults to None):
                The timestamp of the message, if None, it will be set to
                current time.
            **kwargs (`Any`):
                Other attributes of the message.
        """  # noqa
        # id and timestamp will be added to the object as its attributes
        # rather than items in dict
        self.id = uuid4().hex
        if timestamp is None:
            self.timestamp = _get_timestamp()
        else:
            self.timestamp = timestamp

        self.name = name
        self.content = content
        self.role = role

        if url:
            self.url = url
        else:
            self.url = None

        self.update(kwargs)

    def __getattr__(self, key: Any) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(f"no attribute '{key}'") from e

    def __setattr__(self, key: Any, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: Any) -> None:
        try:
            del self[key]
        except KeyError as e:
            raise AttributeError(f"no attribute '{key}'") from e

    def to_str(self) -> str:
        """Return the string representation of the message"""
        raise NotImplementedError

    def serialize(self) -> str:
        """Return the serialized message."""
        raise NotImplementedError


class Msg(MessageBase):
    """The Message class."""

    def __init__(
        self,
        name: str,
        content: Any,
        role: Literal["system", "user", "assistant"] = None,
        url: Optional[Union[Sequence[str], str]] = None,
        timestamp: Optional[str] = None,
        echo: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the message object

        Args:
            name (`str`):
                The name of who send the message.
            content (`Any`):
                The content of the message.
            role (`Literal["system", "user", "assistant"]`):
                Used to identify the source of the message, e.g. the system
                information, the user input, or the model response. This
                argument is used to accommodate most Chat API formats.
            url (`Optional[Union[list[str], str]]`, defaults to `None`):
                A url to file, image, video, audio or website.
            timestamp (`Optional[str]`, defaults to `None`):
                The timestamp of the message, if None, it will be set to
                current time.
            **kwargs (`Any`):
                Other attributes of the message.
        """

        if role is None:
            logger.warning(
                "A new field `role` is newly added to the message. "
                "Please specify the role of the message. Currently we use "
                'a default "assistant" value.',
            )

        super().__init__(
            name=name,
            content=content,
            role=role or "assistant",
            url=url,
            timestamp=timestamp,
            **kwargs,
        )
        if echo:
            logger.chat(self)

    def to_str(self) -> str:
        """Return the string representation of the message"""
        return f"{self.name}: {self.content}"

    def serialize(self) -> str:
        return json.dumps({"__type": "Msg", **self})


class Tht(MessageBase):
    """The Thought message is used to record the thought of the agent to
    help them make decisions and responses. Generally, it shouldn't be
    passed to or seen by the other agents.

    In our framework, we formulate the thought in prompt as follows:
    - For OpenAI API calling:

    .. code-block:: python

        [
            ...
            {
                "role": "assistant",
                "name": "thought",
                "content": "I should ..."
            },
            ...
        ]

    - For open-source models that accepts string as input:

    .. code-block:: python

        ...
        {self.name} thought: I should ...
        ...

    We admit that there maybe better ways to formulate the thought. Users
    are encouraged to create their own thought formulation methods by
    inheriting `MessageBase` class and rewrite the `__init__` and `to_str`
    function.

    .. code-block:: python

        class MyThought(MessageBase):
            def to_str(self) -> str:
                # implement your own thought formulation method
                pass
    """

    def __init__(
        self,
        content: Any,
        timestamp: Optional[str] = None,
    ) -> None:
        super().__init__(
            name="thought",
            content=content,
            role="assistant",
            timestamp=timestamp,
        )

    def to_str(self) -> str:
        """Return the string representation of the message"""
        return f"{self.name} thought: {self.content}"

    def serialize(self) -> str:
        return json.dumps({"__type": "Tht", **self})
