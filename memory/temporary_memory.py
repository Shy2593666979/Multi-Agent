# -*- coding: utf-8 -*-
"""
Memory module for conversation
"""

import json
import os
from typing import Iterable, Sequence
from typing import Optional
from typing import Union
from typing import Callable

from loguru import logger

from memory import MemoryBase

class TemporaryMemory(MemoryBase):
    """
    In-memory memory module, not writing to hard disk
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        embedding_model: Union[str, Callable] = None,
    ) -> None:
        super().__init__(config)

        self._content = []

        # prepare embedding model if needed
        
        self.embedding_model = embedding_model

    def add(
        self,
        memories: Union[Sequence[dict], dict, None],
        embed: bool = False,
    ) -> None:
        if memories is None:
            return

        if not isinstance(memories, list):
            record_memories = [memories]
        else:
            record_memories = memories

        # if memory doesn't have id attribute, we skip the checking
        memories_idx = set(_.id for _ in self._content if hasattr(_, "id"))
        for memory_unit in record_memories:
            # add to memory if it's new
            if (
                not hasattr(memory_unit, "id")
                or memory_unit.id not in memories_idx
            ):
                if embed:
                    if self.embedding_model:
                        # TODO: embed only content or its string representation
                        memory_unit.embedding = self.embedding_model(
                            [memory_unit],
                            return_embedding_only=True,
                        )
                    else:
                        raise RuntimeError("Embedding model is not provided.")
                self._content.append(memory_unit)

    def delete(self, index: Union[Iterable, int]) -> None:
        if self.size() == 0:
            logger.warning(
                "The memory is empty, and the delete operation is "
                "skipping.",
            )
            return

        if isinstance(index, int):
            index = [index]

        if isinstance(index, list):
            index = set(index)

            invalid_index = [_ for _ in index if _ >= self.size() or _ < 0]
            if len(invalid_index) > 0:
                logger.warning(
                    f"Skip delete operation for the invalid "
                    f"index {invalid_index}",
                )

            self._content = [
                _ for i, _ in enumerate(self._content) if i not in index
            ]
        else:
            raise NotImplementedError(
                "index type only supports {None, int, list}",
            )

    def export(
        self,
        to_mem: bool = False,
        file_path: Optional[str] = None,
    ) -> Optional[list]:
        """Export memory to json file"""
        if to_mem:
            return self._content

        if to_mem is False and file_path is not None:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(self._content, f, indent=4)
        else:
            raise NotImplementedError(
                "file type only supports "
                "{json, yaml, pkl}, default is json",
            )
        return None

    def load(
        self,
        memories: Union[str, dict, list],
        overwrite: bool = False,
    ) -> None:
        if isinstance(memories, str):
            if os.path.isfile(memories):
                with open(memories, "r", encoding="utf-8") as f:
                    self.add(json.load(f))
            else:
                try:
                    load_memories = json.loads(memories)
                    if not isinstance(load_memories, dict) and not isinstance(
                        load_memories,
                        list,
                    ):
                        logger.warning(
                            "The memory loaded by json.loads is "
                            "neither a dict nor a list, which may "
                            "cause unpredictable errors.",
                        )
                except json.JSONDecodeError as e:
                    raise json.JSONDecodeError(
                        f"Cannot load [{memories}] via " f"json.loads.",
                        e.doc,
                        e.pos,
                    )
        else:
            load_memories = memories

        # overwrite the original memories after loading the new ones
        if overwrite:
            self.clear()

        self.add(load_memories)

    def clear(self) -> None:
        """Clean memory, depending on how the memory are stored"""
        self._content = []

    def size(self) -> int:
        """Returns the number of memory segments in memory."""
        return len(self._content)

    

    def get_memory(
        self,
        recent_n: Optional[int] = None,
        filter_func: Optional[Callable[[int, dict], bool]] = None,
    ) -> list:
        """Retrieve memory.

        Args:
            recent_n (`Optional[int]`, default `None`):
                The last number of memories to return.
            filter_func
                (`Callable[[int, dict], bool]`, default to `None`):
                The function to filter memories, which take the index and
                memory unit as input, and return a boolean value.
        """
        # extract the recent `recent_n` entries in memories
        if recent_n is None:
            memories = self._content
        else:
            if recent_n > self.size():
                logger.warning(
                    "The retrieved number of memories {} is "
                    "greater than the total number of memories {"
                    "}",
                    recent_n,
                    self.size(),
                )
            memories = self._content[-recent_n:]

        # filter the memories
        if filter_func is not None:
            memories = [_ for i, _ in enumerate(memories) if filter_func(i, _)]

        return memories
