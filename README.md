
## Multi-Agent 例： ⭐⭐⭐
通过定义两个玩五子棋Agent的高手来互相对弈，通过 memory 和 LLM 实现 Think 达成 Action
### Models
- Local model
- 通义千问 API
- ChatGPT API

截取游戏中一张图，使用matplotlib画图。如下图：


![image](https://github.com/Shy2593666979/Multi-Agent/assets/105286202/1f874832-a36e-4932-979a-c7fa083be6af)



💡
### 一个与人工智能进行对弈的Agent

将`game_gomoku.py`文件中修改如下：
```py
# -*- coding: utf-8 -*-
"""The main script to start a Gomoku game between two agents and a board
agent."""

from board_agent import NAME_TO_PIECE, NAME_BLACK, NAME_WHITE, BoardAgent
from gomoku_agent import GomokuAgent
from user_agent import UserAgent

from utils.msghup import msghub


MAX_STEPS = 20

SYS_PROMPT_TEMPLATE = """
You're a skillful Gomoku player. You should play against your opponent according to the following rules:

Game Rules:
1. This Gomoku board is a 15*15 grid. Moves are made by specifying row and column indexes, with [0, 0] marking the top-left corner and [14, 14] indicating the bottom-right corner.
2. The goal is to be the first player to form an unbroken line of your pieces horizontally, vertically, or diagonally.
3. If the board is completely filled with pieces and no player has formed a row of five, the game is declared a draw.

Note:
1. Your pieces are represented by '{}', your opponent's by '{}'. 0 represents an empty spot on the board.
2. You should think carefully about your strategy and moves, considering both your and your opponent's subsequent moves.
3. Make sure you don't place your piece on a spot that has already been occupied.
4. Only an unbroken line of five same pieces will win the game. For example, "xxxoxx" won't be considered a win.
5. Note the unbroken line can be formed in any direction: horizontal, vertical, or diagonal.
"""  # noqa



piece_black = NAME_TO_PIECE[NAME_BLACK]
piece_white = NAME_TO_PIECE[NAME_WHITE]

black = GomokuAgent(
    NAME_BLACK,
    sys_prompt=SYS_PROMPT_TEMPLATE.format(piece_black, piece_white),
)

white = UserAgent(
    NAME_WHITE,
    sys_prompt=SYS_PROMPT_TEMPLATE.format(piece_white, piece_black),
)

board = BoardAgent(name="Host")

# Start the game!
msg = None
i = 0
# Use a msg hub to share conversation between two players, e.g. white player
# can hear what black player says to the board
with msghub(participants=[black, white, board]):
    while not board.game_end and i < MAX_STEPS:
        for player in [black, white]:
            # receive the move from the player, judge if the game ends and
            # remind the player to make a move
            msg = board(msg)

            # end the game if draw or win
            if board.game_end:
                break

            # make a move
            msg = player(msg)

            i += 1

```

这些模型综合能力比较强，但是下棋能力没有进行单独训练，下棋的能力很一般，人类一般都可以获胜Win！ 🤗

### 参考
通义千问多Agent示例
