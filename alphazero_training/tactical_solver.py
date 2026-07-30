"""Exact short tactics plus bounded VCF/VCT solvers for freestyle Gomoku.

The module is deliberately independent from the trainer's candidate filters,
pattern heuristic, neural network, and MCTS.  Boards use the training code's
convention: black is ``1``, white is ``-1``, and empty is ``0``.  Actions are
row-major integers (``action = y * size + x``).

``PROVEN_NO_VCF`` means that no continuous-four win exists *within the supplied
ply bound*.  It is not a claim about arbitrary VCT play.  Resource exhaustion
is reported separately as ``UNKNOWN_BUDGET`` and is never treated as a loss.
Freestyle rules are used throughout, so a line of five or more wins.

The VCT solver is deliberately conservative.  Its attacker may play only a
move that wins now or creates an exact immediate/three-ply threat.  Every
defensive reply that averts that threat is enumerated.  Consequently a
``PROVEN_WIN`` is a sound forced win, while ``PROVEN_NO_VCT`` only excludes a
win inside this bounded threat-space definition.  Candidate, defense, node,
or wall-clock exhaustion returns ``UNKNOWN_BUDGET`` instead of a false loss.

Architecture / 代码架构
-----------------------
``FreestyleBoard`` is an immutable rule-only board. Small helper functions
handle immediate and three-ply tactics. Separate bounded search engines prove
VCF and conservative VCT sequences. ``TacticalSolver`` is the public facade
used by training, evaluation, and desktop search; it has no neural-network or
Pygame dependency.

``FreestyleBoard`` 是不可变的纯规则棋盘；小型函数处理一步与三手战术；两个独立的
有限搜索器分别证明 VCF 和保守 VCT。``TacticalSolver`` 是训练、评估和桌面搜索共用
的公开接口，不依赖神经网络或 Pygame。

Key algorithms / 重要算法
-------------------------
Immediate-win enumeration tests every legal winning point. Three-ply search
checks attacker-defender-attacker forcing lines. VCF proves wins made from
continuous fours; VCT explores a broader but conservative threat space. Every
query has ply, node, and optional time limits. Only ``PROVEN_WIN`` is a proof;
budget exhaustion returns ``UNKNOWN_BUDGET`` so uncertainty is not mislabeled.

一步胜点枚举会检查所有合法获胜位置；三手搜索检查“进攻-防守-进攻”的强制路线；
VCF 证明连续冲四胜，VCT 搜索更广但仍保守的威胁空间。每次查询都有手数、节点数
和可选时间限制。只有 ``PROVEN_WIN`` 才是胜利证明；预算耗尽返回
``UNKNOWN_BUDGET``，避免把“没算完”误判为输棋。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import time
from typing import Iterable, Sequence


BLACK = 1
WHITE = -1
EMPTY = 0
DEFAULT_BOARD_SIZE = 19
DEFAULT_WIN_LENGTH = 5
DIRECTIONS = ((1, 0), (0, 1), (1, 1), (1, -1))


BoardLike = Sequence[int] | Sequence[Sequence[int]]


class SolveStatus(str, Enum):
    """Proof status returned by the bounded tactical solvers."""

    PROVEN_WIN = "PROVEN_WIN"
    PROVEN_NO_VCF = "PROVEN_NO_VCF"
    PROVEN_NO_VCT = "PROVEN_NO_VCT"
    UNKNOWN_BUDGET = "UNKNOWN_BUDGET"


@dataclass(frozen=True)
class SolveLimits:
    """Hard limits for one VCF query.

    ``max_plies`` counts actual moves from the root, including the winning
    placement.  ``time_ms=0`` disables the wall-clock limit; node and ply
    limits remain active.
    """

    max_plies: int = 9
    max_nodes: int = 50_000
    time_ms: float = 1_000.0

    def __post_init__(self) -> None:
        if self.max_plies < 0:
            raise ValueError("max_plies must be non-negative")
        if self.max_nodes <= 0:
            raise ValueError("max_nodes must be positive")
        if self.time_ms < 0:
            raise ValueError("time_ms must be non-negative")


@dataclass(frozen=True)
class SolveResult:
    """Result of a bounded VCF proof.

    ``winning_actions`` contains root moves individually proven to win.  If a
    budget expires after one proof has already been found, the status remains
    ``PROVEN_WIN`` but this tuple can be a proven subset rather than the full
    set.  ``required_defenses`` lists the immediate threat points created by
    the first move of ``principal_variation``.  Two or more such points mean a
    single defensive placement cannot cover them all.
    """

    status: SolveStatus
    winning_actions: tuple[int, ...]
    required_defenses: tuple[int, ...]
    principal_variation: tuple[int, ...]
    nodes: int


@dataclass(frozen=True)
class ThreatSolveLimits:
    """Hard bounds for one conservative VCT/threat-space query.

    ``max_attack_candidates`` may truncate an attack node, and
    ``max_defenses`` may truncate an exhaustive defender node.  Either cutoff
    makes an otherwise negative result ``UNKNOWN_BUDGET``.  Both limits are
    therefore performance controls, never assumptions that omitted moves lose.
    ``time_ms=0`` disables only the wall-clock check.
    """

    max_plies: int = 9
    max_nodes: int = 20_000
    time_ms: float = 100.0
    max_attack_candidates: int = 24
    max_defenses: int = 64

    def __post_init__(self) -> None:
        if self.max_plies < 0:
            raise ValueError("max_plies must be non-negative")
        if self.max_nodes <= 0:
            raise ValueError("max_nodes must be positive")
        if self.time_ms < 0:
            raise ValueError("time_ms must be non-negative")
        if self.max_attack_candidates <= 0:
            raise ValueError("max_attack_candidates must be positive")
        if self.max_defenses <= 0:
            raise ValueError("max_defenses must be positive")


@dataclass(frozen=True)
class ThreatSolveResult:
    """Result of a bounded conservative VCT proof.

    ``winning_actions`` is always a proven subset.  ``elapsed_ms`` and cache
    counters make the CPU cost observable without affecting proof semantics.
    """

    status: SolveStatus
    winning_actions: tuple[int, ...]
    principal_variation: tuple[int, ...]
    nodes: int
    cache_hits: int
    elapsed_ms: float


@dataclass(frozen=True)
class FreestyleBoard:
    """Small immutable board representation used by the tactical oracle."""

    cells: tuple[int, ...]
    size: int = DEFAULT_BOARD_SIZE
    win_length: int = DEFAULT_WIN_LENGTH

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError("size must be positive")
        if self.win_length <= 1:
            raise ValueError("win_length must be at least two")
        if len(self.cells) != self.size * self.size:
            raise ValueError(
                f"expected {self.size * self.size} cells, got {len(self.cells)}"
            )
        invalid = set(self.cells).difference((EMPTY, BLACK, WHITE))
        if invalid:
            raise ValueError(f"invalid board values: {sorted(invalid)}")

    @classmethod
    def from_board(
        cls,
        board: "FreestyleBoard | BoardLike",
        *,
        size: int = DEFAULT_BOARD_SIZE,
        win_length: int = DEFAULT_WIN_LENGTH,
    ) -> "FreestyleBoard":
        if isinstance(board, cls):
            if board.size != size or board.win_length != win_length:
                raise ValueError("board dimensions disagree with requested rules")
            return board

        # NumPy arrays and similar objects expose ``tolist``.  Using it keeps
        # this module dependency-free while accepting the trainer's board.
        if hasattr(board, "tolist"):
            board = board.tolist()  # type: ignore[assignment,union-attr]

        values = list(board)
        if len(values) == size and values and _is_row(values[0]):
            flat: list[int] = []
            for row in values:
                if not _is_row(row) or len(row) != size:  # type: ignore[arg-type]
                    raise ValueError(f"expected a rectangular {size}x{size} board")
                flat.extend(int(value) for value in row)  # type: ignore[arg-type]
            return cls(tuple(flat), size, win_length)

        if len(values) != size * size:
            raise ValueError(
                f"expected a flat board of {size * size} cells, got {len(values)}"
            )
        return cls(tuple(int(value) for value in values), size, win_length)

    @classmethod
    def from_stones(
        cls,
        stones: Iterable[tuple[int, int, int]],
        *,
        size: int = DEFAULT_BOARD_SIZE,
        win_length: int = DEFAULT_WIN_LENGTH,
    ) -> "FreestyleBoard":
        cells = [EMPTY] * (size * size)
        for x, y, player in stones:
            _validate_player(player)
            if not (0 <= x < size and 0 <= y < size):
                raise ValueError(f"point out of range: ({x}, {y})")
            action = y * size + x
            if cells[action] != EMPTY:
                raise ValueError(f"duplicate stone at ({x}, {y})")
            cells[action] = player
        return cls(tuple(cells), size, win_length)

    def legal_actions(self) -> tuple[int, ...]:
        return tuple(action for action, value in enumerate(self.cells) if value == EMPTY)

    def with_move(self, action: int, player: int) -> "FreestyleBoard":
        _validate_player(player)
        _validate_action(action, self.size)
        if self.cells[action] != EMPTY:
            raise ValueError(f"occupied action: {action}")
        cells = list(self.cells)
        cells[action] = player
        return FreestyleBoard(tuple(cells), self.size, self.win_length)

    def would_win(self, action: int, player: int) -> bool:
        """Return whether placing ``player`` at ``action`` makes >= five."""

        _validate_player(player)
        _validate_action(action, self.size)
        if self.cells[action] != EMPTY:
            return False
        y, x = divmod(action, self.size)
        for dx, dy in DIRECTIONS:
            length = 1
            for sign in (-1, 1):
                nx, ny = x + sign * dx, y + sign * dy
                while (
                    0 <= nx < self.size
                    and 0 <= ny < self.size
                    and self.cells[ny * self.size + nx] == player
                ):
                    length += 1
                    nx += sign * dx
                    ny += sign * dy
            if length >= self.win_length:
                return True
        return False

    def winning_actions(self, player: int) -> tuple[int, ...]:
        """Return every legal immediate win, with no heuristic filtering."""

        _validate_player(player)
        return tuple(
            action for action in self._line_relevant_actions(player) if self.would_win(action, player)
        )

    def has_five(self, player: int) -> bool:
        """Return whether ``player`` already has a freestyle winning line."""

        _validate_player(player)
        for action, stone in enumerate(self.cells):
            if stone != player:
                continue
            y, x = divmod(action, self.size)
            for dx, dy in DIRECTIONS:
                px, py = x - dx, y - dy
                if (
                    0 <= px < self.size
                    and 0 <= py < self.size
                    and self.cells[py * self.size + px] == player
                ):
                    continue
                length = 0
                nx, ny = x, y
                while (
                    0 <= nx < self.size
                    and 0 <= ny < self.size
                    and self.cells[ny * self.size + nx] == player
                ):
                    length += 1
                    nx += dx
                    ny += dy
                if length >= self.win_length:
                    return True
        return False

    def forcing_four_actions(self, attacker: int) -> tuple[int, ...]:
        """Return all non-winning moves that create an immediate win threat.

        This is an exact VCF candidate reduction: a continuous-four attack
        must make at least one win available on the next attacker turn, and a
        newly created threat necessarily shares a line with the placed stone.
        """

        _validate_player(attacker)
        forcing: list[int] = []
        for action in self._line_relevant_actions(attacker):
            if self.would_win(action, attacker):
                continue
            after = self.with_move(action, attacker)
            if after.winning_actions(attacker):
                forcing.append(action)
        return tuple(forcing)

    def _line_relevant_actions(self, player: int) -> tuple[int, ...]:
        """Exact superset for moves able to complete or create a five."""

        candidates: set[int] = set()
        reach = self.win_length - 1
        for action, stone in enumerate(self.cells):
            if stone != player:
                continue
            y, x = divmod(action, self.size)
            for dx, dy in DIRECTIONS:
                for step in range(-reach, reach + 1):
                    if step == 0:
                        continue
                    nx, ny = x + step * dx, y + step * dy
                    if 0 <= nx < self.size and 0 <= ny < self.size:
                        candidate = ny * self.size + nx
                        if self.cells[candidate] == EMPTY:
                            candidates.add(candidate)
        return tuple(sorted(candidates))


def _is_row(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _validate_player(player: int) -> None:
    if player not in (BLACK, WHITE):
        raise ValueError(f"player must be BLACK (1) or WHITE (-1), got {player!r}")


def _validate_action(action: int, size: int) -> None:
    if not isinstance(action, int) or isinstance(action, bool):
        raise TypeError("action must be an integer")
    if not 0 <= action < size * size:
        raise ValueError(f"action out of range: {action}")


def _position(
    board: FreestyleBoard | BoardLike,
    board_size: int,
    win_length: int,
) -> FreestyleBoard:
    return FreestyleBoard.from_board(board, size=board_size, win_length=win_length)


def immediate_winning_actions(
    board: FreestyleBoard | BoardLike,
    player: int,
    *,
    board_size: int = DEFAULT_BOARD_SIZE,
    win_length: int = DEFAULT_WIN_LENGTH,
) -> tuple[int, ...]:
    """Return every legal move that wins immediately."""

    return _position(board, board_size, win_length).winning_actions(player)


# A shorter compatibility alias is convenient at call sites.
winning_actions = immediate_winning_actions


def forced_win_in_three_actions(
    board: FreestyleBoard | BoardLike,
    attacker: int,
    *,
    board_size: int = DEFAULT_BOARD_SIZE,
    win_length: int = DEFAULT_WIN_LENGTH,
) -> tuple[int, ...]:
    """Return every non-winning attack that forces a win two plies later.

    The proof is exact: after the attack there must be at least two distinct
    immediate winning points, while the defender must not have an immediate
    win of its own.  A single defensive stone can occupy at most one point.
    """

    position = _position(board, board_size, win_length)
    _validate_player(attacker)
    defender = -attacker
    proven: list[int] = []
    for action in position.forcing_four_actions(attacker):
        after = position.with_move(action, attacker)
        if len(after.winning_actions(attacker)) < 2:
            continue
        if after.winning_actions(defender):
            continue
        proven.append(action)
    return tuple(proven)


def defenses_against_forced_win_in_three(
    board: FreestyleBoard | BoardLike,
    defender: int,
    *,
    board_size: int = DEFAULT_BOARD_SIZE,
    win_length: int = DEFAULT_WIN_LENGTH,
) -> tuple[int, ...]:
    """Return exactly the legal moves that avert all opponent <=3-ply wins.

    An immediate winning defense is always accepted.  Otherwise the resulting
    position must leave the opponent with neither an immediate win nor a
    forced-win-in-three attack.  Enumerating every legal defense makes this an
    exact oracle rather than a proximity or pattern heuristic.
    """

    position = _position(board, board_size, win_length)
    _validate_player(defender)
    attacker = -defender
    legal = position.legal_actions()
    immediate_attacks = position.winning_actions(attacker)
    immediate_defenses = set(position.winning_actions(defender))

    # If the opponent can win now, only an immediate counter-win or occupation
    # of every winning point is safe.  One stone cannot cover two points.
    if immediate_attacks:
        safe = set(immediate_defenses)
        if len(immediate_attacks) == 1:
            safe.add(immediate_attacks[0])
        return tuple(sorted(safe))

    attacks = forced_win_in_three_actions(
        position,
        attacker,
        board_size=board_size,
        win_length=win_length,
    )
    if not attacks:
        # Adding a defender stone cannot create a new line for the attacker.
        return legal

    # Cache each original proof as (attack point, its >=2 winning points).
    proofs = tuple(
        (attack, position.with_move(attack, attacker).winning_actions(attacker))
        for attack in attacks
    )
    safe: list[int] = []
    for defense in legal:
        if defense in immediate_defenses:
            safe.append(defense)
            continue

        after_defense = position.with_move(defense, defender)
        counter_threats = set(after_defense.winning_actions(defender))
        all_proofs_removed = True
        for attack, threat_points in proofs:
            if defense == attack:
                continue
            # The old proof survives unless the defensive stone occupies enough
            # of its threat points or creates a counter-win that the attack does
            # not itself cover.  Opponent stones cannot create a new proof, so
            # checking the original complete proof set is exact.
            remaining_threats = sum(point != defense for point in threat_points)
            defender_can_win = any(point != attack for point in counter_threats)
            if remaining_threats >= 2 and not defender_can_win:
                all_proofs_removed = False
                break
        if all_proofs_removed:
            safe.append(defense)
    return tuple(safe)


# Explicit aliases cover the vocabulary used by the evaluation code.
exact_defenses_against_win_in_three = defenses_against_forced_win_in_three
defenses_against_three_ply_win = defenses_against_forced_win_in_three


@dataclass(frozen=True)
class _SearchOutcome:
    status: SolveStatus
    pv: tuple[int, ...] = ()


class _BudgetExpired(RuntimeError):
    pass


class _VCFSearch:
    def __init__(self, attacker: int, limits: SolveLimits):
        _validate_player(attacker)
        self.attacker = attacker
        self.defender = -attacker
        self.limits = limits
        self.nodes = 0
        self.started = time.perf_counter()

    def _enter_node(self) -> None:
        if self.nodes >= self.limits.max_nodes:
            raise _BudgetExpired
        if self.limits.time_ms and (
            (time.perf_counter() - self.started) * 1_000.0 >= self.limits.time_ms
        ):
            raise _BudgetExpired
        self.nodes += 1

    def attack(self, board: FreestyleBoard, remaining: int) -> _SearchOutcome:
        self._enter_node()
        if board.has_five(self.attacker):
            return _SearchOutcome(SolveStatus.PROVEN_WIN)
        if board.has_five(self.defender) or remaining <= 0:
            return _SearchOutcome(SolveStatus.PROVEN_NO_VCF)

        immediate = board.winning_actions(self.attacker)
        if immediate:
            return _SearchOutcome(SolveStatus.PROVEN_WIN, (immediate[0],))

        saw_unknown = False
        for action in board.forcing_four_actions(self.attacker):
            after = board.with_move(action, self.attacker)
            try:
                outcome = self.defend(after, remaining - 1)
            except _BudgetExpired:
                saw_unknown = True
                break
            if outcome.status is SolveStatus.PROVEN_WIN:
                return _SearchOutcome(SolveStatus.PROVEN_WIN, (action,) + outcome.pv)
            if outcome.status is SolveStatus.UNKNOWN_BUDGET:
                saw_unknown = True

        if saw_unknown:
            return _SearchOutcome(SolveStatus.UNKNOWN_BUDGET)
        return _SearchOutcome(SolveStatus.PROVEN_NO_VCF)

    def defend(self, board: FreestyleBoard, remaining: int) -> _SearchOutcome:
        self._enter_node()
        if board.has_five(self.attacker):
            return _SearchOutcome(SolveStatus.PROVEN_WIN)
        if board.has_five(self.defender):
            return _SearchOutcome(SolveStatus.PROVEN_NO_VCF)

        # A defender win takes precedence over blocking a threat.
        defender_wins = board.winning_actions(self.defender)
        if defender_wins:
            return _SearchOutcome(SolveStatus.PROVEN_NO_VCF, (defender_wins[0],))

        threats = board.winning_actions(self.attacker)
        if not threats:
            return _SearchOutcome(SolveStatus.PROVEN_NO_VCF)

        if len(threats) >= 2:
            # The defender can cover only one point.  Show a concrete legal
            # block followed by a different win in the principal variation.
            if remaining < 2:
                return _SearchOutcome(SolveStatus.PROVEN_NO_VCF)
            defense = threats[0]
            after_defense = board.with_move(defense, self.defender)
            wins = after_defense.winning_actions(self.attacker)
            if not wins:  # Defensive assertion; cannot happen in placement Gomoku.
                return _SearchOutcome(SolveStatus.PROVEN_NO_VCF)
            return _SearchOutcome(SolveStatus.PROVEN_WIN, (defense, wins[0]))

        # One immediate threat has exactly one non-winning reply: occupy it.
        if remaining <= 0:
            return _SearchOutcome(SolveStatus.PROVEN_NO_VCF)
        defense = threats[0]
        after_defense = board.with_move(defense, self.defender)
        try:
            continuation = self.attack(after_defense, remaining - 1)
        except _BudgetExpired:
            return _SearchOutcome(SolveStatus.UNKNOWN_BUDGET)
        return _SearchOutcome(continuation.status, (defense,) + continuation.pv)


def solve_vcf(
    board: FreestyleBoard | BoardLike,
    attacker: int,
    limits: SolveLimits | None = None,
    *,
    board_size: int = DEFAULT_BOARD_SIZE,
    win_length: int = DEFAULT_WIN_LENGTH,
) -> SolveResult:
    """Prove a continuous-four win for ``attacker`` to move.

    Every attacker move considered by the DFS either wins immediately or
    creates an immediate four threat.  Defender replies are exact: an
    immediate counter-win, or the unique block when only one threat point
    exists.  Therefore a completed negative search proves the absence of VCF
    within ``limits.max_plies``; a resource cutoff returns ``UNKNOWN_BUDGET``.
    """

    _validate_player(attacker)
    position = _position(board, board_size, win_length)
    limits = limits or SolveLimits()
    if position.has_five(attacker) and position.has_five(-attacker):
        raise ValueError("invalid position: both players already have a winning line")
    if position.has_five(attacker):
        return SolveResult(SolveStatus.PROVEN_WIN, (), (), (), 1)
    if position.has_five(-attacker):
        return SolveResult(SolveStatus.PROVEN_NO_VCF, (), (), (), 1)

    search = _VCFSearch(attacker, limits)
    try:
        search._enter_node()  # Count the root once; children use search methods.
    except _BudgetExpired:
        return SolveResult(SolveStatus.UNKNOWN_BUDGET, (), (), (), search.nodes)

    if limits.max_plies <= 0:
        return SolveResult(SolveStatus.PROVEN_NO_VCF, (), (), (), search.nodes)

    immediate = position.winning_actions(attacker)
    if immediate:
        return SolveResult(
            SolveStatus.PROVEN_WIN,
            immediate,
            (),
            (immediate[0],),
            search.nodes,
        )

    proven: list[int] = []
    best_pv: tuple[int, ...] = ()
    best_defenses: tuple[int, ...] = ()
    saw_unknown = False

    for action in position.forcing_four_actions(attacker):
        after = position.with_move(action, attacker)
        try:
            outcome = search.defend(after, limits.max_plies - 1)
        except _BudgetExpired:
            saw_unknown = True
            break
        if outcome.status is SolveStatus.PROVEN_WIN:
            proven.append(action)
            candidate_pv = (action,) + outcome.pv
            if not best_pv or len(candidate_pv) < len(best_pv):
                best_pv = candidate_pv
                best_defenses = after.winning_actions(attacker)
        elif outcome.status is SolveStatus.UNKNOWN_BUDGET:
            saw_unknown = True

    if proven:
        status = SolveStatus.PROVEN_WIN
    elif saw_unknown:
        status = SolveStatus.UNKNOWN_BUDGET
    else:
        status = SolveStatus.PROVEN_NO_VCF
    return SolveResult(
        status,
        tuple(proven),
        tuple(best_defenses),
        best_pv,
        search.nodes,
    )


@dataclass(frozen=True)
class _ThreatOutcome:
    status: SolveStatus
    pv: tuple[int, ...] = ()


class _ThreatSearch:
    """Conservative AND/OR search over exact short-threat continuations."""

    def __init__(self, attacker: int, limits: ThreatSolveLimits):
        _validate_player(attacker)
        self.attacker = attacker
        self.defender = -attacker
        self.limits = limits
        self.nodes = 0
        self.cache_hits = 0
        self.started = time.perf_counter()
        # Include side to move and remaining ply depth in every key.  The
        # branch caps are included too, so future per-query reuse cannot blur
        # proof semantics between different budgets.
        self.cache: dict[
            tuple[tuple[int, ...], int, int, int, int, int], _ThreatOutcome
        ] = {}

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.started) * 1_000.0

    def _enter_node(self) -> None:
        if self.nodes >= self.limits.max_nodes:
            raise _BudgetExpired
        if self.limits.time_ms and self.elapsed_ms() >= self.limits.time_ms:
            raise _BudgetExpired
        self.nodes += 1

    def _key(
        self,
        board: FreestyleBoard,
        side_to_move: int,
        remaining: int,
    ) -> tuple[tuple[int, ...], int, int, int, int, int]:
        return (
            board.cells,
            side_to_move,
            remaining,
            self.limits.max_attack_candidates,
            self.limits.max_defenses,
            self.attacker,
        )

    def _cached(
        self,
        board: FreestyleBoard,
        side_to_move: int,
        remaining: int,
    ) -> _ThreatOutcome | None:
        cached = self.cache.get(self._key(board, side_to_move, remaining))
        if cached is not None:
            self.cache_hits += 1
        return cached

    def _remember(
        self,
        board: FreestyleBoard,
        side_to_move: int,
        remaining: int,
        outcome: _ThreatOutcome,
    ) -> _ThreatOutcome:
        # An UNKNOWN can depend on how much of the global node/time budget was
        # left when this transposition was reached, so never cache it.
        if outcome.status is not SolveStatus.UNKNOWN_BUDGET:
            self.cache[self._key(board, side_to_move, remaining)] = outcome
        return outcome

    def _forced_three_actions(self, board: FreestyleBoard) -> tuple[int, ...]:
        return forced_win_in_three_actions(
            board,
            self.attacker,
            board_size=board.size,
            win_length=board.win_length,
        )

    def _threat_candidates(
        self,
        board: FreestyleBoard,
    ) -> tuple[tuple[int, ...], bool]:
        """Return ordered forcing candidates and whether the cap omitted any."""

        def potential(action: int) -> tuple[int, int, int]:
            """Cheap deterministic ordering; never used as proof evidence."""

            y, x = divmod(action, board.size)
            window_score = 0
            strong_windows = 0
            for dx, dy in DIRECTIONS:
                best = 0
                for offset in range(-(board.win_length - 1), 1):
                    own = 0
                    valid = True
                    for step in range(board.win_length):
                        nx = x + (offset + step) * dx
                        ny = y + (offset + step) * dy
                        if not (0 <= nx < board.size and 0 <= ny < board.size):
                            valid = False
                            break
                        value = (
                            self.attacker
                            if nx == x and ny == y
                            else board.cells[ny * board.size + nx]
                        )
                        if value == self.defender:
                            valid = False
                            break
                        own += int(value == self.attacker)
                    if valid:
                        best = max(best, own)
                        strong_windows += int(own >= board.win_length - 2)
                window_score += best * best
            # Larger tactical potential first, then stable row-major order.
            return (-window_score, -strong_windows, action)

        raw = list(board._line_relevant_actions(self.attacker))
        raw.sort(key=potential)
        truncated = len(raw) > self.limits.max_attack_candidates
        raw = raw[: self.limits.max_attack_candidates]
        scored: list[tuple[int, int, int]] = []
        for action in raw:
            self._enter_node()
            if board.would_win(action, self.attacker):
                # Immediate wins are normally handled by the caller, but keep
                # this branch complete for direct use in recursion.
                scored.append((0, 0, action))
                continue
            after = board.with_move(action, self.attacker)
            immediate = after.winning_actions(self.attacker)
            if immediate:
                scored.append((1, -len(immediate), action))
                continue
            threes = self._forced_three_actions(after)
            if threes:
                scored.append((2, -len(threes), action))

        scored.sort()
        return tuple(item[2] for item in scored), truncated

    @staticmethod
    def _short_win_pv(
        board: FreestyleBoard,
        attacker: int,
    ) -> tuple[int, ...]:
        """Construct one concrete <=3-ply line from an exact short proof."""

        wins = board.winning_actions(attacker)
        if wins:
            return (wins[0],)
        forks = forced_win_in_three_actions(
            board,
            attacker,
            board_size=board.size,
            win_length=board.win_length,
        )
        if not forks:
            return ()
        attack = forks[0]
        after_attack = board.with_move(attack, attacker)
        threats = after_attack.winning_actions(attacker)
        if len(threats) < 2:  # Defensive assertion for the exact oracle.
            return ()
        defense = threats[0]
        after_defense = after_attack.with_move(defense, -attacker)
        finishes = after_defense.winning_actions(attacker)
        if not finishes:
            return ()
        return (attack, defense, finishes[0])

    def attack(self, board: FreestyleBoard, remaining: int) -> _ThreatOutcome:
        self._enter_node()
        cached = self._cached(board, self.attacker, remaining)
        if cached is not None:
            return cached

        if board.has_five(self.attacker):
            return self._remember(
                board,
                self.attacker,
                remaining,
                _ThreatOutcome(SolveStatus.PROVEN_WIN),
            )
        if board.has_five(self.defender) or remaining <= 0:
            return self._remember(
                board,
                self.attacker,
                remaining,
                _ThreatOutcome(SolveStatus.PROVEN_NO_VCT),
            )

        immediate = board.winning_actions(self.attacker)
        if immediate:
            return self._remember(
                board,
                self.attacker,
                remaining,
                _ThreatOutcome(SolveStatus.PROVEN_WIN, (immediate[0],)),
            )

        try:
            candidates, truncated = self._threat_candidates(board)
        except _BudgetExpired:
            return _ThreatOutcome(SolveStatus.UNKNOWN_BUDGET)

        saw_unknown = truncated
        for action in candidates:
            after = board.with_move(action, self.attacker)
            outcome = self.defend(after, remaining - 1)
            if outcome.status is SolveStatus.PROVEN_WIN:
                return self._remember(
                    board,
                    self.attacker,
                    remaining,
                    _ThreatOutcome(
                        SolveStatus.PROVEN_WIN,
                        (action,) + outcome.pv,
                    ),
                )
            if outcome.status is SolveStatus.UNKNOWN_BUDGET:
                saw_unknown = True

        status = (
            SolveStatus.UNKNOWN_BUDGET
            if saw_unknown
            else SolveStatus.PROVEN_NO_VCT
        )
        return self._remember(
            board,
            self.attacker,
            remaining,
            _ThreatOutcome(status),
        )

    def _safe_defenses(
        self,
        board: FreestyleBoard,
    ) -> tuple[tuple[int, ...], tuple[int, tuple[int, ...]] | None]:
        """Enumerate every reply that removes all exact <=3-ply threats.

        The second return value records one unsafe reply plus its concrete
        short proof, solely for a representative principal variation.
        """

        self._enter_node()
        safe = defenses_against_forced_win_in_three(
            board,
            self.defender,
            board_size=board.size,
            win_length=board.win_length,
        )
        # The exact helper may perform substantial pure-Python work between
        # checks, so enforce the wall-clock bound immediately on return.
        if self.limits.time_ms and self.elapsed_ms() >= self.limits.time_ms:
            raise _BudgetExpired
        if len(safe) > self.limits.max_defenses:
            raise _BudgetExpired

        safe_set = set(safe)
        example_unsafe: tuple[int, tuple[int, ...]] | None = None
        for defense in board.legal_actions():
            if defense in safe_set:
                continue
            self._enter_node()
            after = board.with_move(defense, self.defender)
            pv = self._short_win_pv(after, self.attacker)
            if pv:
                example_unsafe = (defense, pv)
                break
        return safe, example_unsafe

    def defend(self, board: FreestyleBoard, remaining: int) -> _ThreatOutcome:
        self._enter_node()
        cached = self._cached(board, self.defender, remaining)
        if cached is not None:
            return cached

        if board.has_five(self.attacker):
            return self._remember(
                board,
                self.defender,
                remaining,
                _ThreatOutcome(SolveStatus.PROVEN_WIN),
            )
        if board.has_five(self.defender) or remaining <= 0:
            return self._remember(
                board,
                self.defender,
                remaining,
                _ThreatOutcome(SolveStatus.PROVEN_NO_VCT),
            )

        # The opponent chooses an immediate win instead of answering a threat.
        if board.winning_actions(self.defender):
            return self._remember(
                board,
                self.defender,
                remaining,
                _ThreatOutcome(SolveStatus.PROVEN_NO_VCT),
            )

        immediate = board.winning_actions(self.attacker)
        if len(immediate) >= 2:
            if remaining < 2:
                outcome = _ThreatOutcome(SolveStatus.PROVEN_NO_VCT)
            else:
                defense = immediate[0]
                after = board.with_move(defense, self.defender)
                finishes = after.winning_actions(self.attacker)
                outcome = _ThreatOutcome(
                    SolveStatus.PROVEN_WIN,
                    (defense, finishes[0]),
                )
            return self._remember(board, self.defender, remaining, outcome)

        if len(immediate) == 1:
            defense = immediate[0]
            after = board.with_move(defense, self.defender)
            continuation = self.attack(after, remaining - 1)
            outcome = _ThreatOutcome(
                continuation.status,
                (defense,) + continuation.pv,
            )
            return self._remember(board, self.defender, remaining, outcome)

        # A VCT attacker move must now expose at least one exact three-ply
        # fork.  Otherwise it was not a forcing move in this solver's language.
        if not self._forced_three_actions(board):
            return self._remember(
                board,
                self.defender,
                remaining,
                _ThreatOutcome(SolveStatus.PROVEN_NO_VCT),
            )

        # Any reply not in ``safe`` leaves an exact <=3-ply win.  Four plies
        # are needed from this defender-to-move node (reply plus the proof).
        # With a shallower remaining bound we cannot safely compress those
        # branches, so report unknown rather than inventing a short proof.
        if remaining < 4:
            return _ThreatOutcome(SolveStatus.UNKNOWN_BUDGET)
        try:
            safe, example_unsafe = self._safe_defenses(board)
        except _BudgetExpired:
            return _ThreatOutcome(SolveStatus.UNKNOWN_BUDGET)

        if not safe:
            pv = ()
            if example_unsafe is not None:
                pv = (example_unsafe[0],) + example_unsafe[1]
            return self._remember(
                board,
                self.defender,
                remaining,
                _ThreatOutcome(SolveStatus.PROVEN_WIN, pv),
            )

        saw_unknown = False
        representative: tuple[int, ...] = ()
        for defense in safe:
            after = board.with_move(defense, self.defender)
            continuation = self.attack(after, remaining - 1)
            if continuation.status is SolveStatus.PROVEN_NO_VCT:
                return self._remember(
                    board,
                    self.defender,
                    remaining,
                    _ThreatOutcome(SolveStatus.PROVEN_NO_VCT),
                )
            if continuation.status is SolveStatus.UNKNOWN_BUDGET:
                saw_unknown = True
                continue
            candidate_pv = (defense,) + continuation.pv
            if len(candidate_pv) > len(representative):
                representative = candidate_pv

        outcome = _ThreatOutcome(
            SolveStatus.UNKNOWN_BUDGET if saw_unknown else SolveStatus.PROVEN_WIN,
            representative if not saw_unknown else (),
        )
        return self._remember(board, self.defender, remaining, outcome)


def solve_vct(
    board: FreestyleBoard | BoardLike,
    attacker: int,
    limits: ThreatSolveLimits | None = None,
    *,
    board_size: int = DEFAULT_BOARD_SIZE,
    win_length: int = DEFAULT_WIN_LENGTH,
) -> ThreatSolveResult:
    """Conservatively prove a bounded VCT/threat-space win.

    A positive result is a real minimax proof: every defender reply that
    removes the current immediate/three-ply threat is searched, and all other
    replies retain an exact short win.  This is intentionally incomplete for
    quiet setup moves.  A completed negative result therefore means only "no
    win in this bounded forcing language", never a solved Gomoku position.
    """

    _validate_player(attacker)
    position = _position(board, board_size, win_length)
    limits = limits or ThreatSolveLimits()
    if position.has_five(attacker) and position.has_five(-attacker):
        raise ValueError("invalid position: both players already have a winning line")

    search = _ThreatSearch(attacker, limits)
    if position.has_five(attacker):
        return ThreatSolveResult(
            SolveStatus.PROVEN_WIN, (), (), 1, 0, search.elapsed_ms()
        )
    if position.has_five(-attacker):
        return ThreatSolveResult(
            SolveStatus.PROVEN_NO_VCT, (), (), 1, 0, search.elapsed_ms()
        )

    try:
        search._enter_node()
        if limits.max_plies <= 0:
            outcome = _ThreatOutcome(SolveStatus.PROVEN_NO_VCT)
            candidates: tuple[int, ...] = ()
            truncated = False
        else:
            immediate = position.winning_actions(attacker)
            if immediate:
                return ThreatSolveResult(
                    SolveStatus.PROVEN_WIN,
                    immediate,
                    (immediate[0],),
                    search.nodes,
                    search.cache_hits,
                    search.elapsed_ms(),
                )
            candidates, truncated = search._threat_candidates(position)
            outcome = _ThreatOutcome(SolveStatus.PROVEN_NO_VCT)
    except _BudgetExpired:
        return ThreatSolveResult(
            SolveStatus.UNKNOWN_BUDGET,
            (),
            (),
            search.nodes,
            search.cache_hits,
            search.elapsed_ms(),
        )

    proven: list[int] = []
    best_pv: tuple[int, ...] = ()
    saw_unknown = truncated
    for action in candidates:
        after = position.with_move(action, attacker)
        try:
            branch = search.defend(after, limits.max_plies - 1)
        except _BudgetExpired:
            saw_unknown = True
            break
        if branch.status is SolveStatus.PROVEN_WIN:
            proven.append(action)
            pv = (action,) + branch.pv
            if not best_pv or len(pv) < len(best_pv):
                best_pv = pv
            # One sound root proof is sufficient for root guarding and keeps
            # the wall-clock cost bounded.  ``winning_actions`` is explicitly
            # documented as a proven subset, not an exhaustive move list.
            break
        elif branch.status is SolveStatus.UNKNOWN_BUDGET:
            saw_unknown = True

    if proven:
        status = SolveStatus.PROVEN_WIN
    elif saw_unknown:
        status = SolveStatus.UNKNOWN_BUDGET
    else:
        status = outcome.status
    return ThreatSolveResult(
        status,
        tuple(proven),
        best_pv,
        search.nodes,
        search.cache_hits,
        search.elapsed_ms(),
    )


class TacticalSolver:
    """Reusable rule-bound facade for repeated tactical queries."""

    def __init__(
        self,
        *,
        board_size: int = DEFAULT_BOARD_SIZE,
        win_length: int = DEFAULT_WIN_LENGTH,
    ) -> None:
        if board_size <= 0:
            raise ValueError("board_size must be positive")
        if win_length <= 1:
            raise ValueError("win_length must be at least two")
        self.board_size = board_size
        self.win_length = win_length

    def immediate_wins(self, board: FreestyleBoard | BoardLike, player: int) -> tuple[int, ...]:
        return immediate_winning_actions(
            board,
            player,
            board_size=self.board_size,
            win_length=self.win_length,
        )

    def forced_wins_in_three(
        self, board: FreestyleBoard | BoardLike, attacker: int
    ) -> tuple[int, ...]:
        return forced_win_in_three_actions(
            board,
            attacker,
            board_size=self.board_size,
            win_length=self.win_length,
        )

    def exact_defenses(
        self, board: FreestyleBoard | BoardLike, defender: int
    ) -> tuple[int, ...]:
        return defenses_against_forced_win_in_three(
            board,
            defender,
            board_size=self.board_size,
            win_length=self.win_length,
        )

    def solve_vcf(
        self,
        board: FreestyleBoard | BoardLike,
        attacker: int,
        limits: SolveLimits | None = None,
    ) -> SolveResult:
        return solve_vcf(
            board,
            attacker,
            limits,
            board_size=self.board_size,
            win_length=self.win_length,
        )

    def solve_vct(
        self,
        board: FreestyleBoard | BoardLike,
        attacker: int,
        limits: ThreatSolveLimits | None = None,
    ) -> ThreatSolveResult:
        return solve_vct(
            board,
            attacker,
            limits,
            board_size=self.board_size,
            win_length=self.win_length,
        )


__all__ = [
    "BLACK",
    "DEFAULT_BOARD_SIZE",
    "DEFAULT_WIN_LENGTH",
    "EMPTY",
    "FreestyleBoard",
    "SolveLimits",
    "SolveResult",
    "SolveStatus",
    "TacticalSolver",
    "ThreatSolveLimits",
    "ThreatSolveResult",
    "WHITE",
    "defenses_against_forced_win_in_three",
    "defenses_against_three_ply_win",
    "exact_defenses_against_win_in_three",
    "forced_win_in_three_actions",
    "immediate_winning_actions",
    "solve_vcf",
    "solve_vct",
    "winning_actions",
]
