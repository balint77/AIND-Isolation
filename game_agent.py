"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    raise NotImplementedError


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    raise NotImplementedError


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    raise NotImplementedError


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

min_log_level = 10000

def log(log_level, pause, game, *args):
    if log_level < min_log_level:
        return
    if game is not None:
        print(game.to_string())
    print(*args)
    if pause:
        input("Press Enter to continue...")


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        best_move = (-1, -1)
        best_score = float("-inf")
        tree_root = Node(best_move)
        legal_moves = game.get_legal_moves()
        for move in legal_moves:
            log(10, False, game, "X trying", move)
            try:
                node = Node(move, tree_root)
                score = self.minimax_traverse(game.forecast_move(move), depth - 1, node)
                node.score = score
                log(10, False, game, move,"X total score",score)
            except SearchTimeout:
                log(500, False, game, "X timed out")
                break
            if score > best_score:
                best_score = score
                best_move = move
                tree_root.score = best_score
                log(10, False, game, "X new max", best_move, best_score)
        tree_root.print_tree(100)
        log(100, False, game, "X chosen", best_move, best_score)
        if best_move not in legal_moves:
            log(100, False, None, "X wtf", best_move, best_score, legal_moves)
        return best_move

    def minimax_traverse(self, game, depth, parent_node):
        log(5, False, game, "evaluate", depth, game.active_player)
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if game.is_loser(self):
            log(10, False, game, "we would lose")
            return float("-inf")

        if game.is_winner(self):
            log(10, False, game, "we would win")
            return float("inf")

        if depth == 0:
            log(5, False, None, "depth limit reached",self.score(game, self))
            return self.score(game, self)

        max_level = True if self is game.active_player else False
        best_move = (-1, -1)
        best_score = float("-inf") if max_level else float("inf")
        legal_moves = game.get_legal_moves()
        for move in legal_moves:
            try:
                log(5, False, None, "traversing", move)
                node = Node(move, parent_node)
                score = self.minimax_traverse(game.forecast_move(move), depth - 1, node)
                node.score = score
                log(5, False, None, move, "scores", score)
            except SearchTimeout:
                log(500, False, game, "timed out")
                return best_score
            if max_level and score > best_score:
                best_score = score
                best_move = move
                log(5, False, None, "new max",best_move, best_score)
            if not max_level and score < best_score:
                best_score = score
                best_move = move
                log(5, False, None, "new min", best_move, best_score)
        return best_score

class Node:
    def __init__(self, move, parent=None, score=None, cut=False):
        self.move = move
        self.score = score
        self.cut = cut
        self.parent = parent
        self.children = []

        if parent:
            self.parent.children.append(self)

    def __str__(self):
        str = "-{} {}".format(self.move ,self.score)
        if self.cut:
            str += " CUT"
        return str + "-"

    def print_tree(self, log_level, indent="", last='updown'):

        if log_level < min_log_level:
            return

        nb_children = lambda node: sum(nb_children(child) for child in node.children) + 1
        size_branch = {child: nb_children(child) for child in self.children}

        """ Creation of balanced lists for "up" branch and "down" branch. """
        up = sorted(self.children, key=lambda node: nb_children(node))
        down = []
        while up and sum(size_branch[node] for node in down) < sum(size_branch[node] for node in up):
            down.append(up.pop())

        """ Printing of "up" branch. """
        for child in up:
            next_last = 'up' if up.index(child) is 0 else ''
            next_indent = '{0}{1}{2}'.format(indent, ' ' if 'up' in last else ':', " " * len(self.__str__()))
            child.print_tree(indent=next_indent, last=next_last)

        """ Printing of current node. """
        if last == 'up': start_shape = '/'
        elif last == 'down': start_shape = '\\'
        elif last == 'updown': start_shape = ' '
        else: start_shape = ':'

        if up: end_shape = ':'
        elif down: end_shape = '\\'
        else: end_shape = ''

        print('{0}{1}{2}{3}'.format(indent, start_shape, self, end_shape))

        """ Printing of "down" branch. """
        for child in down:
            next_last = 'down' if down.index(child) is len(down) - 1 else ''
            next_indent = '{0}{1}{2}'.format(indent, ' ' if 'down' in last else ':', " " * len(self.__str__()))
            child.print_tree(indent=next_indent, last=next_last)

class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left
        best_move = (-1,-1)
        try:
            search_depth = 0
            max_depth = len(game.get_blank_spaces())
            log(1000, False, None, "max depth", max_depth)
            while search_depth < max_depth :
                search_depth += 1
                best_move = self.alphabeta(game, search_depth)
            log(1000, False, game, "bottomed", search_depth, best_move)
        except SearchTimeout:
            log(1000, False, game, "timed out at depth", search_depth, best_move)
            pass
        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        best_move = (-1, -1)
        best_score = float("-inf")
        tree_root = Node(best_move)
        legal_moves = game.get_legal_moves()
        for move in legal_moves:
            log(10, False, game, "X trying", move)
            try:
                node = Node(move, tree_root)
                score = self.alphabeta_traverse(game.forecast_move(move), self.search_depth - 1, alpha, beta, parent_node=node)
                node.score = score
                log(10, False, None, move, "X total score", score)
                if score > best_score:
                    best_score = score
                    best_move = move
                    tree_root.score = best_score
                    log(10, False, None, "X new max", best_move, best_score)
                    if best_score >= beta:
                        node.cut = True
                        break
                    alpha = max(alpha, best_score)
            except SearchTimeout:
                log(500, False, game, "X timed out")
                break
        tree_root.print_tree(100)
        if best_move not in legal_moves:
            log(100, False, None, "X wtf", best_move, best_score, legal_moves)
        log(100, False, game, "X chosen", best_move, best_score)
        return best_move

    def alphabeta_traverse(self, game, depth, alpha=float("-inf"), beta=float("inf"), parent_node=None):
        """Similar to alphabeta, but log decision moves and scores into a tree linked on parent_node"""

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if game.is_loser(self):
            return float("-inf")

        if game.is_winner(self):
            return float("inf")

        if depth == 0:
            return self.score(game, self)

        max_level = True if self is game.active_player else False
        best_move = (-1, -1)
        best_score = float("-inf") if max_level else float("inf")
        legal_moves = game.get_legal_moves()
        try:
            for move in legal_moves:
                node = Node(move, parent_node)
                score = self.alphabeta_traverse(game.forecast_move(move), depth - 1, alpha, beta, node)
                node.score = score
                if max_level and score > best_score:
                    best_score = score
                    best_move = move
                    if best_score >= beta:
                        node.cut = True
                        break
                    alpha = max(alpha, best_score)
                if not max_level and score < best_score:
                    best_score = score
                    best_move = move
                    if best_score <= alpha:
                        node.cut = True
                        break
                    beta = min(beta, best_score)
        except SearchTimeout:
            log(500, False, game, "timed out")
            pass
        return best_score